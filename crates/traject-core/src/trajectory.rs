use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    AgentMemory, PinInfo, PrefixNodeId, Result, Step, StepId, StepOutcome, TrajectError,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TrajectoryId(pub Uuid);

impl TrajectoryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for TrajectoryId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for TrajectoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(pub String);

impl TenantId {
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Lifecycle of a Trajectory.
///
/// ```text
/// Created → Running ⇄ WaitingTool → Running → Finished
///                 ↘ Failed
///                 ↘ Suspended ⇄ Running
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrajectoryState {
    Created,
    Running,
    WaitingTool,
    Suspended,
    Finished,
    Failed,
}

impl TrajectoryState {
    pub fn is_terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Failed)
    }

    pub fn can_schedule(self) -> bool {
        matches!(self, Self::Running | Self::WaitingTool)
    }
}

/// Scheduling priority for a Trajectory (higher = preferred).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct TrajectoryPriority(pub i32);

impl Default for TrajectoryPriority {
    fn default() -> Self {
        Self(0)
    }
}

/// Creation parameters for a new Trajectory.
#[derive(Debug, Clone)]
pub struct TrajectoryConfig {
    pub tenant: TenantId,
    pub priority: TrajectoryPriority,
    /// Optional stable system/tools prefix already resident in the tree.
    pub stable_prefix: Option<PrefixNodeId>,
    pub initial_memory: AgentMemory,
}

impl Default for TrajectoryConfig {
    fn default() -> Self {
        Self {
            tenant: TenantId::new("default"),
            priority: TrajectoryPriority::default(),
            stable_prefix: None,
            initial_memory: AgentMemory::default(),
        }
    }
}

/// First-class Agent execution process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trajectory {
    pub id: TrajectoryId,
    pub tenant_id: TenantId,
    pub state: TrajectoryState,
    pub current_prefix: Option<PrefixNodeId>,
    pub memory: AgentMemory,
    pub priority: TrajectoryPriority,
    /// Accumulated fairness credit for anti-starvation.
    pub fairness_credit: i64,
    pub pin: PinInfo,
    pub history: Vec<StepRecord>,
    pub active_step: Option<Step>,
    pub error: Option<String>,
}

/// Immutable record of a completed (or in-flight) step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepRecord {
    pub step: Step,
    pub outcome: Option<StepOutcome>,
}

impl Trajectory {
    pub fn create(config: TrajectoryConfig) -> Self {
        Self {
            id: TrajectoryId::new(),
            tenant_id: config.tenant,
            state: TrajectoryState::Created,
            current_prefix: config.stable_prefix,
            memory: config.initial_memory,
            priority: config.priority,
            fairness_credit: 0,
            pin: PinInfo::default(),
            history: Vec::new(),
            active_step: None,
            error: None,
        }
    }

    pub fn is_finished(&self) -> bool {
        self.state.is_terminal()
    }

    /// Attempt a legal state transition.
    pub fn transition(&mut self, to: TrajectoryState) -> Result<()> {
        if !is_legal_transition(self.state, to) {
            return Err(TrajectError::InvalidTransition {
                from: self.state,
                to,
            });
        }
        self.state = to;
        Ok(())
    }

    pub fn start(&mut self) -> Result<()> {
        self.transition(TrajectoryState::Running)
    }

    pub fn submit_step(&mut self, step: Step) -> Result<()> {
        if self.state == TrajectoryState::Created {
            self.start()?;
        }
        if !matches!(
            self.state,
            TrajectoryState::Running | TrajectoryState::WaitingTool | TrajectoryState::Suspended
        ) {
            return Err(TrajectError::InvalidTransition {
                from: self.state,
                to: TrajectoryState::Running,
            });
        }
        if matches!(self.state, TrajectoryState::Suspended) {
            self.transition(TrajectoryState::Running)?;
        }
        if step.is_tool() {
            self.transition(TrajectoryState::WaitingTool)?;
        } else if self.state == TrajectoryState::WaitingTool && step.is_generate() {
            self.transition(TrajectoryState::Running)?;
        }
        self.active_step = Some(step);
        Ok(())
    }

    pub fn complete_step(&mut self, outcome: StepOutcome) -> Result<()> {
        let step = self
            .active_step
            .take()
            .ok_or_else(|| TrajectError::Other("no active step to complete".into()))?;

        let step_id = match &outcome {
            StepOutcome::Generated { step_id, .. }
            | StepOutcome::ToolDone { step_id, .. }
            | StepOutcome::ControlDone { step_id, .. } => *step_id,
        };
        if step.id() != step_id {
            return Err(TrajectError::StepNotFound(step_id));
        }

        if let StepOutcome::Generated {
            tool_call: Some(_),
            ..
        } = &outcome
        {
            // Generate ended in a tool call → expect WaitingTool next; stay Running
            // until Tool step is submitted.
        }

        if matches!(outcome, StepOutcome::ToolDone { .. })
            && self.state == TrajectoryState::WaitingTool
        {
            self.transition(TrajectoryState::Running)?;
        }

        self.history.push(StepRecord {
            step,
            outcome: Some(outcome),
        });
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        self.active_step = None;
        self.pin.unpin();
        self.transition(TrajectoryState::Finished)
    }

    pub fn fail(&mut self, reason: impl Into<String>) -> Result<()> {
        self.error = Some(reason.into());
        self.active_step = None;
        self.pin.unpin();
        self.transition(TrajectoryState::Failed)
    }

    pub fn suspend(&mut self) -> Result<()> {
        self.transition(TrajectoryState::Suspended)
    }

    pub fn resume(&mut self) -> Result<()> {
        self.transition(TrajectoryState::Running)
    }

    pub fn last_outcome(&self) -> Option<&StepOutcome> {
        self.history
            .iter()
            .rev()
            .find_map(|r| r.outcome.as_ref())
    }

    pub fn bind_prefix(&mut self, node: PrefixNodeId) {
        self.current_prefix = Some(node);
    }

    pub fn active_step_id(&self) -> Option<StepId> {
        self.active_step.as_ref().map(|s| s.id())
    }
}

fn is_legal_transition(from: TrajectoryState, to: TrajectoryState) -> bool {
    use TrajectoryState::*;
    matches!(
        (from, to),
        (Created, Running)
            | (Running, WaitingTool)
            | (WaitingTool, Running)
            | (Running, Suspended)
            | (Suspended, Running)
            | (WaitingTool, Suspended)
            | (Suspended, WaitingTool)
            | (Running, Finished)
            | (WaitingTool, Finished)
            | (Suspended, Finished)
            | (Running, Failed)
            | (WaitingTool, Failed)
            | (Suspended, Failed)
            | (Created, Failed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Constraints, ControlKind, GenerateDelta, ToolCall};

    #[test]
    fn lifecycle_happy_path() {
        let mut t = Trajectory::create(TrajectoryConfig::default());
        assert_eq!(t.state, TrajectoryState::Created);

        t.submit_step(Step::generate(
            GenerateDelta::from_text("hi"),
            Constraints::default(),
            64,
        ))
        .unwrap();
        assert_eq!(t.state, TrajectoryState::Running);

        let sid = t.active_step_id().unwrap();
        t.complete_step(StepOutcome::Generated {
            step_id: sid,
            text: "call tool".into(),
            token_ids: vec![],
            finish_reason: crate::step::FinishReason::ToolCall,
            tool_call: Some(ToolCall {
                name: "search".into(),
                arguments: "{}".into(),
                call_id: None,
            }),
        })
        .unwrap();

        t.submit_step(Step::tool(
            ToolCall {
                name: "search".into(),
                arguments: "{}".into(),
                call_id: None,
            },
            5_000,
        ))
        .unwrap();
        assert_eq!(t.state, TrajectoryState::WaitingTool);

        let sid = t.active_step_id().unwrap();
        t.complete_step(StepOutcome::ToolDone {
            step_id: sid,
            result: crate::ToolResult {
                call_id: None,
                name: "search".into(),
                output: "ok".into(),
                is_error: false,
            },
        })
        .unwrap();
        assert_eq!(t.state, TrajectoryState::Running);

        t.finish().unwrap();
        assert!(t.is_finished());
    }

    #[test]
    fn illegal_transition_rejected() {
        let mut t = Trajectory::create(TrajectoryConfig::default());
        let err = t.transition(TrajectoryState::Finished).unwrap_err();
        assert!(matches!(err, TrajectError::InvalidTransition { .. }));
    }

    #[test]
    fn control_step_keeps_running() {
        let mut t = Trajectory::create(TrajectoryConfig::default());
        t.submit_step(Step::control(ControlKind::Reflect, None))
            .unwrap();
        assert_eq!(t.state, TrajectoryState::Running);
    }
}
