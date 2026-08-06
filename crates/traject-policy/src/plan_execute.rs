//! Plan-Execute style policy (Phase 1 sketch).

use async_trait::async_trait;
use traject_core::{
    Constraints, ControlKind, FinishReason, GenerateDelta, Result, Step, StepOutcome, Trajectory,
};

use crate::{Policy, PolicyDecision};

/// Two-phase policy: emit a Plan control step once, then ReAct-like generate/tool loop.
#[derive(Debug)]
pub struct PlanExecutePolicy {
    pub max_steps: usize,
    pub goal: String,
    planned: std::sync::atomic::AtomicBool,
}

impl PlanExecutePolicy {
    pub fn new(goal: impl Into<String>) -> Self {
        Self {
            max_steps: 16,
            goal: goal.into(),
            planned: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

#[async_trait]
impl Policy for PlanExecutePolicy {
    fn name(&self) -> &str {
        "plan-execute"
    }

    async fn decide(&self, traj: &Trajectory) -> Result<PolicyDecision> {
        if traj.history.len() >= self.max_steps {
            return Ok(PolicyDecision::Finish);
        }
        if !self
            .planned
            .swap(true, std::sync::atomic::Ordering::SeqCst)
        {
            return Ok(PolicyDecision::NextStep(Step::control(
                ControlKind::Plan,
                Some(format!("plan for: {}", self.goal)),
            )));
        }
        Ok(PolicyDecision::NextStep(Step::generate(
            GenerateDelta::from_text(self.goal.clone()),
            Constraints::default(),
            256,
        )))
    }

    async fn on_outcome(
        &self,
        traj: &Trajectory,
        outcome: &StepOutcome,
    ) -> Result<PolicyDecision> {
        if traj.history.len() >= self.max_steps {
            return Ok(PolicyDecision::Finish);
        }
        match outcome {
            StepOutcome::ControlDone {
                kind: ControlKind::Plan,
                ..
            } => Ok(PolicyDecision::NextStep(Step::generate(
                GenerateDelta::from_text(format!("execute plan: {}", self.goal)),
                Constraints::default(),
                256,
            ))),
            StepOutcome::Generated {
                finish_reason: FinishReason::ToolCall,
                tool_call: Some(call),
                ..
            } => Ok(PolicyDecision::NextStep(Step::tool(call.clone(), 30_000))),
            StepOutcome::Generated {
                finish_reason: FinishReason::Stop,
                ..
            } => Ok(PolicyDecision::Finish),
            StepOutcome::ToolDone { result, .. } => Ok(PolicyDecision::NextStep(Step::generate(
                GenerateDelta::from_text(format!("tool {} => {}", result.name, result.output)),
                Constraints::default(),
                256,
            ))),
            _ => self.decide(traj).await,
        }
    }
}
