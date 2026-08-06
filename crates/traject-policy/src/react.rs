use async_trait::async_trait;
use traject_core::{Constraints, FinishReason, GenerateDelta, Result, Step, StepOutcome, Trajectory};

use crate::{Policy, PolicyDecision};

/// Minimal ReAct: Generate → (optional Tool) → Generate → … → Finish.
#[derive(Debug, Clone, Default)]
pub struct ReActPolicy {
    pub max_steps: usize,
    pub system_prompt: String,
}

impl ReActPolicy {
    pub fn new(system_prompt: impl Into<String>) -> Self {
        Self {
            max_steps: 16,
            system_prompt: system_prompt.into(),
        }
    }
}

#[async_trait]
impl Policy for ReActPolicy {
    fn name(&self) -> &str {
        "react"
    }

    async fn decide(&self, traj: &Trajectory) -> Result<PolicyDecision> {
        if traj.history.len() >= self.max_steps {
            return Ok(PolicyDecision::Finish);
        }
        let delta = if traj.history.is_empty() {
            GenerateDelta::from_text(self.system_prompt.clone())
        } else {
            GenerateDelta::from_text("")
        };
        Ok(PolicyDecision::NextStep(Step::generate(
            delta,
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
            StepOutcome::Generated {
                finish_reason: FinishReason::ToolCall,
                tool_call: Some(call),
                ..
            } => Ok(PolicyDecision::NextStep(Step::tool(call.clone(), 30_000))),
            StepOutcome::Generated {
                finish_reason: FinishReason::Stop,
                ..
            } => Ok(PolicyDecision::Finish),
            StepOutcome::Generated { .. } => Ok(PolicyDecision::NextStep(Step::generate(
                GenerateDelta::default(),
                Constraints::default(),
                256,
            ))),
            StepOutcome::ToolDone { result, .. } => {
                let delta = GenerateDelta::from_text(format!(
                    "tool {} => {}",
                    result.name, result.output
                ));
                Ok(PolicyDecision::NextStep(Step::generate(
                    delta,
                    Constraints::default(),
                    256,
                )))
            }
            StepOutcome::ControlDone { .. } => self.decide(traj).await,
        }
    }
}

/// Helper to build a tool call step from raw fields (used by tests / drivers).
#[allow(dead_code)]
fn tool_step(name: &str, arguments: &str) -> Step {
    use traject_core::ToolCall;
    Step::tool(
        ToolCall {
            name: name.into(),
            arguments: arguments.into(),
            call_id: None,
        },
        30_000,
    )
}
