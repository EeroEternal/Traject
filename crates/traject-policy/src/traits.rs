use async_trait::async_trait;
use traject_core::{Result, Step, StepOutcome, Trajectory};

/// What the policy wants to do next on a Trajectory.
#[derive(Debug, Clone)]
pub enum PolicyDecision {
    /// Submit this step to the scheduler.
    NextStep(Step),
    /// Trajectory is complete.
    Finish,
    /// Hard failure.
    Fail(String),
}

/// Agent logic that turns trajectory state into the next Step.
#[async_trait]
pub trait Policy: Send + Sync {
    fn name(&self) -> &str;

    /// Called when a trajectory is created / resumed with no active step.
    async fn decide(&self, traj: &Trajectory) -> Result<PolicyDecision>;

    /// Called after a step completes to choose the follow-up.
    async fn on_outcome(
        &self,
        traj: &Trajectory,
        outcome: &StepOutcome,
    ) -> Result<PolicyDecision>;
}
