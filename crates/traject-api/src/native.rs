use std::sync::Arc;

use traject_core::{
    Constraints, GenerateDelta, Result, Step, ToolCall, ToolResult, TrajectoryConfig, TrajectoryId,
    TrajectoryState,
};
use traject_policy::Policy;
use traject_runtime::{Driver, DriverConfig};

/// Process-wide handle that owns a Driver (Phase 0: in-process).
pub struct Traject {
    driver: Driver,
}

impl Traject {
    pub fn new() -> Self {
        Self {
            driver: Driver::new(DriverConfig::default()),
        }
    }

    pub fn with_policy(mut self, policy: Arc<dyn Policy>) -> Self {
        self.driver = self.driver.with_policy(policy);
        self
    }

    pub fn create_handle(mut self, config: TrajectoryConfig) -> TrajectoryHandle {
        let id = self.driver.create_trajectory(config);
        TrajectoryHandle {
            id,
            driver: self.driver,
        }
    }
}

impl Default for Traject {
    fn default() -> Self {
        Self::new()
    }
}

/// Native per-trajectory handle.
pub struct TrajectoryHandle {
    pub id: TrajectoryId,
    driver: Driver,
}

impl TrajectoryHandle {
    pub fn id(&self) -> TrajectoryId {
        self.id
    }

    pub fn is_finished(&self) -> Result<bool> {
        self.driver.manager.is_finished(self.id)
    }

    pub fn state(&self) -> Result<TrajectoryState> {
        Ok(self.driver.manager.get(self.id)?.state)
    }

    /// Run until the trajectory reaches a terminal state.
    pub async fn step(&mut self) -> Result<()> {
        self.driver.run_until_finished(self.id).await
    }

    pub async fn generate(
        &mut self,
        delta: GenerateDelta,
        constraints: Constraints,
        max_tokens: u32,
    ) -> Result<()> {
        self.driver
            .manager
            .submit_step(self.id, Step::generate(delta, constraints, max_tokens))?;
        self.driver.run_until_finished(self.id).await
    }

    pub async fn execute_tool(&mut self, call: ToolCall) -> Result<Option<ToolResult>> {
        self.driver
            .manager
            .submit_step(self.id, Step::tool(call, 30_000))?;
        self.driver.run_until_finished(self.id).await?;
        let traj = self.driver.manager.get(self.id)?;
        Ok(traj.last_outcome().and_then(|o| match o {
            traject_core::StepOutcome::ToolDone { result, .. } => Some(result.clone()),
            _ => None,
        }))
    }
}
