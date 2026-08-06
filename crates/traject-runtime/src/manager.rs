use std::sync::Arc;

use dashmap::DashMap;
use traject_core::{
    Result, Step, StepOutcome, Trajectory, TrajectoryConfig, TrajectoryId, TrajectError,
};
use traject_policy::{Policy, PolicyDecision};

use crate::state_machine::{apply_transition, Transition};

/// Owns all live Trajectories and coordinates policy → step submission.
pub struct TrajectoryManager {
    trajectories: DashMap<TrajectoryId, Trajectory>,
}

impl TrajectoryManager {
    pub fn new() -> Self {
        Self {
            trajectories: DashMap::new(),
        }
    }

    pub fn create(&self, config: TrajectoryConfig) -> TrajectoryId {
        let traj = Trajectory::create(config);
        let id = traj.id;
        self.trajectories.insert(id, traj);
        id
    }

    pub fn get(&self, id: TrajectoryId) -> Result<Trajectory> {
        self.trajectories
            .get(&id)
            .map(|t| t.clone())
            .ok_or(TrajectError::TrajectoryNotFound(id))
    }

    pub fn with_mut<R>(
        &self,
        id: TrajectoryId,
        f: impl FnOnce(&mut Trajectory) -> R,
    ) -> Result<R> {
        let mut entry = self
            .trajectories
            .get_mut(&id)
            .ok_or(TrajectError::TrajectoryNotFound(id))?;
        Ok(f(&mut entry))
    }

    pub fn start(&self, id: TrajectoryId) -> Result<()> {
        self.with_mut(id, |t| apply_transition(t, Transition::Start))?
    }

    pub fn submit_step(&self, id: TrajectoryId, step: Step) -> Result<()> {
        self.with_mut(id, |t| t.submit_step(step))?
    }

    pub fn complete_step(&self, id: TrajectoryId, outcome: StepOutcome) -> Result<()> {
        self.with_mut(id, |t| t.complete_step(outcome))?
    }

    pub fn finish(&self, id: TrajectoryId) -> Result<()> {
        self.with_mut(id, |t| t.finish())?
    }

    pub fn fail(&self, id: TrajectoryId, reason: impl Into<String>) -> Result<()> {
        let reason = reason.into();
        self.with_mut(id, |t| t.fail(reason))?
    }

    pub fn is_finished(&self, id: TrajectoryId) -> Result<bool> {
        Ok(self.get(id)?.is_finished())
    }

    pub fn remove(&self, id: TrajectoryId) -> Option<Trajectory> {
        self.trajectories.remove(&id).map(|(_, t)| t)
    }

    pub fn len(&self) -> usize {
        self.trajectories.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trajectories.is_empty()
    }

    /// Ask policy for the next action and apply Finish/Fail immediately.
    pub async fn advance_with_policy(
        &self,
        id: TrajectoryId,
        policy: &Arc<dyn Policy>,
    ) -> Result<Option<Step>> {
        let traj = self.get(id)?;
        let decision = if let Some(outcome) = traj.last_outcome() {
            // Only consult on_outcome when there is no active step pending.
            if traj.active_step.is_some() {
                return Ok(traj.active_step.clone());
            }
            policy.on_outcome(&traj, outcome).await?
        } else {
            policy.decide(&traj).await?
        };

        match decision {
            PolicyDecision::NextStep(step) => {
                self.submit_step(id, step.clone())?;
                Ok(Some(step))
            }
            PolicyDecision::Finish => {
                self.finish(id)?;
                Ok(None)
            }
            PolicyDecision::Fail(reason) => {
                self.fail(id, reason)?;
                Ok(None)
            }
        }
    }
}

impl Default for TrajectoryManager {
    fn default() -> Self {
        Self::new()
    }
}
