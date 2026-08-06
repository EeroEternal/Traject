use traject_core::{Result, Trajectory, TrajectoryState, TrajectError};

/// Explicit transition intents used by the manager / driver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    Start,
    BeginToolWait,
    ToolReturned,
    Suspend,
    Resume,
    Finish,
    Fail,
}

pub fn apply_transition(traj: &mut Trajectory, t: Transition) -> Result<()> {
    let to = match (traj.state, t) {
        (TrajectoryState::Created, Transition::Start) => TrajectoryState::Running,
        (TrajectoryState::Running, Transition::BeginToolWait) => TrajectoryState::WaitingTool,
        (TrajectoryState::WaitingTool, Transition::ToolReturned) => TrajectoryState::Running,
        (TrajectoryState::Running | TrajectoryState::WaitingTool, Transition::Suspend) => {
            TrajectoryState::Suspended
        }
        (TrajectoryState::Suspended, Transition::Resume) => TrajectoryState::Running,
        (
            TrajectoryState::Running | TrajectoryState::WaitingTool | TrajectoryState::Suspended,
            Transition::Finish,
        ) => TrajectoryState::Finished,
        (
            TrajectoryState::Created
            | TrajectoryState::Running
            | TrajectoryState::WaitingTool
            | TrajectoryState::Suspended,
            Transition::Fail,
        ) => TrajectoryState::Failed,
        (from, _) => {
            return Err(TrajectError::InvalidTransition {
                from,
                to: from, // placeholder; detailed mapping below
            });
        }
    };
    traj.transition(to)
}

#[cfg(test)]
mod tests {
    use super::*;
    use traject_core::TrajectoryConfig;

    #[test]
    fn start_and_finish() {
        let mut t = Trajectory::create(TrajectoryConfig::default());
        apply_transition(&mut t, Transition::Start).unwrap();
        apply_transition(&mut t, Transition::Finish).unwrap();
        assert!(t.is_finished());
    }
}
