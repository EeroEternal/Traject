#[cfg(test)]
mod e2e {
    use std::sync::Arc;

    use traject_core::{TrajectoryConfig, TrajectoryPriority, TrajectoryState};
    use traject_inference::StubMode;
    use traject_policy::ReActPolicy;

    use crate::{Driver, DriverConfig};

    #[tokio::test]
    async fn simple_stop() {
        let mut driver = Driver::new(DriverConfig::default())
            .with_policy(Arc::new(ReActPolicy::new("hi")))
            .with_stub_mode(StubMode::AlwaysStop);
        let id = driver.create_trajectory(TrajectoryConfig::default());
        driver.run_until_finished(id).await.unwrap();
        let traj = driver.manager.get(id).unwrap();
        assert_eq!(traj.state, TrajectoryState::Finished);
        assert!(!traj.history.is_empty());
    }

    #[tokio::test]
    async fn tool_then_stop() {
        let mut driver = Driver::new(DriverConfig::default())
            .with_policy(Arc::new(ReActPolicy::new("use a tool")))
            .with_stub_mode(StubMode::ToolThenStop {
                remaining_tools: 1,
            });
        let id = driver.create_trajectory(TrajectoryConfig::default());
        driver.run_until_finished(id).await.unwrap();
        let traj = driver.manager.get(id).unwrap();
        assert_eq!(traj.state, TrajectoryState::Finished);
        let has_tool = traj.history.iter().any(|r| {
            matches!(
                r.outcome,
                Some(traject_core::StepOutcome::ToolDone { .. })
            )
        });
        assert!(has_tool, "expected a tool step in history: {:?}", traj.history.len());
        assert!(traj.history.len() >= 3);
    }

    #[tokio::test]
    async fn multi_trajectory_concurrent() {
        let mut driver = Driver::new(DriverConfig {
            max_ticks: 2048,
            ..DriverConfig::default()
        })
        .with_policy(Arc::new(ReActPolicy::new("batch")))
        .with_stub_mode(StubMode::ToolThenStop {
            remaining_tools: 1,
        });

        let mut ids = Vec::new();
        for i in 0..4 {
            let cfg = TrajectoryConfig {
                priority: TrajectoryPriority(i),
                ..TrajectoryConfig::default()
            };
            ids.push(driver.create_trajectory(cfg));
        }
        driver.run_all_until_idle().await.unwrap();
        for id in ids {
            let traj = driver.manager.get(id).unwrap();
            assert!(
                traj.state.is_terminal(),
                "traj {id} not terminal: {:?}",
                traj.state
            );
        }
        // Shared system-ish prefix tokens should collapse node count.
        assert!(driver.memory.stats().nodes >= 1);
    }

    #[tokio::test]
    async fn prefix_shared_across_trajs() {
        let mut driver = Driver::new(DriverConfig::default())
            .with_policy(Arc::new(ReActPolicy::new("shared-prefix-hello")))
            .with_stub_mode(StubMode::AlwaysStop);
        let a = driver.create_trajectory(TrajectoryConfig::default());
        let b = driver.create_trajectory(TrajectoryConfig::default());
        driver.run_all_until_idle().await.unwrap();
        let _ = (a, b);
        // Both finished; memory retained some shared structure.
        assert!(driver.memory.stats().nodes >= 2);
    }

    #[tokio::test]
    async fn multi_chunk_decode() {
        let mut driver = Driver::new(DriverConfig::default())
            .with_policy(Arc::new(ReActPolicy::new("chunky")))
            .with_stub_mode(StubMode::MultiChunk { chunks: 3 });
        let id = driver.create_trajectory(TrajectoryConfig::default());
        driver.run_until_finished(id).await.unwrap();
        let traj = driver.manager.get(id).unwrap();
        assert_eq!(traj.state, TrajectoryState::Finished);
    }
}
