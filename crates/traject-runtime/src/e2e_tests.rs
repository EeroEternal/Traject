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

    #[tokio::test]
    async fn external_generate_and_tool_via_driver() {
        use traject_core::{
            Constraints, GenerateDelta, ToolCall, ToolResult, TrajectoryConfig,
        };

        let mut driver = Driver::new(DriverConfig::default())
            .with_stub_mode(StubMode::AlwaysStop);
        let id = driver.create_external_trajectory(TrajectoryConfig::default());
        assert!(driver.is_external(id));

        let outcome = driver
            .run_generate_step(
                id,
                GenerateDelta::from_text("hello agent"),
                Constraints::default(),
                32,
            )
            .await
            .unwrap();
        assert!(matches!(
            outcome,
            traject_core::StepOutcome::Generated { .. }
        ));

        driver
            .run_external_tool_step(
                id,
                ToolCall {
                    name: "Glob".into(),
                    arguments: "{}".into(),
                    call_id: Some("c1".into()),
                },
                ToolResult {
                    call_id: Some("c1".into()),
                    name: "Glob".into(),
                    output: "a.txt".into(),
                    is_error: false,
                },
                5_000,
            )
            .unwrap();

        let traj = driver.manager.get(id).unwrap();
        assert_eq!(traj.history.len(), 2);
        assert!(driver.memory.binding(id).is_some());
        assert!(driver.memory.engine_prefix_hint(id).is_some());

        driver.finish_trajectory(id).unwrap();
    }

    #[tokio::test]
    async fn local_weight_runner_trajectory() {
        use traject_inference::LocalWeightConfig;
        use traject_policy::ReActPolicy;

        let mut driver = Driver::new(DriverConfig {
            max_ticks: 64,
            ..DriverConfig::default()
        })
        .with_policy(Arc::new(ReActPolicy {
            max_steps: 2,
            system_prompt: "hi".into(),
        }))
        .with_local_weight_runner(LocalWeightConfig {
            max_new_tokens_default: 8,
            ..LocalWeightConfig::default()
        });
        let id = driver.create_trajectory(TrajectoryConfig::default());
        driver.run_until_finished(id).await.unwrap();
        let traj = driver.manager.get(id).unwrap();
        assert!(traj.state.is_terminal());
        assert!(!traj.history.is_empty());
    }

    #[tokio::test]
    async fn tool_latency_influences_pin_and_prefetch() {
        use traject_core::{
            Constraints, GenerateDelta, PinReason, ToolCall, ToolResult, TrajectoryConfig,
        };

        let mut driver = Driver::new(DriverConfig::default())
            .with_stub_mode(StubMode::AlwaysStop);
        let id = driver.create_external_trajectory(TrajectoryConfig::default());

        // Simulate tool gap pin start + slow tool.
        driver.pin_for_tool_gap(id).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(30));
        driver
            .run_external_tool_step(
                id,
                ToolCall {
                    name: "Glob".into(),
                    arguments: "{}".into(),
                    call_id: Some("c1".into()),
                },
                ToolResult {
                    call_id: Some("c1".into()),
                    name: "Glob".into(),
                    output: "a.txt".into(),
                    is_error: false,
                },
                5_000,
            )
            .unwrap();

        // Prefetch pin should be active after tool.
        let traj = driver.manager.get(id).unwrap();
        assert_eq!(traj.pin.reason, PinReason::Prefetch);
        assert!(traj.pin.pin_until_ms.is_some());

        // Second sample builds histogram.
        driver.pin_for_tool_gap(id).unwrap();
        driver
            .run_external_tool_step(
                id,
                ToolCall {
                    name: "Glob".into(),
                    arguments: "{}".into(),
                    call_id: Some("c2".into()),
                },
                ToolResult {
                    call_id: Some("c2".into()),
                    name: "Glob".into(),
                    output: "b.txt".into(),
                    is_error: false,
                },
                5_000,
            )
            .unwrap();

        let _ = driver
            .run_generate_step(
                id,
                GenerateDelta::from_text("after tools"),
                Constraints::default(),
                16,
            )
            .await
            .unwrap();
        driver.finish_trajectory(id).unwrap();
    }
}
