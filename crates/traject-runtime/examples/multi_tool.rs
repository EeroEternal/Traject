//! Multi-trajectory + tool gap demo (stub backend).
//!
//! ```bash
//! cargo run -p traject-runtime --example multi_tool
//! ```

use std::sync::Arc;

use traject_core::{TrajectoryConfig, TrajectoryPriority};
use traject_inference::StubMode;
use traject_policy::ReActPolicy;
use traject_runtime::{Driver, DriverConfig};

#[tokio::main]
async fn main() -> traject_core::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    let mut driver = Driver::new(DriverConfig::default())
        .with_policy(Arc::new(ReActPolicy::new("multi-tool demo")))
        .with_stub_mode(StubMode::ToolThenStop {
            remaining_tools: 2,
        });

    for i in 0..3 {
        driver.create_trajectory(TrajectoryConfig {
            priority: TrajectoryPriority(i * 10),
            ..TrajectoryConfig::default()
        });
    }

    driver.run_all_until_idle().await?;

    println!(
        "done: {} trajectories, memory nodes={}, blocks={}/{}",
        driver.manager.len(),
        driver.memory.stats().nodes,
        driver.memory.stats().blocks_allocated,
        driver.memory.stats().blocks_capacity
    );
    Ok(())
}
