use traject_core::TrajectoryConfig;
use traject_policy::ReActPolicy;
use traject_runtime::{Driver, DriverConfig};

#[tokio::main]
async fn main() -> traject_core::Result<()> {
    let mut driver = Driver::new(DriverConfig::default())
        .with_policy(std::sync::Arc::new(ReActPolicy::new("Hello from Traject")));
    let id = driver.create_trajectory(TrajectoryConfig::default());
    driver.run_until_finished(id).await?;
    let traj = driver.manager.get(id)?;
    println!(
        "finished trajectory {id} after {} steps",
        traj.history.len()
    );
    Ok(())
}
