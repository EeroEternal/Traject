//! Trajectory Manager + driver main loop.

mod driver;
mod e2e_tests;
mod manager;
mod state_machine;

pub use driver::{run_simple_react, Driver, DriverConfig};
pub use manager::TrajectoryManager;
pub use state_machine::{apply_transition, Transition};
