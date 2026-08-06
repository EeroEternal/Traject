//! Integrate the Zene coding agent into Traject.
//!
//! Architecture:
//! - Vendored Zene owns agent semantics (ReAct loop, tools, permissions, plan mode)
//! - Traject owns Trajectory ledger, MemoryManager prefix tree, and inference
//!   (`TrajectLlmProvider` → sglang-lite engine with session/prefix metadata)

mod provider;
mod runner;

pub use provider::{TrajectLlmProvider, TrajectSession};
pub use runner::{ZeneRunConfig, ZeneRunner, ZeneRunResult};
pub use zene_config::AgentProfile;
