//! Integrate the Zene coding agent into Traject.
//!
//! Architecture:
//! - Vendored Zene owns agent semantics (ReAct loop, tools, permissions, plan mode)
//! - Every LLM / Tool step goes through Traject `Driver` → `Scheduler` →
//!   `MemoryManager` + `InferenceEngine` (sglang-lite with session/prefix metadata)
//! - `--legacy-http` / tool-bridge remains a demoted compatibility path only

mod provider;
mod runner;

pub use provider::{TrajectLlmProvider, TrajectSession};
pub use runner::{ZeneRunConfig, ZeneRunner, ZeneRunResult};
pub use zene_config::AgentProfile;
