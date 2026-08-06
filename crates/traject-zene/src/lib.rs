//! Integrate the Zene coding agent into Traject.
//!
//! Architecture:
//! - Zene owns agent semantics (ReAct loop, tools, permissions, plan mode, compaction)
//! - Traject owns the host process, inference endpoint routing, and (later) Trajectory
//!   scheduling / KV. For the first merge, each `ZeneRunner::prompt` is one Trajectory
//!   that runs Zene's turn loop against a Traject-routed OpenAI-compatible endpoint
//!   (typically sglang-lite on the 5090 host).

mod runner;

pub use runner::{ZeneRunConfig, ZeneRunner, ZeneRunResult};
pub use zene_config::AgentProfile;
