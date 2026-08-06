//! External API surface: native Trajectory API + OpenAI-compatible shim.

mod native;
mod openai_compat;
mod server;
mod tool_bridge;

pub use native::{Traject, TrajectoryHandle};
pub use openai_compat::{
    ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage, OpenAiCompat,
};
pub use server::{router, serve, AppState};
pub use tool_bridge::{serve_tool_bridge, spawn_tool_bridge, BridgeState};
