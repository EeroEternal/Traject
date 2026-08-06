//! Internal protocol between Rust control plane and Python execution core.

use serde::{Deserialize, Serialize};

use crate::openai::ChatMessage;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationRequest {
    pub request_id: String,
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<i32>,
    pub seed: Option<u64>,
    pub stop: Option<Vec<String>>,
    pub stream: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prefix_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub step_id: Option<String>,
}

/// A single generated token delta (or final).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenDelta {
    #[serde(default)]
    pub text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_tokens: Option<u32>,
}

/// Blocking result used by non-stream path.
#[derive(Debug, Clone)]
pub struct GenerationResult {
    pub text: String,
    pub finish_reason: String,
    pub usage: Usage,
}
