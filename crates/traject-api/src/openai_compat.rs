use serde::{Deserialize, Serialize};
use traject_core::{Result, TrajectoryConfig};

/// Minimal OpenAI-compatible chat types (shim only).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub choices: Vec<ChatChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: String,
}

/// Compatibility layer: each chat call becomes a short-lived Trajectory.
pub struct OpenAiCompat;

impl OpenAiCompat {
    pub async fn chat_completions(
        req: ChatCompletionRequest,
    ) -> Result<ChatCompletionResponse> {
        let prompt = req
            .messages
            .iter()
            .map(|m| format!("{}: {}", m.role, m.content))
            .collect::<Vec<_>>()
            .join("\n");

        let id = traject_runtime::run_simple_react(&prompt).await?;

        Ok(ChatCompletionResponse {
            id: id.to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "(stub) trajectory finished".into(),
                },
                finish_reason: "stop".into(),
            }],
        })
    }

    pub fn default_trajectory_config() -> TrajectoryConfig {
        TrajectoryConfig::default()
    }
}
