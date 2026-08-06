//! HTTP bridge to an OpenAI-compatible inference server (e.g. SGLang / vLLM).
//!
//! This is the first real-backend path: Traject keeps Trajectory scheduling,
//! while token generation is delegated over HTTP. Later this becomes an
//! in-process FlashInfer / custom kernel backend.

use async_trait::async_trait;
use traject_core::{FinishReason, Result, TrajectError};

use crate::{ChunkRequest, ChunkResult, InferenceBackend};

#[derive(Debug, Clone)]
pub struct HttpOpenAiBackend {
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
    client: reqwest::Client,
}

impl HttpOpenAiBackend {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            api_key: None,
            client: reqwest::Client::new(),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

#[derive(serde::Serialize)]
struct ChatReq<'a> {
    model: &'a str,
    messages: Vec<Msg<'a>>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(serde::Serialize)]
struct Msg<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(serde::Deserialize)]
struct ChatResp {
    choices: Vec<Choice>,
}

#[derive(serde::Deserialize)]
struct Choice {
    message: MsgOwned,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(serde::Deserialize)]
struct MsgOwned {
    content: Option<String>,
}

#[async_trait]
impl InferenceBackend for HttpOpenAiBackend {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult> {
        // Phase 1 bridge: each "chunk" is a full completion. Interruptible
        // streaming lands when we switch to SSE / native kernels.
        let prompt = req
            .delta
            .text
            .clone()
            .unwrap_or_else(|| format!("tokens:{:?}", req.delta.token_ids));

        let body = ChatReq {
            model: &self.model,
            messages: vec![Msg {
                role: "user",
                content: &prompt,
            }],
            max_tokens: req.chunk_tokens.max(1),
            temperature: req.constraints.temperature.unwrap_or(0.7),
        };

        let url = format!("{}/v1/chat/completions", self.base_url);
        let mut builder = self.client.post(&url).json(&body);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }

        let resp = builder.send().await.map_err(|e| {
            TrajectError::Inference(format!("http backend request failed: {e}"))
        })?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(TrajectError::Inference(format!(
                "http backend {status}: {text}"
            )));
        }

        let parsed: ChatResp = resp.json().await.map_err(|e| {
            TrajectError::Inference(format!("http backend decode failed: {e}"))
        })?;

        let text = parsed
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();
        let finish = parsed
            .choices
            .first()
            .and_then(|c| c.finish_reason.as_deref())
            .unwrap_or("stop");
        let finish_reason = match finish {
            "length" => FinishReason::Length,
            "tool_calls" | "function_call" => FinishReason::ToolCall,
            _ => FinishReason::Stop,
        };

        Ok(ChunkResult {
            token_ids: text.bytes().map(|b| b as u32).collect(),
            tokens_produced: text.len() as u32,
            text,
            finished: true,
            finish_reason: Some(finish_reason),
            tool_call: None,
            new_prefix: None,
        })
    }
}
