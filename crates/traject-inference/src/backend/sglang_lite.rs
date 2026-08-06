//! Native sglang-lite engine protocol (`/v1/generate` NDJSON TokenDelta).
//!
//! Traject-owned path: requests carry `trajectory_id` / `session_id` / `prefix_id`
//! so the engine (and MemoryManager) can track agent sessions across turns.
//! Prefix reuse is performed by the engine radix/V4 cache on growing prompts;
//! `cache_hit_tokens` is returned on the final delta.

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::json;
use traject_core::{FinishReason, Result, TrajectError};

use crate::{ChunkRequest, ChunkResult, InferenceBackend};

#[derive(Debug, Clone)]
pub struct SglangLiteEngineBackend {
    pub base_url: String,
    pub model: String,
    client: reqwest::Client,
}

impl SglangLiteEngineBackend {
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            model: model.into(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(600))
                .build()
                .expect("reqwest client"),
        }
    }

    pub async fn wait_ready(&self, timeout: std::time::Duration) -> Result<()> {
        let url = format!("{}/readyz", self.base_url);
        let start = std::time::Instant::now();
        loop {
            if let Ok(resp) = self.client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(());
                }
            }
            if start.elapsed() > timeout {
                return Err(TrajectError::Inference(format!(
                    "engine not ready at {} after {:?}",
                    self.base_url, timeout
                )));
            }
            tokio::time::sleep(std::time::Duration::from_secs(2)).await;
        }
    }

    pub async fn healthz(&self) -> Result<bool> {
        let url = format!("{}/healthz", self.base_url);
        let resp = self
            .client
            .get(&url)
            .send()
            .await
            .map_err(|e| TrajectError::Inference(e.to_string()))?;
        Ok(resp.status().is_success())
    }
}

#[derive(serde::Deserialize)]
struct TokenDelta {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    token: Option<u32>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    usage: Option<UsageDelta>,
}

#[derive(serde::Deserialize)]
struct UsageDelta {
    #[serde(default)]
    cache_hit_tokens: Option<u32>,
    /// Session prompt longest-common-prefix (token count) vs previous turn.
    #[serde(default)]
    session_lcp_tokens: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    prompt_tokens: Option<u32>,
    #[serde(default)]
    #[allow(dead_code)]
    completion_tokens: Option<u32>,
}

#[async_trait]
impl InferenceBackend for SglangLiteEngineBackend {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult> {
        let prompt = req
            .delta
            .text
            .clone()
            .unwrap_or_else(|| format!("tokens:{:?}", req.delta.token_ids));

        let request_id = format!("{}:{}", req.trajectory_id, req.step_id);
        let body = json!({
            "request_id": request_id,
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": req.chunk_tokens.max(1).min(req.max_tokens.max(1)),
            "temperature": req.constraints.temperature.unwrap_or(0.0),
            "top_p": req.constraints.top_p.unwrap_or(1.0),
            "stream": true,
            "trajectory_id": req.trajectory_id.to_string(),
            "session_id": req.session_id,
            "prefix_id": req.prefix_hint.or_else(|| req.prefix.map(|p| p.to_string())),
            "step_id": req.step_id.to_string(),
        });

        let url = format!("{}/v1/generate", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| TrajectError::Inference(format!("engine request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(TrajectError::Inference(format!(
                "engine {status}: {text}"
            )));
        }

        let mut text = String::new();
        let mut token_ids = Vec::new();
        let mut finish_reason = FinishReason::Stop;
        let mut cache_hit_tokens = 0u32;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let bytes = chunk
                .map_err(|e| TrajectError::Inference(format!("engine stream: {e}")))?;
            buf.push_str(&String::from_utf8_lossy(&bytes));
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim().to_string();
                buf = buf[pos + 1..].to_string();
                if line.is_empty() {
                    continue;
                }
                let delta: TokenDelta = serde_json::from_str(&line).map_err(|e| {
                    TrajectError::Inference(format!("bad TokenDelta `{line}`: {e}"))
                })?;
                if let Some(err) = delta.error {
                    return Err(TrajectError::Inference(err));
                }
                if let Some(t) = delta.text {
                    text.push_str(&t);
                }
                if let Some(id) = delta.token {
                    token_ids.push(id);
                }
                if let Some(usage) = delta.usage {
                    let mut hit = usage.cache_hit_tokens.unwrap_or(0);
                    if let Some(lcp) = usage.session_lcp_tokens {
                        // Prefer engine floor(max(v4, lcp)); also accept lcp alone.
                        hit = hit.max(lcp);
                    }
                    if hit > 0 {
                        cache_hit_tokens = hit;
                    }
                }
                if let Some(reason) = delta.finish_reason.as_deref() {
                    finish_reason = match reason {
                        "length" => FinishReason::Length,
                        "tool_calls" | "function_call" => FinishReason::ToolCall,
                        "cancelled" => FinishReason::Cancelled,
                        _ => FinishReason::Stop,
                    };
                }
            }
        }

        tracing::info!(
            trajectory = %req.trajectory_id,
            step = %req.step_id,
            cache_hit_tokens,
            out_chars = text.len(),
            "sglang-lite generate finished"
        );

        let produced = token_ids.len() as u32;
        Ok(ChunkResult {
            tokens_produced: if produced > 0 {
                produced
            } else {
                text.len() as u32
            },
            token_ids: if token_ids.is_empty() {
                text.bytes().map(|b| b as u32).collect()
            } else {
                token_ids
            },
            text,
            finished: true,
            finish_reason: Some(finish_reason),
            tool_call: None,
            new_prefix: None,
            cache_hit_tokens,
        })
    }
}
