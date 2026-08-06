//! In-process stub engine client (control-plane unit tests / --stub mode only).

use anyhow::Result;
use tokio::sync::mpsc;

use crate::protocol::{GenerationRequest, GenerationResult, TokenDelta, Usage};

#[derive(Clone, Default)]
pub struct StubEngineClient;

impl StubEngineClient {
    pub fn new() -> Self {
        Self
    }

    pub async fn generate_blocking(&self, req: GenerationRequest) -> Result<GenerationResult> {
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        let prompt_tokens = estimate_prompt_tokens(&req);
        let mut completion = simulate_completion(&req, req.seed.unwrap_or(0));
        let max_t = req.max_tokens as usize;
        if completion.len() > max_t {
            completion.truncate(max_t);
        }
        let completion_tokens = completion.len() as u32;
        Ok(GenerationResult {
            text: completion.join(" "),
            finish_reason: "stop".to_string(),
            usage: Usage {
                prompt_tokens,
                completion_tokens,
                total_tokens: prompt_tokens + completion_tokens,
                cache_hit_tokens: None,
            },
        })
    }

    pub async fn generate_stream(&self, req: GenerationRequest) -> mpsc::Receiver<TokenDelta> {
        let (tx, rx) = mpsc::channel(128);
        tokio::spawn(async move {
            let prompt_tokens = estimate_prompt_tokens(&req);
            let mut tokens = simulate_completion(&req, req.seed.unwrap_or(42));
            if tokens.len() > req.max_tokens as usize {
                tokens.truncate(req.max_tokens as usize);
            }
            let n = tokens.len();
            let mut completion_tokens = 0u32;
            for (i, tok) in tokens.into_iter().enumerate() {
                completion_tokens += 1;
                let is_last = i + 1 == n;
                let delta = TokenDelta {
                    text: format!("{} ", tok),
                    finish_reason: if is_last {
                        Some("stop".to_string())
                    } else {
                        None
                    },
                    usage: if is_last {
                        Some(Usage {
                            prompt_tokens,
                            completion_tokens,
                            total_tokens: prompt_tokens + completion_tokens,
                            cache_hit_tokens: None,
                        })
                    } else {
                        None
                    },
                    error: None,
                    token: None,
                };
                if tx.send(delta).await.is_err() {
                    break;
                }
            }
        });
        rx
    }
}

fn estimate_prompt_tokens(req: &GenerationRequest) -> u32 {
    let mut n = 0u32;
    for m in &req.messages {
        match m {
            crate::openai::ChatMessage::System { content, .. } => {
                n += (content.len() as u32) / 3 + 4
            }
            crate::openai::ChatMessage::User { content, .. } => n += (content.len() as u32) / 3 + 4,
            crate::openai::ChatMessage::Assistant { content, .. } => {
                if let Some(c) = content {
                    n += (c.len() as u32) / 3 + 4;
                }
            }
            crate::openai::ChatMessage::Tool { content, .. } => n += (content.len() as u32) / 3 + 4,
        }
    }
    n.max(12)
}

fn simulate_completion(req: &GenerationRequest, _seed: u64) -> Vec<String> {
    let last_user = req
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            crate::openai::ChatMessage::User { content, .. } => Some(content.as_str()),
            _ => None,
        })
        .unwrap_or("Hello");
    format!("stub reply to: {}", last_user)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect()
}
