//! HTTP client for the Python engine process (true NDJSON TokenDelta stream).

use anyhow::{anyhow, Result};
use futures::StreamExt;
use reqwest::Client;
use tokio::sync::mpsc;
use tracing::warn;

use crate::protocol::{GenerationRequest, GenerationResult, TokenDelta, Usage};

#[derive(Clone)]
pub struct HttpEngineClient {
    base: String,
    client: Client,
    request_timeout: std::time::Duration,
}

impl HttpEngineClient {
    pub fn new(base: impl Into<String>) -> Self {
        let client = Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| Client::new());
        Self {
            base: base.into().trim_end_matches('/').to_string(),
            client,
            request_timeout: std::time::Duration::from_secs(300),
        }
    }

    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    fn generate_url(&self) -> String {
        format!("{}/v1/generate", self.base)
    }

    fn cancel_url(&self) -> String {
        format!("{}/v1/cancel", self.base)
    }

    pub async fn ready(&self) -> bool {
        let url = format!("{}/readyz", self.base);
        match self.client.get(&url).send().await {
            Ok(r) => r.status().is_success(),
            Err(_) => false,
        }
    }

    pub async fn cancel(&self, request_id: &str) -> Result<()> {
        let body = serde_json::json!({ "request_id": request_id });
        let resp = self
            .client
            .post(self.cancel_url())
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            warn!("cancel failed: {}", resp.status());
        }
        Ok(())
    }

    fn body_json(req: &GenerationRequest, stream: bool) -> serde_json::Value {
        serde_json::json!({
            "request_id": req.request_id,
            "model": req.model,
            "messages": req.messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "top_k": req.top_k,
            "seed": req.seed,
            "stop": req.stop,
            "stream": stream,
            "timeout_s": 300.0,
        })
    }

    pub async fn generate_blocking(&self, req: GenerationRequest) -> Result<GenerationResult> {
        let mut stream_req = req.clone();
        stream_req.stream = true;
        let mut rx = self.generate_stream(stream_req).await;
        let mut text = String::new();
        let mut finish = "stop".to_string();
        let mut usage = Usage {
            prompt_tokens: 0,
            completion_tokens: 0,
            total_tokens: 0,
            cache_hit_tokens: None,
        };
        while let Some(delta) = rx.recv().await {
            if let Some(err) = delta.error {
                anyhow::bail!("engine error: {}", err);
            }
            text.push_str(&delta.text);
            if let Some(fr) = delta.finish_reason {
                finish = fr;
            }
            if let Some(u) = delta.usage {
                usage = u;
            }
        }
        Ok(GenerationResult {
            text,
            finish_reason: finish,
            usage,
        })
    }

    /// True token/delta NDJSON stream from the Python engine process.
    pub async fn generate_stream(&self, req: GenerationRequest) -> mpsc::Receiver<TokenDelta> {
        let (tx, rx) = mpsc::channel::<TokenDelta>(128);
        let client = self.client.clone();
        let url = self.generate_url();
        let cancel_url = self.cancel_url();
        let timeout = self.request_timeout;
        let request_id = req.request_id.clone();
        let body = Self::body_json(&req, true);

        tokio::spawn(async move {
            let result = async {
                let resp = client
                    .post(&url)
                    .timeout(timeout)
                    .json(&body)
                    .send()
                    .await
                    .map_err(|e| anyhow!("engine connect: {}", e))?;

                if !resp.status().is_success() {
                    let txt = resp.text().await.unwrap_or_default();
                    return Err(anyhow!("engine HTTP error: {}", txt));
                }

                let mut byte_stream = resp.bytes_stream();
                let mut buf: Vec<u8> = Vec::new();

                while let Some(item) = byte_stream.next().await {
                    let chunk = item.map_err(|e| anyhow!("stream read: {}", e))?;
                    buf.extend_from_slice(&chunk);

                    while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                        let line_bytes = buf[..pos].to_vec();
                        buf = buf[pos + 1..].to_vec();
                        if line_bytes.is_empty() {
                            continue;
                        }
                        let line = std::str::from_utf8(&line_bytes)
                            .map_err(|e| anyhow!("utf-8 TokenDelta line: {}", e))?
                            .trim()
                            .to_string();
                        if line.is_empty() {
                            continue;
                        }
                        let delta: TokenDelta = serde_json::from_str(&line)
                            .map_err(|e| anyhow!("bad TokenDelta json: {} ({})", e, line))?;

                        let done = delta.finish_reason.is_some() || delta.error.is_some();
                        if tx.send(delta).await.is_err() {
                            // Client disconnected / backpressure abandoned → cancel engine
                            let _ = client
                                .post(&cancel_url)
                                .json(&serde_json::json!({ "request_id": request_id }))
                                .send()
                                .await;
                            return Ok(());
                        }
                        if done {
                            return Ok(());
                        }
                    }
                }
                Err(anyhow!("engine stream ended without finish_reason"))
            }
            .await;

            if let Err(e) = result {
                let _ = tx
                    .send(TokenDelta {
                        text: String::new(),
                        finish_reason: Some("error".to_string()),
                        usage: None,
                        error: Some(e.to_string()),
                        token: None,
                    })
                    .await;
            }
        });

        rx
    }
}
