//! HTTP client for Traject ↔ sglang-lite prefix pin/unpin (MemoryManager alignment).

use serde_json::json;
use tracing::{debug, warn};

/// Fire-and-forget friendly client for engine prefix lifecycle APIs.
#[derive(Debug, Clone)]
pub struct EnginePrefixClient {
    base_url: String,
    client: reqwest::Client,
}

impl EnginePrefixClient {
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into().trim_end_matches('/').to_string(),
            client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("reqwest"),
        }
    }

    pub async fn pin(
        &self,
        prefix_id: &str,
        session_id: Option<&str>,
        trajectory_id: Option<&str>,
        ttl_ms: u64,
        reason: &str,
    ) {
        let url = format!("{}/v1/prefix/pin", self.base_url);
        let body = json!({
            "prefix_id": prefix_id,
            "session_id": session_id,
            "trajectory_id": trajectory_id,
            "ttl_ms": ttl_ms,
            "reason": reason,
        });
        match self.client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!(%prefix_id, reason, "engine prefix pinned");
            }
            Ok(resp) => {
                warn!(%prefix_id, status = %resp.status(), "engine prefix pin rejected");
            }
            Err(e) => {
                // Engine may be older / offline — soft fail.
                debug!(%prefix_id, error = %e, "engine prefix pin skipped");
            }
        }
    }

    pub async fn unpin(&self, prefix_id: &str) {
        let url = format!("{}/v1/prefix/unpin", self.base_url);
        let body = json!({ "prefix_id": prefix_id });
        match self.client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!(%prefix_id, "engine prefix unpinned");
            }
            Ok(resp) => {
                warn!(%prefix_id, status = %resp.status(), "engine prefix unpin rejected");
            }
            Err(e) => {
                debug!(%prefix_id, error = %e, "engine prefix unpin skipped");
            }
        }
    }

    /// MemoryManager eviction: drop pin + V4 snapshot for this handle.
    pub async fn free(&self, prefix_id: &str, session_id: Option<&str>) {
        let url = format!("{}/v1/prefix/free", self.base_url);
        let body = json!({
            "prefix_id": prefix_id,
            "session_id": session_id,
        });
        match self.client.post(&url).json(&body).send().await {
            Ok(resp) if resp.status().is_success() => {
                debug!(%prefix_id, "engine prefix freed");
            }
            Ok(resp) => {
                warn!(%prefix_id, status = %resp.status(), "engine prefix free rejected");
            }
            Err(e) => {
                debug!(%prefix_id, error = %e, "engine prefix free skipped");
            }
        }
    }
}
