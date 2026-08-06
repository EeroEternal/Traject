//! Unified engine client: stub (tests) or HTTP (production).

use anyhow::Result;
use tokio::sync::mpsc;

use crate::http_engine::HttpEngineClient;
use crate::protocol::{GenerationRequest, GenerationResult, TokenDelta};
use crate::stub_engine::StubEngineClient;

#[derive(Clone)]
pub enum EngineClient {
    Stub(StubEngineClient),
    Http(HttpEngineClient),
}

impl EngineClient {
    pub fn stub() -> Self {
        Self::Stub(StubEngineClient::new())
    }

    pub fn http(base: impl Into<String>) -> Self {
        Self::Http(HttpEngineClient::new(base))
    }

    pub async fn generate_blocking(&self, req: GenerationRequest) -> Result<GenerationResult> {
        match self {
            Self::Stub(c) => c.generate_blocking(req).await,
            Self::Http(c) => c.generate_blocking(req).await,
        }
    }

    pub async fn generate_stream(&self, req: GenerationRequest) -> mpsc::Receiver<TokenDelta> {
        match self {
            Self::Stub(c) => c.generate_stream(req).await,
            Self::Http(c) => c.generate_stream(req).await,
        }
    }

    pub async fn cancel(&self, request_id: &str) -> Result<()> {
        match self {
            Self::Stub(_) => Ok(()),
            Self::Http(c) => c.cancel(request_id).await,
        }
    }

    pub async fn ready(&self) -> bool {
        match self {
            Self::Stub(_) => true,
            Self::Http(c) => c.ready().await,
        }
    }

    pub fn is_http(&self) -> bool {
        matches!(self, Self::Http(_))
    }
}
