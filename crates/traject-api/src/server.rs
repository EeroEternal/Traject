//! HTTP OpenAI-compatible server (stub-backed Phase 1).

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use tokio::sync::Mutex;
use traject_core::TrajectoryConfig;
use traject_inference::StubMode;
use traject_policy::ReActPolicy;
use traject_runtime::{Driver, DriverConfig};

use crate::openai_compat::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage};

#[derive(Clone)]
pub struct AppState {
    inner: Arc<Mutex<Driver>>,
}

impl AppState {
    pub fn new() -> Self {
        let driver = Driver::new(DriverConfig::default())
            .with_policy(Arc::new(ReActPolicy::new("openai-compat")))
            .with_stub_mode(StubMode::AlwaysStop);
        Self {
            inner: Arc::new(Mutex::new(driver)),
        }
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Health {
    status: &'static str,
    service: &'static str,
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "traject",
    })
}

async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, (axum::http::StatusCode, String)> {
    let prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let mut driver = state.inner.lock().await;
    driver.policy = Arc::new(ReActPolicy::new(prompt));
    let id = driver.create_trajectory(TrajectoryConfig::default());
    driver
        .run_until_finished(id)
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let traj = driver
        .manager
        .get(id)
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    let content = traj
        .last_outcome()
        .and_then(|o| match o {
            traject_core::StepOutcome::Generated { text, .. } => Some(text.clone()),
            _ => None,
        })
        .unwrap_or_else(|| "(empty)".into());

    Ok(Json(ChatCompletionResponse {
        id: id.to_string(),
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason: "stop".into(),
        }],
    }))
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn serve(addr: SocketAddr) -> Result<(), std::io::Error> {
    let app = router(AppState::new());
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "traject openai-compat listening");
    axum::serve(listener, app).await
}
