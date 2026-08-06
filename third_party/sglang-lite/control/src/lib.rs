//! sglang-lite thin control plane library (OpenAI API layer + engine client).

use std::{net::SocketAddr, sync::Arc, time::Duration};

use anyhow::Result;
use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use futures::Stream;
use std::convert::Infallible;
use tower_http::{cors::CorsLayer, trace::TraceLayer};
use tracing::info;
use uuid::Uuid;

pub mod engine_client;
pub mod http_engine;
pub mod openai;
pub mod protocol;
pub mod stub_engine;

pub use engine_client::EngineClient;
pub use http_engine::HttpEngineClient;
pub use openai::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Delta, Role};
pub use protocol::{GenerationRequest, TokenDelta, Usage};
pub use stub_engine::StubEngineClient;

#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<EngineClient>,
    pub model_list: Arc<Vec<String>>,
    pub core_url: Option<String>,
    pub ready: Arc<std::sync::atomic::AtomicBool>,
    pub draining: Arc<std::sync::atomic::AtomicBool>,
}

/// Build the axum router for the sglang-lite control plane.
pub fn build_router(
    engine: Arc<EngineClient>,
    model_list: Arc<Vec<String>>,
    core_url: Option<String>,
    ready: Arc<std::sync::atomic::AtomicBool>,
    draining: Arc<std::sync::atomic::AtomicBool>,
) -> Router {
    let state = AppState {
        engine,
        model_list,
        core_url,
        ready,
        draining,
    };

    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/healthz", get(healthz))
        .route("/v1/health", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .with_state(state)
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
}

/// Run a standalone HTTP server with the given configuration.
pub async fn serve(
    engine: Arc<EngineClient>,
    model_list: Arc<Vec<String>>,
    core_url: Option<String>,
    port: u16,
    ready: Arc<std::sync::atomic::AtomicBool>,
    draining: Arc<std::sync::atomic::AtomicBool>,
) -> Result<()> {
    let app = build_router(engine, model_list, core_url, ready, draining);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));
    info!("sglang-lite control plane listening on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            let _ = tokio::signal::ctrl_c().await;
            info!("shutdown signal received");
        })
        .await?;
    Ok(())
}

/// POST /v1/chat/completions — the primary control surface.
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, (axum::http::StatusCode, Json<serde_json::Value>)> {
    use std::sync::atomic::Ordering;

    if state.draining.load(Ordering::Relaxed) {
        return Err(api_err(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "server is draining; not accepting new requests",
            "draining",
        ));
    }

    if !state.ready.load(Ordering::Relaxed) {
        return Err(api_err(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "server_error",
            "model not ready",
            "not_ready",
        ));
    }

    if !state.model_list.iter().any(|m| m == &req.model) {
        return Err(api_err(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            &format!(
                "model '{}' not supported in sglang-lite (see GET /v1/models).",
                req.model
            ),
            "model_not_found",
        ));
    }

    if req.messages.is_empty() {
        return Err(api_err(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "messages must be non-empty",
            "invalid_messages",
        ));
    }

    if req.messages.iter().any(|m| {
        matches!(m, ChatMessage::User { content: c, .. } if c.contains("data:image") || c.contains("<image>"))
    }) {
        return Err(api_err(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Multimodal content is not supported in sglang-lite core.",
            "multimodal_not_supported",
        ));
    }

    if req.response_format.is_some() {
        return Err(api_err(
            axum::http::StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "response_format / structured output is not supported inside the engine.",
            "structured_output_not_supported",
        ));
    }

    if let Some(mt) = req.max_tokens {
        if mt < 1 {
            return Err(api_err(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "max_tokens must be >= 1",
                "invalid_max_tokens",
            ));
        }
    }
    if let Some(t) = req.temperature {
        if t < 0.0 {
            return Err(api_err(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "temperature must be >= 0",
                "invalid_temperature",
            ));
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0 < p && p <= 1.0) {
            return Err(api_err(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "top_p must be in (0, 1]",
                "invalid_top_p",
            ));
        }
    }

    // Reject clearly out-of-scope extras
    for key in ["tools", "tool_choice", "logit_bias", "n", "functions"] {
        if req.extra.contains_key(key) {
            return Err(api_err(
                axum::http::StatusCode::BAD_REQUEST,
                "invalid_request_error",
                &format!("parameter '{}' is not supported in sglang-lite", key),
                "unsupported_parameter",
            ));
        }
    }

    let request_id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    let gen_req = GenerationRequest {
        request_id: request_id.clone(),
        model: req.model.clone(),
        messages: req.messages.clone(),
        max_tokens: req.max_tokens.unwrap_or(512),
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: req.top_k,
        seed: req.seed,
        stop: req.stop.clone(),
        stream: req.stream.unwrap_or(false),
        trajectory_id: None,
        session_id: None,
        prefix_id: None,
        step_id: None,
    };

    if gen_req.stream {
        let stream = stream_chat(state.engine.clone(), gen_req, created, request_id);
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::default())
            .into_response())
    } else {
        let resp = state.engine.generate_blocking(gen_req).await.map_err(|e| {
            api_err(
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                &format!("engine error: {}", e),
                "engine_error",
            )
        })?;

        let choice = openai::ChatChoice {
            index: 0,
            message: ChatMessage::Assistant {
                content: Some(resp.text),
                tool_calls: None,
            },
            finish_reason: Some(resp.finish_reason),
        };

        let openai_usage = openai::Usage {
            prompt_tokens: resp.usage.prompt_tokens,
            completion_tokens: resp.usage.completion_tokens,
            total_tokens: resp.usage.total_tokens,
            cache_hit_tokens: resp.usage.cache_hit_tokens,
        };
        let response = ChatCompletionResponse {
            id: request_id,
            object: "chat.completion".to_string(),
            created,
            model: req.model,
            choices: vec![choice],
            usage: Some(openai_usage),
        };
        Ok(Json(response).into_response())
    }
}

/// SSE stream helper — cancels engine work if the client disconnects.
pub fn stream_chat(
    engine: Arc<EngineClient>,
    gen_req: GenerationRequest,
    created: i64,
    request_id: String,
) -> impl Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let model = gen_req.model.clone();
        let cancel_id = gen_req.request_id.clone();
        let mut token_stream = engine.generate_stream(gen_req).await;

        let first = Event::default()
            .json_data(openai::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![openai::ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: Some(Role::Assistant),
                        content: None,
                    },
                    finish_reason: None,
                }],
                usage: None,
            })
            .unwrap_or_else(|_| Event::default().data("data: [ERROR]"));
        yield Ok(first);

        let mut saw_finish = false;
        while let Some(delta) = token_stream.recv().await {
            if let Some(err) = delta.error.clone() {
                let ev = Event::default().data(format!("engine error: {}", err));
                yield Ok(ev);
                let _ = engine.cancel(&cancel_id).await;
                break;
            }
            let finish = delta.finish_reason.clone();
            let usage = delta.usage.as_ref().map(|u| openai::Usage {
                prompt_tokens: u.prompt_tokens,
                completion_tokens: u.completion_tokens,
                total_tokens: u.total_tokens,
                cache_hit_tokens: u.cache_hit_tokens,
            });
            let chunk = openai::ChatCompletionChunk {
                id: request_id.clone(),
                object: "chat.completion.chunk".to_string(),
                created,
                model: model.clone(),
                choices: vec![openai::ChunkChoice {
                    index: 0,
                    delta: Delta {
                        role: None,
                        content: Some(delta.text),
                    },
                    finish_reason: finish.clone(),
                }],
                usage,
            };

            match Event::default().json_data(chunk) {
                Ok(ev) => yield Ok(ev),
                Err(_) => {
                    let _ = engine.cancel(&cancel_id).await;
                    break;
                }
            }

            if finish.is_some() {
                saw_finish = true;
                break;
            }
        }

        // If the SSE consumer dropped early, cancel backend work.
        if !saw_finish {
            let _ = engine.cancel(&cancel_id).await;
        }

        yield Ok(Event::default().data("[DONE]"));
    }
}

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<openai::ModelsResponse> {
    let data = state
        .model_list
        .iter()
        .map(|id| openai::ModelObject {
            id: id.clone(),
            object: "model".to_string(),
            owned_by: "sglang-lite".to_string(),
        })
        .collect();

    Json(openai::ModelsResponse {
        object: "list".to_string(),
        data,
    })
}

/// GET /healthz — liveness
pub async fn healthz() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "sglang-lite",
    }))
}

/// GET /readyz — readiness (model loaded + engine warm)
pub async fn readyz(State(state): State<AppState>) -> impl IntoResponse {
    use std::sync::atomic::Ordering;
    if state.ready.load(Ordering::Relaxed) && state.engine.ready().await {
        (
            axum::http::StatusCode::OK,
            Json(serde_json::json!({"status": "ready"})),
        )
    } else {
        (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"status": "not_ready"})),
        )
    }
}

/// GET /metrics
pub async fn metrics(
    State(state): State<AppState>,
) -> Result<String, (axum::http::StatusCode, String)> {
    if let Some(base) = state.core_url.as_ref() {
        let client = reqwest::Client::builder()
            .no_proxy()
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        let url = format!("{}/metrics", base.trim_end_matches('/'));
        if let Ok(resp) = client
            .get(&url)
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            if resp.status().is_success() {
                if let Ok(body) = resp.text().await {
                    return Ok(body);
                }
            }
        }
    }

    let mut output = String::from("# sglang-lite metrics\n");
    output.push_str("sglang_lite_up 1\n");
    Ok(output)
}

fn api_err(
    status: axum::http::StatusCode,
    typ: &str,
    message: &str,
    code: &str,
) -> (axum::http::StatusCode, Json<serde_json::Value>) {
    (
        status,
        Json(serde_json::json!({
            "error": {
                "message": message,
                "type": typ,
                "param": null,
                "code": code
            }
        })),
    )
}
