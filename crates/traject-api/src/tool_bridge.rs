//! OpenAI tool-calling bridge for engines that reject `tools` (e.g. sglang-lite).
//!
//! Accepts OpenAI chat requests with tools, injects a text protocol into the
//! prompt, forwards to an upstream `/v1/chat/completions` without `tools`,
//! then parses assistant text back into OpenAI `tool_calls`.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::State;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

const TOOL_PROTOCOL: &str = r#"You are a coding agent with tools. When you need a tool, reply with ONLY this JSON (no markdown fences):
{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"TOOL_NAME","arguments":"{\"arg\":\"value\"}"}}]}
When you are done and do not need tools, reply with plain text only (no tool_calls JSON).
"#;

#[derive(Clone)]
pub struct BridgeState {
    pub upstream: String,
    pub client: reqwest::Client,
}

#[derive(Debug, Deserialize)]
struct BridgeChatRequest {
    #[serde(default)]
    model: String,
    messages: Vec<Value>,
    #[serde(default)]
    tools: Option<Vec<Value>>,
    #[serde(default)]
    tool_choice: Option<Value>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(flatten)]
    _extra: Value,
}

#[derive(Debug, Serialize)]
struct Health {
    status: &'static str,
    service: &'static str,
    mode: &'static str,
}

async fn health() -> Json<Health> {
    Json(Health {
        status: "ok",
        service: "traject",
        mode: "tool-bridge",
    })
}

fn inject_tools(messages: &mut Vec<Value>, tools: &[Value]) {
    let tools_json = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".into());
    let system = format!(
        "{TOOL_PROTOCOL}\nAvailable tools (OpenAI schema):\n{tools_json}\n"
    );
    messages.insert(
        0,
        json!({
            "role": "system",
            "content": system,
        }),
    );
}

fn normalize_messages(messages: Vec<Value>) -> Vec<Value> {
    messages
        .into_iter()
        .map(|mut m| {
            // OpenAI tool results / assistant tool_calls → plain text the upstream understands.
            let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("").to_string();
            if role == "tool" {
                let name = m
                    .get("name")
                    .and_then(|n| n.as_str())
                    .unwrap_or("tool");
                let content = m
                    .get("content")
                    .map(|c| match c {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();
                return json!({
                    "role": "user",
                    "content": format!("Tool `{name}` result:\n{content}"),
                });
            }
            if role == "assistant" {
                if let Some(calls) = m.get("tool_calls").cloned() {
                    let text = m
                        .get("content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("")
                        .to_string();
                    let payload = if text.is_empty() {
                        json!({ "tool_calls": calls }).to_string()
                    } else {
                        format!("{text}\n{}", json!({ "tool_calls": calls }))
                    };
                    return json!({
                        "role": "assistant",
                        "content": payload,
                    });
                }
                // Ensure content is a string.
                if m.get("content").is_none() {
                    m["content"] = Value::String(String::new());
                } else if !m["content"].is_string() {
                    m["content"] = Value::String(m["content"].to_string());
                }
            }
            m
        })
        .collect()
}

fn extract_json_object(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    // fenced ```json
    if let Some(start) = trimmed.find('{') {
        if let Some(end) = trimmed.rfind('}') {
            if end > start {
                if let Ok(v) = serde_json::from_str::<Value>(&trimmed[start..=end]) {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn parse_tool_calls(text: &str) -> Option<Vec<Value>> {
    let v = extract_json_object(text)?;
    let calls = v.get("tool_calls")?.as_array()?;
    if calls.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for (i, c) in calls.iter().enumerate() {
        let id = c
            .get("id")
            .and_then(|x| x.as_str())
            .map(str::to_string)
            .unwrap_or_else(|| format!("call_{i}"));
        let name = c
            .pointer("/function/name")
            .or_else(|| c.get("name"))
            .and_then(|x| x.as_str())
            .unwrap_or("")
            .to_string();
        if name.is_empty() {
            continue;
        }
        let arguments = match c.pointer("/function/arguments").or_else(|| c.get("arguments")) {
            Some(Value::String(s)) => s.clone(),
            Some(other) => other.to_string(),
            None => "{}".into(),
        };
        out.push(json!({
            "id": id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments,
            }
        }));
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

async fn chat_completions(
    State(state): State<Arc<BridgeState>>,
    Json(req): Json<BridgeChatRequest>,
) -> Result<Json<Value>, (axum::http::StatusCode, String)> {
    if req.stream == Some(true) {
        return Err((
            axum::http::StatusCode::BAD_REQUEST,
            "streaming not supported by traject tool-bridge".into(),
        ));
    }

    let mut messages = normalize_messages(req.messages);
    let has_tools = req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false);
    if has_tools {
        inject_tools(messages.as_mut(), req.tools.as_ref().unwrap());
    }
    let _ = req.tool_choice;

    let body = json!({
        "model": req.model,
        "messages": messages,
        "max_tokens": req.max_tokens.unwrap_or(1024),
        "temperature": req.temperature.unwrap_or(0.2),
    });

    let url = format!(
        "{}/v1/chat/completions",
        state.upstream.trim_end_matches('/')
    );
    let resp = state
        .client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;
    let status = resp.status();
    let upstream_body = resp
        .text()
        .await
        .map_err(|e| (axum::http::StatusCode::BAD_GATEWAY, e.to_string()))?;
    if !status.is_success() {
        return Err((
            axum::http::StatusCode::from_u16(status.as_u16()).unwrap_or(axum::http::StatusCode::BAD_GATEWAY),
            format!("upstream {status}: {upstream_body}"),
        ));
    }

    let mut parsed: Value = serde_json::from_str(&upstream_body).map_err(|e| {
        (
            axum::http::StatusCode::BAD_GATEWAY,
            format!("upstream json: {e}: {upstream_body}"),
        )
    })?;

    if has_tools {
        if let Some(content) = parsed
            .pointer("/choices/0/message/content")
            .and_then(|c| c.as_str())
            .map(str::to_string)
        {
            if let Some(tool_calls) = parse_tool_calls(&content) {
                if let Some(msg) = parsed.pointer_mut("/choices/0/message") {
                    *msg = json!({
                        "role": "assistant",
                        "content": null,
                        "tool_calls": tool_calls,
                    });
                }
                if let Some(fr) = parsed.pointer_mut("/choices/0/finish_reason") {
                    *fr = json!("tool_calls");
                }
            }
        }
    }

    Ok(Json(parsed))
}

pub fn bridge_router(state: Arc<BridgeState>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

pub async fn serve_tool_bridge(addr: SocketAddr, upstream: String) -> Result<(), std::io::Error> {
    let state = Arc::new(BridgeState {
        upstream,
        client: reqwest::Client::new(),
    });
    let app = bridge_router(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!(%addr, "traject tool-bridge listening");
    axum::serve(listener, app).await
}

/// Bind an ephemeral port and return (bound_addr, server_future).
pub async fn spawn_tool_bridge(
    upstream: String,
) -> Result<(SocketAddr, tokio::task::JoinHandle<Result<(), std::io::Error>>), std::io::Error> {
    let state = Arc::new(BridgeState {
        upstream,
        client: reqwest::Client::new(),
    });
    let app = bridge_router(state);
    let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await?;
    let addr = listener.local_addr()?;
    let handle = tokio::spawn(async move { axum::serve(listener, app).await });
    Ok((addr, handle))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_plain_tool_calls_json() {
        let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Glob","arguments":"{\"pattern\":\"*.txt\"}"}}]}"#;
        let calls = parse_tool_calls(text).expect("parse");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["function"]["name"], "Glob");
    }

    #[test]
    fn ignores_plain_text() {
        assert!(parse_tool_calls("任务已完成，没有文件需要修改。").is_none());
    }
}
