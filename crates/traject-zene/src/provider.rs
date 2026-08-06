//! Traject-owned LLM provider for the vendored Zene agent.
//!
//! Every Zene `chat` call is a Generate step executed through
//! `Driver` → `Scheduler` → `InferenceEngine` → `MemoryManager`.
//! Tool call/result events become Tool steps on the same Trajectory
//! (pin / fairness / history), still via the Driver.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::{stream, Stream};
use serde_json::Value;
use tokio::sync::Mutex;
use tracing::info;
use traject_core::{
    Constraints, GenerateDelta, StepOutcome, ToolCall as TrajectToolCall,
    ToolResult as TrajectToolResult, Trajectory, TrajectoryConfig, TrajectoryId,
};
use traject_runtime::Driver;
use zene_core::{AgentEvent, EventHandler};
use zene_llm::{
    ChatRequest, ChatResponse, Message, Provider, Role, StreamEvent, ToolCall, ToolDefinition,
};

const TOOL_PROTOCOL: &str = r#"You are a coding agent with tools. When you need a tool, reply with ONLY this JSON (no markdown fences):
{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"TOOL_NAME","arguments":"{\"arg\":\"value\"}"}}]}
When you are done and do not need tools, reply with plain text only (no tool_calls JSON).
"#;

/// Shared host: one Driver + one external Trajectory for a Zene prompt.
pub struct TrajectSession {
    pub driver: Driver,
    pub trajectory_id: TrajectoryId,
    pub model: String,
    pub max_tokens: u32,
    pub generate_steps: u32,
    pub tool_steps: u32,
}

impl TrajectSession {
    pub fn new(mut driver: Driver, model: impl Into<String>) -> Self {
        let trajectory_id = driver.create_external_trajectory(TrajectoryConfig {
            tenant: traject_core::TenantId::new("zene"),
            ..TrajectoryConfig::default()
        });
        Self {
            driver,
            trajectory_id,
            model: model.into(),
            max_tokens: 1024,
            generate_steps: 0,
            tool_steps: 0,
        }
    }

    pub fn trajectory_id(&self) -> TrajectoryId {
        self.trajectory_id
    }

    pub fn total_cache_hit_tokens(&self) -> u32 {
        self.driver.cache_hit_tokens(self.trajectory_id)
    }

    /// Sync event handler: records tool pin/complete when the host is free
    /// (Zene fires these between chat awaits, so try_lock succeeds).
    pub fn event_handler(host: Arc<Mutex<Self>>) -> EventHandler {
        Arc::new(move |event: AgentEvent| {
            let Ok(mut s) = host.try_lock() else {
                tracing::warn!("traject session busy; dropping agent event");
                return;
            };
            let traj_id = s.trajectory_id;
            match event {
                AgentEvent::ToolCall { .. } => {
                    if let Err(e) = s.driver.pin_for_tool_gap(traj_id) {
                        tracing::warn!(error = %e, "pin_for_tool_gap failed");
                    }
                }
                AgentEvent::ToolResult {
                    id,
                    name,
                    content,
                    is_error,
                    ..
                } => {
                    s.tool_steps += 1;
                    let call = TrajectToolCall {
                        name: name.clone(),
                        arguments: String::new(),
                        call_id: Some(id.clone()),
                    };
                    let result = TrajectToolResult {
                        call_id: Some(id),
                        name,
                        output: content,
                        is_error,
                    };
                    match s
                        .driver
                        .run_external_tool_step(traj_id, call, result, 120_000)
                    {
                        Ok(_) => {
                            info!(
                                trajectory = %traj_id,
                                tool_steps = s.tool_steps,
                                "tool step via driver/scheduler"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "external tool step failed");
                        }
                    }
                }
                _ => {}
            }
        })
    }
}

/// Zene `Provider` that drives every LLM turn through Traject Driver/Scheduler.
pub struct TrajectLlmProvider {
    host: Arc<Mutex<TrajectSession>>,
}

impl TrajectLlmProvider {
    pub fn new(host: Arc<Mutex<TrajectSession>>) -> Self {
        Self { host }
    }

    pub fn host(&self) -> Arc<Mutex<TrajectSession>> {
        Arc::clone(&self.host)
    }
}

#[async_trait]
impl Provider for TrajectLlmProvider {
    async fn chat(&self, request: ChatRequest) -> Result<ChatResponse> {
        let prompt = render_prompt(&request.messages, &request.tools);

        let mut s = self.host.lock().await;
        s.generate_steps += 1;
        let traj_id = s.trajectory_id;
        let max_tokens = s.max_tokens;
        let outcome = s
            .driver
            .run_generate_step(
                traj_id,
                GenerateDelta::from_text(prompt),
                Constraints {
                    temperature: Some(0.2),
                    top_p: Some(0.95),
                    ..Constraints::default()
                },
                max_tokens,
            )
            .await
            .map_err(|e| anyhow!("driver generate: {e}"))?;

        let cache_hits = s.driver.cache_hit_tokens(traj_id);
        let history_len = s
            .driver
            .manager
            .get(traj_id)
            .map(|t| t.history.len())
            .unwrap_or(0);
        let gen_steps = s.generate_steps;
        drop(s);

        let text = match &outcome {
            StepOutcome::Generated { text, .. } => text.clone(),
            other => return Err(anyhow!("expected Generated outcome, got {other:?}")),
        };
        let message = parse_assistant_message(&text);

        info!(
            trajectory = %traj_id,
            generate_steps = gen_steps,
            cache_hit_tokens = cache_hits,
            history = history_len,
            "generate step via driver/scheduler"
        );

        Ok(ChatResponse {
            message,
            usage: None,
        })
    }

    async fn chat_stream(
        &self,
        request: ChatRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
        let resp = self.chat(request).await?;
        let mut events = Vec::new();
        if let Some(text) = resp.message.content.clone() {
            if !text.is_empty() {
                events.push(Ok(StreamEvent::TextDelta(text)));
            }
        }
        if let Some(calls) = resp.message.tool_calls.clone() {
            for (index, call) in calls.into_iter().enumerate() {
                events.push(Ok(StreamEvent::ToolCallDelta {
                    index,
                    id: Some(call.id),
                    name: Some(call.name),
                    arguments: Some(call.arguments),
                }));
            }
        }
        events.push(Ok(StreamEvent::Done {
            usage: resp.usage,
        }));
        Ok(Box::pin(stream::iter(events)))
    }
}

fn render_prompt(messages: &[Message], tools: &[ToolDefinition]) -> String {
    let mut out = String::new();
    if !tools.is_empty() {
        out.push_str(TOOL_PROTOCOL);
        out.push_str("\nAvailable tools (OpenAI schema):\n");
        let schemas: Vec<Value> = tools.iter().map(|t| t.to_openai_tool()).collect();
        out.push_str(&serde_json::to_string_pretty(&schemas).unwrap_or_else(|_| "[]".into()));
        out.push_str("\n\n");
    }
    for m in messages {
        let role = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        };
        let mut body = m.content.clone().unwrap_or_default();
        if let Some(calls) = &m.tool_calls {
            if !body.is_empty() {
                body.push('\n');
            }
            body.push_str(
                &serde_json::to_string(&serde_json::json!({ "tool_calls": calls }))
                    .unwrap_or_default(),
            );
        }
        if m.role == Role::Tool {
            let name = m.name.as_deref().unwrap_or("tool");
            body = format!("Tool `{name}` result:\n{body}");
        }
        out.push_str(&format!("{role}: {body}\n"));
    }
    out
}

fn parse_assistant_message(text: &str) -> Message {
    if let Some(calls) = parse_tool_calls(text) {
        return Message::assistant_with_tools(None, calls);
    }
    Message::assistant(text.trim())
}

fn parse_tool_calls(text: &str) -> Option<Vec<ToolCall>> {
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
        out.push(ToolCall {
            id,
            name,
            arguments,
        });
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

fn extract_json_object(text: &str) -> Option<Value> {
    let trimmed = text.trim();
    if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
        return Some(v);
    }
    let start = trimmed.find('{')?;
    let end = trimmed.rfind('}')?;
    if end > start {
        serde_json::from_str(&trimmed[start..=end]).ok()
    } else {
        None
    }
}

/// Snapshot + finish trajectory after the agent loop returns.
pub async fn finish_session_snapshot(host: &Arc<Mutex<TrajectSession>>) -> Result<Trajectory> {
    let mut s = host.lock().await;
    let traj_id = s.trajectory_id;
    let traj = s
        .driver
        .manager
        .get(traj_id)
        .map_err(|e| anyhow!("get trajectory: {e}"))?;
    let generate_steps = s.generate_steps;
    let tool_steps = s.tool_steps;
    let cache = s.driver.cache_hit_tokens(traj_id);
    let mut snapshot = traj;
    if !snapshot.is_finished() {
        let _ = s.driver.manager.finish(traj_id);
        snapshot.state = traject_core::TrajectoryState::Finished;
    }
    snapshot
        .memory
        .set_slot("generate_steps", generate_steps.to_string());
    snapshot
        .memory
        .set_slot("tool_steps", tool_steps.to_string());
    snapshot
        .memory
        .set_slot("total_cache_hit_tokens", cache.to_string());
    s.driver.memory.release_trajectory(traj_id);
    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_tool_json() {
        let text = r#"{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Glob","arguments":"{\"pattern\":\"*.txt\"}"}}]}"#;
        let calls = parse_tool_calls(text).expect("parse");
        assert_eq!(calls[0].name, "Glob");
    }
}
