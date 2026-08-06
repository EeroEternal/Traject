//! Traject-owned LLM provider for the vendored Zene agent.
//!
//! Each Zene `chat` call becomes a Generate step on a shared Trajectory.
//! Tool call/result events become Tool steps. Inference goes through
//! `InferenceBackend` (typically sglang-lite `:9001`) with trajectory/session/
//! prefix metadata so the engine can track the agent session.

use std::pin::Pin;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use futures::{stream, Stream};
use parking_lot::Mutex;
use serde_json::Value;
use tracing::info;
use traject_core::{
    Constraints, FinishReason, GenerateDelta, PinInfo, PinReason, Step, StepOutcome,
    ToolCall as TrajectToolCall, ToolResult as TrajectToolResult, Trajectory, TrajectoryConfig,
    TrajectoryId,
};
use traject_inference::{ChunkRequest, InferenceBackend};
use traject_memory::MemoryManager;
use zene_core::{AgentEvent, EventHandler};
use zene_llm::{
    ChatRequest, ChatResponse, Message, Provider, Role, StreamEvent, ToolCall, ToolDefinition,
};

const TOOL_PROTOCOL: &str = r#"You are a coding agent with tools. When you need a tool, reply with ONLY this JSON (no markdown fences):
{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"TOOL_NAME","arguments":"{\"arg\":\"value\"}"}}]}
When you are done and do not need tools, reply with plain text only (no tool_calls JSON).
"#;

/// Shared host state for one agent prompt (one Trajectory).
pub struct TrajectSession {
    pub trajectory: Trajectory,
    pub memory: MemoryManager,
    pub backend: Arc<dyn InferenceBackend>,
    pub session_id: String,
    pub model: String,
    pub max_tokens: u32,
    pub generate_steps: u32,
    pub tool_steps: u32,
    pub total_cache_hit_tokens: u32,
}

impl TrajectSession {
    pub fn new(backend: Arc<dyn InferenceBackend>, model: impl Into<String>) -> Self {
        let mut trajectory = Trajectory::create(TrajectoryConfig {
            tenant: traject_core::TenantId::new("zene"),
            ..TrajectoryConfig::default()
        });
        let _ = trajectory.start();
        let mut memory = MemoryManager::new(8192);
        let root = memory.root_id();
        let _ = memory.bind_trajectory(trajectory.id, root);
        trajectory.bind_prefix(root);
        let session_id = trajectory.id.to_string();
        Self {
            trajectory,
            memory,
            backend,
            session_id,
            model: model.into(),
            max_tokens: 1024,
            generate_steps: 0,
            tool_steps: 0,
            total_cache_hit_tokens: 0,
        }
    }

    pub fn trajectory_id(&self) -> TrajectoryId {
        self.trajectory.id
    }

    pub fn event_handler(host: Arc<Mutex<Self>>) -> EventHandler {
        Arc::new(move |event: AgentEvent| {
            let mut s = host.lock();
            match event {
                AgentEvent::ToolCall { .. } => {
                    if let Some(prefix) = s.trajectory.current_prefix {
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis() as u64)
                            .unwrap_or(0);
                        let pin = PinInfo::pin_until(now + 120_000, PinReason::WaitingTool, 1);
                        let _ = s.memory.pin_node(prefix, pin);
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
                    let step = Step::tool(
                        TrajectToolCall {
                            name: name.clone(),
                            arguments: String::new(),
                            call_id: Some(id.clone()),
                        },
                        120_000,
                    );
                    let step_id = step.id();
                    let _ = s.trajectory.submit_step(step);
                    let _ = s.trajectory.complete_step(StepOutcome::ToolDone {
                        step_id,
                        result: TrajectToolResult {
                            call_id: Some(id),
                            name,
                            output: content,
                            is_error,
                        },
                    });
                    if let Some(prefix) = s.trajectory.current_prefix {
                        let _ = s.memory.unpin_node(prefix);
                    }
                    info!(
                        trajectory = %s.trajectory.id,
                        tool_steps = s.tool_steps,
                        "recorded tool step on trajectory"
                    );
                }
                _ => {}
            }
        })
    }
}

/// Zene `Provider` that drives Traject inference + Trajectory ledger.
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
        let (backend, chunk_req, step_id) = {
            let mut s = self.host.lock();
            s.generate_steps += 1;
            let prompt = render_prompt(&request.messages, &request.tools);
            let step = Step::generate(
                GenerateDelta::from_text(prompt.clone()),
                Constraints {
                    temperature: Some(0.2),
                    top_p: Some(0.95),
                    ..Constraints::default()
                },
                s.max_tokens,
            );
            let step_id = step.id();
            s.trajectory
                .submit_step(step)
                .map_err(|e| anyhow!("submit generate: {e}"))?;

            let prefix = s.trajectory.current_prefix;
            let chunk_req = ChunkRequest {
                trajectory_id: s.trajectory.id,
                step_id,
                prefix,
                delta: GenerateDelta::from_text(prompt),
                constraints: Constraints {
                    temperature: Some(0.2),
                    top_p: Some(0.95),
                    ..Constraints::default()
                },
                chunk_tokens: s.max_tokens,
                decoded_so_far: 0,
                max_tokens: s.max_tokens,
                session_id: Some(s.session_id.clone()),
                prefix_hint: prefix.map(|p| p.to_string()),
            };
            (Arc::clone(&s.backend), chunk_req, step_id)
        };

        let result = backend
            .generate_chunk(chunk_req)
            .await
            .map_err(|e| anyhow!("traject inference: {e}"))?;

        let message = parse_assistant_message(&result.text);
        let finish = result.finish_reason.unwrap_or(FinishReason::Stop);
        let tool_call = message.tool_calls.as_ref().and_then(|calls| {
            calls.first().map(|c| TrajectToolCall {
                name: c.name.clone(),
                arguments: c.arguments.clone(),
                call_id: Some(c.id.clone()),
            })
        });

        {
            let mut s = self.host.lock();
            s.total_cache_hit_tokens += result.cache_hit_tokens;
            let hit = result.cache_hit_tokens.to_string();
            let total = s.total_cache_hit_tokens.to_string();
            let traj_id = s.trajectory.id;
            s.trajectory
                .memory
                .set_slot("last_cache_hit_tokens", &hit);
            s.trajectory
                .memory
                .set_slot("total_cache_hit_tokens", &total);
            if !result.token_ids.is_empty() {
                if let Ok(node) = s.memory.append_tokens(traj_id, result.token_ids.clone()) {
                    s.trajectory.bind_prefix(node);
                }
            }
            s.trajectory
                .complete_step(StepOutcome::Generated {
                    step_id,
                    text: result.text.clone(),
                    token_ids: result.token_ids,
                    finish_reason: if tool_call.is_some() {
                        FinishReason::ToolCall
                    } else {
                        finish
                    },
                    tool_call,
                })
                .map_err(|e| anyhow!("complete generate: {e}"))?;
            info!(
                trajectory = %s.trajectory.id,
                generate_steps = s.generate_steps,
                cache_hit_tokens = result.cache_hit_tokens,
                total_cache_hit = s.total_cache_hit_tokens,
                history = s.trajectory.history.len(),
                "recorded generate step on trajectory"
            );
        }

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

/// Helper used by tests / runner to finish a trajectory cleanly.
pub fn finish_session(host: &Arc<Mutex<TrajectSession>>) -> Result<Trajectory> {
    let mut s = host.lock();
    s.trajectory
        .finish()
        .context("finish trajectory")?;
    Ok(s.trajectory.clone())
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
