use async_trait::async_trait;
use traject_core::{
    Constraints, FinishReason, GenerateDelta, PrefixNodeId, Result, StepId, ToolCall, TrajectoryId,
};

use crate::{ChunkRequest, ChunkResult};

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub trajectory_id: TrajectoryId,
    pub step_id: StepId,
    pub prefix: Option<PrefixNodeId>,
    pub delta: GenerateDelta,
    pub constraints: Constraints,
    pub max_tokens: u32,
    /// Stable agent session across Generate/Tool turns.
    pub session_id: Option<String>,
    /// Engine radix / V4 prefix handle from MemoryManager.
    pub prefix_hint: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GenerateResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub finish_reason: FinishReason,
    pub tool_call: Option<ToolCall>,
    pub new_prefix: Option<PrefixNodeId>,
}

/// Pluggable model backend.
#[async_trait]
pub trait InferenceBackend: Send + Sync {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult>;

    /// Free physical resources for a prefix handle (paged KV / engine snapshots).
    /// Default: no-op for backends without owned device memory.
    async fn free_prefix(&self, prefix_id: &str, session_id: Option<&str>) -> Result<()> {
        let _ = (prefix_id, session_id);
        Ok(())
    }
}

/// Thin engine over a backend; owns chunking policy.
pub struct InferenceEngine {
    backend: std::sync::Arc<dyn InferenceBackend>,
    default_chunk: u32,
}

impl InferenceEngine {
    pub fn new(backend: impl InferenceBackend + 'static, default_chunk: u32) -> Self {
        Self {
            backend: std::sync::Arc::new(backend),
            default_chunk,
        }
    }

    pub fn from_arc(backend: std::sync::Arc<dyn InferenceBackend>, default_chunk: u32) -> Self {
        Self {
            backend,
            default_chunk,
        }
    }

    pub fn backend(&self) -> &dyn InferenceBackend {
        self.backend.as_ref()
    }

    pub fn backend_arc(&self) -> std::sync::Arc<dyn InferenceBackend> {
        std::sync::Arc::clone(&self.backend)
    }

    pub async fn run_chunk(
        &self,
        req: &GenerateRequest,
        chunk_tokens: u32,
        decoded_so_far: u32,
    ) -> Result<ChunkResult> {
        let chunk = ChunkRequest {
            trajectory_id: req.trajectory_id,
            step_id: req.step_id,
            prefix: req.prefix,
            delta: req.delta.clone(),
            constraints: req.constraints.clone(),
            chunk_tokens: chunk_tokens.min(self.default_chunk.max(chunk_tokens)),
            decoded_so_far,
            max_tokens: req.max_tokens,
            session_id: req.session_id.clone(),
            prefix_hint: req
                .prefix_hint
                .clone()
                .or_else(|| req.prefix.map(|p| p.to_string())),
        };
        self.backend.generate_chunk(chunk).await
    }

    pub async fn free_prefix(&self, prefix_id: &str, session_id: Option<&str>) -> Result<()> {
        self.backend.free_prefix(prefix_id, session_id).await
    }
}

/// Stub behavior for bringing up the runtime without weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StubMode {
    /// Always emit a final answer.
    #[default]
    AlwaysStop,
    /// Always emit a tool call (policy must cap steps).
    AlwaysTool,
    /// First N generates emit tool calls, then stop.
    ToolThenStop {
        remaining_tools: u32,
    },
    /// Emit `chunks` partial results then stop (tests interruptible decode).
    MultiChunk {
        chunks: u32,
    },
}

/// Deterministic stub for bringing up the runtime without a real model.
#[derive(Debug)]
pub struct StubBackend {
    pub mode: StubMode,
    /// Per-trajectory remaining tool emissions for ToolThenStop.
    tool_budget: std::sync::Mutex<std::collections::HashMap<TrajectoryId, u32>>,
    chunk_progress: std::sync::Mutex<std::collections::HashMap<StepId, u32>>,
}

impl Default for StubBackend {
    fn default() -> Self {
        Self {
            mode: StubMode::AlwaysStop,
            tool_budget: std::sync::Mutex::new(std::collections::HashMap::new()),
            chunk_progress: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    }
}

impl StubBackend {
    pub fn always_stop() -> Self {
        Self::default()
    }

    pub fn always_tool() -> Self {
        Self {
            mode: StubMode::AlwaysTool,
            ..Self::default()
        }
    }

    pub fn tool_then_stop(n: u32) -> Self {
        Self {
            mode: StubMode::ToolThenStop { remaining_tools: n },
            ..Self::default()
        }
    }

    pub fn multi_chunk(chunks: u32) -> Self {
        Self {
            mode: StubMode::MultiChunk {
                chunks: chunks.max(1),
            },
            ..Self::default()
        }
    }

    fn should_emit_tool(&self, traj: TrajectoryId) -> bool {
        match self.mode {
            StubMode::AlwaysStop | StubMode::MultiChunk { .. } => false,
            StubMode::AlwaysTool => true,
            StubMode::ToolThenStop { remaining_tools } => {
                let mut map = self.tool_budget.lock().expect("stub budget lock");
                let left = map.entry(traj).or_insert(remaining_tools);
                if *left == 0 {
                    false
                } else {
                    *left -= 1;
                    true
                }
            }
        }
    }
}

#[async_trait]
impl InferenceBackend for StubBackend {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult> {
        if let StubMode::MultiChunk { chunks } = self.mode {
            let mut map = self.chunk_progress.lock().expect("chunk progress");
            let seen = map.entry(req.step_id).or_insert(0);
            *seen += 1;
            let idx = *seen;
            let produced = req.chunk_tokens.max(1);
            let finished = idx >= chunks;
            return Ok(ChunkResult {
                text: if finished {
                    "ok".into()
                } else {
                    "…".into()
                },
                token_ids: (0..produced).collect(),
                tokens_produced: produced,
                finished,
                finish_reason: if finished {
                    Some(FinishReason::Stop)
                } else {
                    None
                },
                tool_call: None,
                new_prefix: None,
                cache_hit_tokens: 0,
            });
        }

        // Default: one chunk completes the Generate step.
        let produced = req
            .chunk_tokens
            .min(req.max_tokens.saturating_sub(req.decoded_so_far))
            .max(1);

        let emit_tool = self.should_emit_tool(req.trajectory_id);
        let (text, finish_reason, tool_call) = if emit_tool {
            (
                "tool_call".to_string(),
                FinishReason::ToolCall,
                Some(ToolCall {
                    name: "echo".into(),
                    arguments: "{\"msg\":\"stub\"}".into(),
                    call_id: Some(format!("call-{}", req.step_id)),
                }),
            )
        } else {
            ("ok".to_string(), FinishReason::Stop, None)
        };

        Ok(ChunkResult {
            text,
            token_ids: (0..produced).collect(),
            tokens_produced: produced,
            finished: true,
            finish_reason: Some(finish_reason),
            tool_call,
            new_prefix: None,
            cache_hit_tokens: 0,
        })
    }
}
