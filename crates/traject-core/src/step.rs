use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Unique id for a Step within (or across) trajectories.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StepId(pub Uuid);

impl StepId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for StepId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for StepId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// One action on a Trajectory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Step {
    Generate {
        id: StepId,
        delta: GenerateDelta,
        constraints: Constraints,
        max_tokens: u32,
    },
    Tool {
        id: StepId,
        call: ToolCall,
        timeout_ms: u64,
    },
    Control {
        id: StepId,
        kind: ControlKind,
        payload: Option<String>,
    },
}

impl Step {
    pub fn id(&self) -> StepId {
        match self {
            Step::Generate { id, .. } | Step::Tool { id, .. } | Step::Control { id, .. } => *id,
        }
    }

    pub fn generate(delta: GenerateDelta, constraints: Constraints, max_tokens: u32) -> Self {
        Self::Generate {
            id: StepId::new(),
            delta,
            constraints,
            max_tokens,
        }
    }

    pub fn tool(call: ToolCall, timeout_ms: u64) -> Self {
        Self::Tool {
            id: StepId::new(),
            call,
            timeout_ms,
        }
    }

    pub fn control(kind: ControlKind, payload: Option<String>) -> Self {
        Self::Control {
            id: StepId::new(),
            kind,
            payload,
        }
    }

    pub fn is_generate(&self) -> bool {
        matches!(self, Step::Generate { .. })
    }

    pub fn is_tool(&self) -> bool {
        matches!(self, Step::Tool { .. })
    }
}

/// Prompt / context delta relative to the current prefix node.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerateDelta {
    /// Token ids already known to the engine (preferred path).
    pub token_ids: Vec<u32>,
    /// Optional text form for backends that tokenize themselves.
    pub text: Option<String>,
}

impl GenerateDelta {
    pub fn from_text(text: impl Into<String>) -> Self {
        Self {
            token_ids: Vec::new(),
            text: Some(text.into()),
        }
    }

    pub fn from_tokens(token_ids: Vec<u32>) -> Self {
        Self {
            token_ids,
            text: None,
        }
    }
}

/// Decoding / structured-output constraints for a Generate step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constraints {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
    /// Optional JSON schema / grammar handle for constrained decoding.
    pub grammar: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
    pub call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: Option<String>,
    pub name: String,
    pub output: String,
    pub is_error: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ControlKind {
    Reflect,
    Plan,
    Branch,
    EarlyStop,
}

/// Outcome produced after a Step finishes (fed back into the Trajectory).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StepOutcome {
    Generated {
        step_id: StepId,
        text: String,
        token_ids: Vec<u32>,
        finish_reason: FinishReason,
        /// Parsed tool call if the model emitted one.
        tool_call: Option<ToolCall>,
    },
    ToolDone {
        step_id: StepId,
        result: ToolResult,
    },
    ControlDone {
        step_id: StepId,
        kind: ControlKind,
        note: Option<String>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FinishReason {
    Stop,
    Length,
    ToolCall,
    Cancelled,
}
