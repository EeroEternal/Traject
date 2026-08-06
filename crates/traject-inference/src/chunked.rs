use traject_core::{
    Constraints, FinishReason, GenerateDelta, PrefixNodeId, StepId, ToolCall, TrajectoryId,
};

#[derive(Debug, Clone)]
pub struct ChunkRequest {
    pub trajectory_id: TrajectoryId,
    pub step_id: StepId,
    pub prefix: Option<PrefixNodeId>,
    pub delta: GenerateDelta,
    pub constraints: Constraints,
    pub chunk_tokens: u32,
    pub decoded_so_far: u32,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub text: String,
    pub token_ids: Vec<u32>,
    pub tokens_produced: u32,
    pub finished: bool,
    pub finish_reason: Option<FinishReason>,
    pub tool_call: Option<ToolCall>,
    pub new_prefix: Option<PrefixNodeId>,
}
