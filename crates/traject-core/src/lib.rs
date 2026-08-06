//! Traject core abstractions.
//!
//! Trajectory is the first-class unit of scheduling and optimization.
//! Inference is one kind of Step; tool execution is another.

mod error;
mod memory;
mod step;
mod trajectory;

pub use error::{Result, TrajectError};
pub use memory::{AgentMemory, PinInfo, PinReason, PrefixNodeId, Scratchpad};
pub use step::{
    Constraints, ControlKind, FinishReason, GenerateDelta, Step, StepId, StepOutcome, ToolCall,
    ToolResult,
};
pub use trajectory::{
    TenantId, Trajectory, TrajectoryConfig, TrajectoryId, TrajectoryPriority, TrajectoryState,
};
