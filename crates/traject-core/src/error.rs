use thiserror::Error;

pub type Result<T> = std::result::Result<T, TrajectError>;

#[derive(Debug, Error)]
pub enum TrajectError {
    #[error("invalid trajectory state transition: {from:?} → {to:?}")]
    InvalidTransition {
        from: crate::TrajectoryState,
        to: crate::TrajectoryState,
    },

    #[error("trajectory not found: {0}")]
    TrajectoryNotFound(crate::TrajectoryId),

    #[error("step not found: {0}")]
    StepNotFound(crate::StepId),

    #[error("prefix node not found: {0}")]
    PrefixNotFound(crate::PrefixNodeId),

    #[error("tool timed out: {tool}")]
    ToolTimeout { tool: String },

    #[error("tool failed: {tool}: {reason}")]
    ToolFailed { tool: String, reason: String },

    #[error("budget exhausted: {kind}")]
    BudgetExhausted { kind: &'static str },

    #[error("inference backend error: {0}")]
    Inference(String),

    #[error("memory pressure: {0}")]
    MemoryPressure(String),

    #[error("{0}")]
    Other(String),
}
