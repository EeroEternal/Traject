//! Unified scheduler: token-budget + tool-concurrency over Trajectory Steps.

mod budget;
mod latency;
mod pinning;
mod priority;
mod scheduler;

pub use budget::{BudgetSnapshot, SchedulerBudget, TokenBudget, ToolConcurrencyBudget};
pub use latency::{LatencyHistogram, ToolLatencyTracker};
pub use pinning::{apply_pin, pin_for_prefetch, pin_for_tool_wait, should_force_unpin};
pub use priority::{PinAction, PinDecision, PinPolicy, SchedPriority, SchedulableKind};
pub use scheduler::{
    ReadyStep, SchedEvent, Scheduler, SchedulerConfig, SchedulerTick, TickAction,
};
