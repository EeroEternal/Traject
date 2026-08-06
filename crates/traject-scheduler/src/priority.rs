use serde::{Deserialize, Serialize};
use traject_core::{PinInfo, PinReason, TrajectoryId, TrajectoryPriority};

/// Why a step is eligible this tick, ordered high → low.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SchedulableKind {
    /// Already decoding — never preempt lightly.
    ActiveDecode = 0,
    /// Tool just returned; resume Generate ASAP for pin locality.
    PostToolGenerate = 1,
    /// High-priority new / resumed trajectory.
    HighPriorityNew = 2,
    /// Async tool execution.
    Tool = 3,
    /// Ordinary newly submitted trajectory.
    NormalNew = 4,
}

/// Composite sort key for the ready queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SchedPriority {
    pub kind: SchedulableKind,
    pub trajectory_priority: TrajectoryPriority,
    pub fairness_credit: i64,
}

impl SchedPriority {
    pub fn new(
        kind: SchedulableKind,
        trajectory_priority: TrajectoryPriority,
        fairness_credit: i64,
    ) -> Self {
        Self {
            kind,
            trajectory_priority,
            fairness_credit,
        }
    }
}

impl PartialOrd for SchedPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SchedPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lower SchedulableKind discriminant = higher urgency.
        self.kind
            .cmp(&other.kind)
            .then_with(|| other.trajectory_priority.cmp(&self.trajectory_priority))
            .then_with(|| other.fairness_credit.cmp(&self.fairness_credit))
    }
}

/// Pin / unpin advice from the scheduler after a tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinDecision {
    pub trajectory_id: TrajectoryId,
    pub action: PinAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PinAction {
    Pin {
        until_ms: u64,
        reason: PinReason,
        strength: u8,
    },
    Unpin,
    Keep(PinInfo),
}

/// Policy knobs for computing pin TTL when entering WaitingTool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinPolicy {
    pub base_ttl_ms: u64,
    pub max_ttl_ms: u64,
    pub min_ttl_ms: u64,
    /// Extra TTL multiplier from historical tool latency (p95 / base).
    pub latency_factor: f32,
}

impl Default for PinPolicy {
    fn default() -> Self {
        Self {
            base_ttl_ms: 5_000,
            max_ttl_ms: 60_000,
            min_ttl_ms: 500,
            latency_factor: 1.5,
        }
    }
}

impl PinPolicy {
    pub fn ttl_for_tool_latency(&self, observed_p95_ms: Option<u64>) -> u64 {
        let raw = match observed_p95_ms {
            Some(p95) => ((p95 as f32) * self.latency_factor) as u64,
            None => self.base_ttl_ms,
        };
        raw.clamp(self.min_ttl_ms, self.max_ttl_ms)
    }
}
