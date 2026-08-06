use serde::{Deserialize, Serialize};
use traject_core::{Step, StepId, TrajectoryId};

use crate::budget::{BudgetSnapshot, SchedulerBudget};
use crate::priority::{PinDecision, PinPolicy, SchedPriority, SchedulableKind};

/// Tunables for the unified scheduler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub budget: SchedulerBudget,
    pub pin_policy: PinPolicy,
    /// Default chunk size for interruptible Generate execution.
    pub chunk_tokens: u32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            budget: SchedulerBudget::default(),
            pin_policy: PinPolicy::default(),
            chunk_tokens: 64,
        }
    }
}

/// A Step that is eligible for the current tick.
#[derive(Debug, Clone)]
pub struct ReadyStep {
    pub trajectory_id: TrajectoryId,
    pub step: Step,
    pub priority: SchedPriority,
    /// Tokens already produced in an in-flight Generate (for ActiveDecode).
    pub decoded_so_far: u32,
}

/// What the scheduler wants executors to do this tick.
#[derive(Debug, Clone)]
pub enum TickAction {
    /// Run up to `chunk_tokens` more tokens on a Generate step.
    RunGenerateChunk {
        trajectory_id: TrajectoryId,
        step_id: StepId,
        chunk_tokens: u32,
    },
    /// Launch (or continue) async tool execution.
    RunTool {
        trajectory_id: TrajectoryId,
        step_id: StepId,
    },
    /// Pure runtime control — no GPU / tool budget.
    RunControl {
        trajectory_id: TrajectoryId,
        step_id: StepId,
    },
}

/// Result of one scheduler iteration.
#[derive(Debug, Clone)]
pub struct SchedulerTick {
    pub actions: Vec<TickAction>,
    pub pin_decisions: Vec<PinDecision>,
    pub budget: BudgetSnapshot,
}

/// Events fed into the scheduler between ticks.
#[derive(Debug, Clone)]
pub enum SchedEvent {
    StepSubmitted {
        trajectory_id: TrajectoryId,
        step: Step,
        kind: SchedulableKind,
        priority: SchedPriority,
    },
    GenerateChunkDone {
        trajectory_id: TrajectoryId,
        step_id: StepId,
        tokens_produced: u32,
        finished: bool,
    },
    ToolFinished {
        trajectory_id: TrajectoryId,
        step_id: StepId,
    },
    ToolStarted {
        trajectory_id: TrajectoryId,
        step_id: StepId,
    },
    MemoryPressure {
        high: bool,
    },
}

/// Unified Trajectory Step scheduler.
///
/// Main loop contract:
/// ```text
/// loop {
///   1. collect ready steps (decode / post-tool generate / tools / new)
///   2. sort by SchedPriority
///   3. fill token_budget + tool_concurrency_budget
///   4. emit TickActions (chunked generate / async tool)
///   5. apply pin decisions
/// }
/// ```
#[derive(Debug)]
pub struct Scheduler {
    config: SchedulerConfig,
    budget: SchedulerBudget,
    ready: Vec<ReadyStep>,
    memory_pressure_high: bool,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let budget = config.budget.clone();
        Self {
            config,
            budget,
            ready: Vec::new(),
            memory_pressure_high: false,
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn budget_snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot::from(&self.budget)
    }

    pub fn push_ready(&mut self, item: ReadyStep) {
        self.ready.push(item);
    }

    pub fn has_step(&self, trajectory_id: TrajectoryId, step_id: StepId) -> bool {
        self.ready
            .iter()
            .any(|r| r.trajectory_id == trajectory_id && r.step.id() == step_id)
    }

    pub fn ready_len(&self) -> usize {
        self.ready.len()
    }

    pub fn ingest(&mut self, event: SchedEvent) {
        match event {
            SchedEvent::StepSubmitted {
                trajectory_id,
                step,
                kind: _,
                priority,
            } => {
                self.ready.push(ReadyStep {
                    trajectory_id,
                    step,
                    priority,
                    decoded_so_far: 0,
                });
            }
            SchedEvent::GenerateChunkDone {
                trajectory_id,
                step_id,
                tokens_produced,
                finished,
            } => {
                self.budget.tokens.remaining = self
                    .budget
                    .tokens
                    .remaining
                    .saturating_add(0); // consumption already accounted at schedule time
                let _ = (trajectory_id, step_id, tokens_produced, finished);
            }
            SchedEvent::ToolStarted { .. } => {
                // in_flight already incremented when action was emitted
            }
            SchedEvent::ToolFinished { .. } => {
                self.budget.tools.release();
            }
            SchedEvent::MemoryPressure { high } => {
                self.memory_pressure_high = high;
            }
        }
    }

    /// One scheduling iteration.
    pub fn tick(&mut self) -> SchedulerTick {
        self.budget.tokens.refill();
        self.ready.sort_by(|a, b| a.priority.cmp(&b.priority));

        let mut actions = Vec::new();
        let mut pin_decisions = Vec::new();
        let mut still_ready = Vec::new();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);

        for item in self.ready.drain(..) {
            match item.step {
                Step::Generate {
                    id,
                    max_tokens,
                    delta,
                    constraints,
                } => {
                    let chunk = self
                        .config
                        .chunk_tokens
                        .min(max_tokens.saturating_sub(item.decoded_so_far))
                        .max(1);
                    if self.budget.tokens.try_consume(chunk) {
                        actions.push(TickAction::RunGenerateChunk {
                            trajectory_id: item.trajectory_id,
                            step_id: id,
                            chunk_tokens: chunk,
                        });
                        // Do not re-queue here — the driver re-enqueues ActiveDecode
                        // after a partial chunk via decode_progress.
                        let _ = (delta, constraints);
                    } else {
                        still_ready.push(ReadyStep {
                            trajectory_id: item.trajectory_id,
                            step: Step::Generate {
                                id,
                                max_tokens,
                                delta,
                                constraints,
                            },
                            priority: item.priority,
                            decoded_so_far: item.decoded_so_far,
                        });
                    }
                }
                Step::Tool { id, call, timeout_ms } => {
                    if self.budget.tools.try_acquire() {
                        // Advise pin for tool gap (driver applies + may refine with p95).
                        let ttl = self.config.pin_policy.base_ttl_ms;
                        pin_decisions.push(crate::priority::PinDecision {
                            trajectory_id: item.trajectory_id,
                            action: crate::priority::PinAction::Pin {
                                until_ms: now.saturating_add(ttl),
                                reason: traject_core::PinReason::WaitingTool,
                                strength: 2,
                            },
                        });
                        actions.push(TickAction::RunTool {
                            trajectory_id: item.trajectory_id,
                            step_id: id,
                        });
                        let _ = call;
                    } else {
                        still_ready.push(ReadyStep {
                            trajectory_id: item.trajectory_id,
                            step: Step::Tool { id, call, timeout_ms },
                            priority: item.priority,
                            decoded_so_far: item.decoded_so_far,
                        });
                    }
                }
                Step::Control { id, kind, payload } => {
                    actions.push(TickAction::RunControl {
                        trajectory_id: item.trajectory_id,
                        step_id: id,
                    });
                    let _ = (kind, payload);
                }
            }
        }

        self.ready = still_ready;

        // Under memory pressure, emit unpin advice for weak pins (driver enforces).
        if self.memory_pressure_high {
            // Driver walks live trajs; keep marker decision empty here.
        }

        SchedulerTick {
            actions,
            pin_decisions,
            budget: BudgetSnapshot::from(&self.budget),
        }
    }

    pub fn adjust_token_capacity(&mut self, capacity: u32) {
        self.budget.tokens.adjust_capacity(capacity);
        self.config.budget.tokens.capacity = capacity;
    }

    pub fn pin_policy(&self) -> &PinPolicy {
        &self.config.pin_policy
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use traject_core::{Constraints, GenerateDelta, TrajectoryPriority};

    fn gen_ready(kind: SchedulableKind) -> ReadyStep {
        ReadyStep {
            trajectory_id: TrajectoryId::new(),
            step: Step::generate(GenerateDelta::from_text("x"), Constraints::default(), 128),
            priority: SchedPriority::new(kind, TrajectoryPriority(0), 0),
            decoded_so_far: 0,
        }
    }

    #[test]
    fn decode_preferred_over_new() {
        let mut sched = Scheduler::new(SchedulerConfig {
            budget: SchedulerBudget {
                tokens: crate::TokenBudget::new(64),
                tools: crate::ToolConcurrencyBudget::new(8),
            },
            ..SchedulerConfig::default()
        });
        let normal = gen_ready(SchedulableKind::NormalNew);
        let decode = gen_ready(SchedulableKind::ActiveDecode);
        let normal_id = normal.trajectory_id;
        let decode_id = decode.trajectory_id;
        sched.push_ready(normal);
        sched.push_ready(decode);

        let tick = sched.tick();
        assert!(!tick.actions.is_empty());
        match &tick.actions[0] {
            TickAction::RunGenerateChunk { trajectory_id, .. } => {
                assert_eq!(*trajectory_id, decode_id);
                assert_ne!(*trajectory_id, normal_id);
            }
            _ => panic!("expected generate chunk"),
        }
    }

    #[test]
    fn tool_respects_concurrency_budget() {
        let mut sched = Scheduler::new(SchedulerConfig {
            budget: SchedulerBudget {
                tokens: crate::TokenBudget::new(0),
                tools: crate::ToolConcurrencyBudget::new(1),
            },
            ..SchedulerConfig::default()
        });
        for _ in 0..3 {
            sched.push_ready(ReadyStep {
                trajectory_id: TrajectoryId::new(),
                step: Step::tool(
                    traject_core::ToolCall {
                        name: "x".into(),
                        arguments: "{}".into(),
                        call_id: None,
                    },
                    1000,
                ),
                priority: SchedPriority::new(
                    SchedulableKind::Tool,
                    TrajectoryPriority(0),
                    0,
                ),
                decoded_so_far: 0,
            });
        }
        let tick = sched.tick();
        assert_eq!(tick.actions.len(), 1);
        assert_eq!(sched.ready_len(), 2);
    }
}
