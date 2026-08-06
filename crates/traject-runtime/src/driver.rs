use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use tracing::{debug, info, info_span, warn, Instrument};
use traject_core::{
    Constraints, FinishReason, GenerateDelta, Result, Step, StepId, StepOutcome, ToolCall,
    ToolResult, TrajectoryConfig, TrajectoryId, TrajectError,
};
use traject_inference::{
    EnginePrefixClient, HttpOpenAiBackend, InferenceBackend, InferenceEngine, KernelSmokeBackend,
    SglangLiteEngineBackend, StubBackend, StubMode,
};
use traject_memory::MemoryManager;
use traject_policy::{Policy, ReActPolicy};
use traject_scheduler::{
    apply_pin, pin_for_prefetch, pin_for_tool_wait, should_force_unpin, PinAction, ReadyStep,
    SchedPriority, SchedulableKind, Scheduler, SchedulerConfig, TickAction, ToolLatencyTracker,
};
use traject_tool::{
    EchoTool, ToolFinishedEvent, ToolHandler, ToolRegistry, ToolRuntime, ToolRuntimeConfig,
};

use crate::manager::TrajectoryManager;

#[derive(Clone)]
pub struct DriverConfig {
    pub scheduler: SchedulerConfig,
    pub max_ticks: usize,
    pub block_capacity: usize,
    pub eviction_pressure: f32,
}

impl Default for DriverConfig {
    fn default() -> Self {
        Self {
            scheduler: SchedulerConfig::default(),
            max_ticks: 1024,
            block_capacity: 4096,
            eviction_pressure: 0.9,
        }
    }
}

/// Phase 0/1 driver: multi-trajectory loop with scheduler + memory + inference + tools.
///
/// Two drive modes:
/// - **Policy-driven** (`run_until_finished`): ReAct/Plan policy submits next steps.
/// - **External** (`run_generate_step` / `run_external_tool_step`): Zene (or other agents)
///   submit each step; the Driver/Scheduler still owns scheduling, pin, prefix, inference.
pub struct Driver {
    pub manager: TrajectoryManager,
    pub scheduler: Scheduler,
    pub memory: MemoryManager,
    pub policy: Arc<dyn Policy>,
    pub engine: InferenceEngine,
    tool_runtime: ToolRuntime,
    tool_rx: tokio::sync::mpsc::UnboundedReceiver<ToolFinishedEvent>,
    config: DriverConfig,
    decode_progress: HashMap<(TrajectoryId, StepId), u32>,
    /// Steps already handed to executors (avoid double enqueue / double spawn).
    in_flight: HashSet<(TrajectoryId, StepId)>,
    /// Live trajectories the driver is responsible for.
    live: HashSet<TrajectoryId>,
    /// Externally-driven trajectories: do not auto-advance via policy.
    external: HashSet<TrajectoryId>,
    /// Accumulated engine cache hits per trajectory (mirrors MemoryManager session).
    cache_hits: HashMap<TrajectoryId, u32>,
    /// Rolling tool latency samples → pin TTL (p95).
    tool_latency: ToolLatencyTracker,
    /// When a tool gap pin started (ms), for latency measurement.
    tool_pin_started: HashMap<TrajectoryId, u64>,
    /// Optional engine prefix pin/unpin client (sglang-lite).
    engine_prefix: Option<EnginePrefixClient>,
}

impl Driver {
    pub fn new(config: DriverConfig) -> Self {
        let mut registry = ToolRegistry::new();
        registry.register(Arc::new(EchoTool) as Arc<dyn ToolHandler>);
        let (tool_runtime, tool_rx) =
            ToolRuntime::new(Arc::new(registry), ToolRuntimeConfig::default());

        Self {
            manager: TrajectoryManager::new(),
            scheduler: Scheduler::new(config.scheduler.clone()),
            memory: MemoryManager::new(config.block_capacity),
            policy: Arc::new(ReActPolicy::new("You are a helpful agent.")),
            engine: InferenceEngine::new(StubBackend::default(), config.scheduler.chunk_tokens),
            tool_runtime,
            tool_rx,
            config,
            decode_progress: HashMap::new(),
            in_flight: HashSet::new(),
            live: HashSet::new(),
            external: HashSet::new(),
            cache_hits: HashMap::new(),
            tool_latency: ToolLatencyTracker::new(),
            tool_pin_started: HashMap::new(),
            engine_prefix: None,
        }
    }

    /// Wire sglang-lite `/v1/prefix/pin|unpin` for MemoryManager decisions.
    pub fn with_engine_prefix_client(mut self, base_url: &str) -> Self {
        self.engine_prefix = Some(EnginePrefixClient::new(base_url));
        self
    }

    pub fn with_policy(mut self, policy: Arc<dyn Policy>) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_backend(mut self, backend: impl InferenceBackend + 'static) -> Self {
        self.engine = InferenceEngine::new(backend, self.config.scheduler.chunk_tokens);
        self
    }

    pub fn with_backend_arc(mut self, backend: Arc<dyn InferenceBackend>) -> Self {
        self.engine = InferenceEngine::from_arc(backend, self.config.scheduler.chunk_tokens);
        self
    }

    pub fn set_engine_prefix_client(&mut self, base_url: &str) {
        self.engine_prefix = Some(EnginePrefixClient::new(base_url));
    }

    pub fn with_http_backend(self, base_url: &str, model: &str) -> Self {
        self.with_backend(HttpOpenAiBackend::new(base_url, model))
    }

    /// Native sglang-lite engine API (typically `:9001`).
    pub fn with_engine_backend(self, base_url: &str, model: &str) -> Self {
        self.with_backend(SglangLiteEngineBackend::new(base_url, model))
            .with_engine_prefix_client(base_url)
    }

    /// In-process kernel smoke (CPU ref or FlashInfer-backed KernelSmokeBackend).
    pub fn with_kernel_smoke(self, backend: KernelSmokeBackend) -> Self {
        self.with_backend(backend)
    }

    pub fn with_stub_mode(mut self, mode: StubMode) -> Self {
        let backend = match mode {
            StubMode::AlwaysStop => StubBackend::always_stop(),
            StubMode::AlwaysTool => StubBackend::always_tool(),
            StubMode::ToolThenStop { remaining_tools } => {
                StubBackend::tool_then_stop(remaining_tools)
            }
            StubMode::MultiChunk { chunks } => StubBackend::multi_chunk(chunks),
        };
        self.engine = InferenceEngine::new(backend, self.config.scheduler.chunk_tokens);
        self
    }

    /// Backward-compatible helper.
    pub fn with_stub_tool_calls(self, emit: bool) -> Self {
        if emit {
            self.with_stub_mode(StubMode::ToolThenStop {
                remaining_tools: 1,
            })
        } else {
            self.with_stub_mode(StubMode::AlwaysStop)
        }
    }

    pub fn create_trajectory(&mut self, config: TrajectoryConfig) -> TrajectoryId {
        let id = self.manager.create(config);
        let root = self.memory.root_id();
        let _ = self.memory.bind_trajectory(id, root);
        self.memory.bind_session(id, id.to_string());
        let _ = self.manager.with_mut(id, |t| t.bind_prefix(root));
        self.live.insert(id);
        info!(trajectory = %id, "created trajectory");
        id
    }

    /// Create a trajectory owned by an external agent (Zene). Policy will not auto-advance.
    pub fn create_external_trajectory(&mut self, config: TrajectoryConfig) -> TrajectoryId {
        let id = self.create_trajectory(config);
        self.external.insert(id);
        info!(trajectory = %id, "external trajectory (agent-driven)");
        id
    }

    pub fn is_external(&self, id: TrajectoryId) -> bool {
        self.external.contains(&id)
    }

    pub fn cache_hit_tokens(&self, id: TrajectoryId) -> u32 {
        self.cache_hits
            .get(&id)
            .copied()
            .unwrap_or_else(|| self.memory.total_cache_hits(id))
    }

    /// Run one Generate step through Scheduler → InferenceEngine → MemoryManager.
    /// Does not invoke policy afterward (caller / external agent decides the next step).
    pub async fn run_generate_step(
        &mut self,
        id: TrajectoryId,
        delta: GenerateDelta,
        constraints: Constraints,
        max_tokens: u32,
    ) -> Result<StepOutcome> {
        if !self.live.contains(&id) {
            self.live.insert(id);
        }
        self.external.insert(id);
        let step = Step::generate(delta, constraints, max_tokens);
        let step_id = step.id();
        self.manager.submit_step(id, step)?;
        self.run_until_step_complete(id, step_id).await
    }

    /// Record an externally-executed tool (Zene tools) as a Tool step:
    /// pin prefix → submit/complete through manager → prefetch pin → fairness.
    /// Sync so agent event handlers can record without spawning tasks.
    pub fn run_external_tool_step(
        &mut self,
        id: TrajectoryId,
        call: ToolCall,
        result: ToolResult,
        timeout_ms: u64,
    ) -> Result<StepOutcome> {
        if !self.live.contains(&id) {
            self.live.insert(id);
        }
        self.external.insert(id);
        let now = now_ms();
        let tool_name = call.name.clone();
        // Measure latency from earlier pin_for_tool_gap if present.
        if let Some(start) = self.tool_pin_started.remove(&id) {
            let elapsed = now.saturating_sub(start);
            self.tool_latency.record(&tool_name, elapsed);
            debug!(
                trajectory = %id,
                tool = %tool_name,
                elapsed_ms = elapsed,
                p95 = ?self.tool_latency.p95_for(Some(&tool_name)),
                "tool latency sample"
            );
        }
        let p95 = self.tool_latency.p95_for(Some(&tool_name));
        let traj = self.manager.get(id)?;
        let decision = pin_for_tool_wait(&traj, now, self.scheduler.pin_policy(), p95);
        self.apply_pin_decision(&decision)?;

        let step = Step::tool(call, timeout_ms);
        let step_id = step.id();
        self.manager.submit_step(id, step)?;

        // External tools are already finished; complete synchronously (no ToolRuntime spawn).
        let outcome = StepOutcome::ToolDone { step_id, result };
        self.manager.complete_step(id, outcome.clone())?;
        // Drop WaitingTool pin, then soft-pin for predicted next Generate (prefetch).
        self.manager.with_mut(id, |t| t.pin.unpin())?;
        let _ = self.memory.unpin_trajectory(id);
        let traj = self.manager.get(id)?;
        let prefetch = pin_for_prefetch(&traj, now_ms(), self.scheduler.pin_policy());
        self.apply_pin_decision(&prefetch)?;
        self.bump_fairness(id, 5);
        self.in_flight.remove(&(id, step_id));

        info!(
            trajectory = %id,
            step = %step_id,
            tool = %tool_name,
            p95_ms = ?p95,
            "external tool step completed via driver"
        );
        Ok(outcome)
    }

    /// Pin current prefix while the external agent is about to run tools.
    pub fn pin_for_tool_gap(&mut self, id: TrajectoryId) -> Result<()> {
        let now = now_ms();
        self.tool_pin_started.insert(id, now);
        let p95 = self.tool_latency.global_p95();
        let traj = self.manager.get(id)?;
        let decision = pin_for_tool_wait(&traj, now, self.scheduler.pin_policy(), p95);
        self.apply_pin_decision(&decision)?;
        Ok(())
    }

    fn apply_pin_decision(&mut self, decision: &traject_scheduler::PinDecision) -> Result<()> {
        let id = decision.trajectory_id;
        self.manager.with_mut(id, |t| apply_pin(t, decision))?;
        match &decision.action {
            PinAction::Pin {
                until_ms,
                reason,
                strength,
            } => {
                let pin = traject_core::PinInfo::pin_until(*until_ms, *reason, *strength);
                let _ = self.memory.pin_trajectory(id, pin);
                self.spawn_engine_pin(id, *until_ms, format!("{reason:?}"));
            }
            PinAction::Unpin => {
                let _ = self.memory.unpin_trajectory(id);
                self.spawn_engine_unpin(id);
            }
            PinAction::Keep(info) => {
                let _ = self.memory.pin_trajectory(id, info.clone());
            }
        }
        Ok(())
    }

    fn spawn_engine_pin(&self, id: TrajectoryId, until_ms: u64, reason: String) {
        let Some(client) = self.engine_prefix.clone() else {
            return;
        };
        let prefix = match self.memory.engine_prefix_hint(id) {
            Some(p) => p,
            None => return,
        };
        let session = self.memory.session_id(id).map(|s| s.to_string());
        let now = now_ms();
        let ttl = until_ms.saturating_sub(now).max(500);
        let traj = id.to_string();
        tokio::spawn(async move {
            client
                .pin(
                    &prefix,
                    session.as_deref(),
                    Some(&traj),
                    ttl,
                    &reason,
                )
                .await;
        });
    }

    fn spawn_engine_unpin(&self, id: TrajectoryId) {
        let Some(client) = self.engine_prefix.clone() else {
            return;
        };
        let prefix = match self.memory.engine_prefix_hint(id) {
            Some(p) => p,
            None => return,
        };
        tokio::spawn(async move {
            client.unpin(&prefix).await;
        });
    }

    pub fn finish_trajectory(&mut self, id: TrajectoryId) -> Result<()> {
        self.manager.finish(id)?;
        self.cleanup_trajectory(id);
        Ok(())
    }

    async fn run_until_step_complete(
        &mut self,
        id: TrajectoryId,
        step_id: StepId,
    ) -> Result<StepOutcome> {
        let span = info_span!("driver.step", %id, %step_id);
        async {
            for tick_idx in 0..self.config.max_ticks {
                // Already completed?
                if let Some(outcome) = self.completed_outcome(id, step_id)? {
                    return Ok(outcome);
                }

                self.drain_tool_events().await?;
                self.maybe_unpin_under_pressure();
                self.enqueue_active(id)?;
                let tick = self.scheduler.tick();
                for decision in &tick.pin_decisions {
                    let _ = self.apply_pin_decision(decision);
                }
                debug!(
                    tick_idx,
                    actions = tick.actions.len(),
                    pins = tick.pin_decisions.len(),
                    ready = self.scheduler.ready_len(),
                    "scheduler tick (external step)"
                );

                if tick.actions.is_empty() {
                    if self.in_flight.is_empty() {
                        // Nothing scheduled and nothing in flight — stall.
                        if self.manager.get(id)?.active_step.is_none() {
                            if let Some(outcome) = self.completed_outcome(id, step_id)? {
                                return Ok(outcome);
                            }
                            return Err(TrajectError::Other(format!(
                                "step {step_id} stalled with empty scheduler queue"
                            )));
                        }
                    }
                    tokio::task::yield_now().await;
                    continue;
                }

                for action in tick.actions {
                    // Only execute actions for this trajectory's step (multi-traj safe).
                    let for_us = match &action {
                        TickAction::RunGenerateChunk {
                            trajectory_id,
                            step_id: sid,
                            ..
                        }
                        | TickAction::RunTool {
                            trajectory_id,
                            step_id: sid,
                        }
                        | TickAction::RunControl {
                            trajectory_id,
                            step_id: sid,
                        } => *trajectory_id == id && *sid == step_id,
                    };
                    if for_us || matches!(action, TickAction::RunTool { trajectory_id, .. } if trajectory_id == id) {
                        self.execute_action(action).await?;
                    } else if matches!(action, TickAction::RunGenerateChunk { trajectory_id, .. } if trajectory_id == id)
                        || matches!(action, TickAction::RunControl { trajectory_id, .. } if trajectory_id == id)
                    {
                        self.execute_action(action).await?;
                    } else {
                        // Re-queue foreign actions by not executing; push back ready later.
                        // For single-focus external runs this is rare.
                        debug!("skipping non-focus scheduler action");
                    }
                }
            }
            Err(TrajectError::Other(format!(
                "driver exceeded max_ticks={} waiting for step {step_id}",
                self.config.max_ticks
            )))
        }
        .instrument(span)
        .await
    }

    fn completed_outcome(
        &self,
        id: TrajectoryId,
        step_id: StepId,
    ) -> Result<Option<StepOutcome>> {
        let traj = self.manager.get(id)?;
        for rec in traj.history.iter().rev() {
            let sid = match &rec.outcome {
                Some(StepOutcome::Generated { step_id, .. })
                | Some(StepOutcome::ToolDone { step_id, .. })
                | Some(StepOutcome::ControlDone { step_id, .. }) => *step_id,
                None => continue,
            };
            if sid == step_id {
                return Ok(rec.outcome.clone());
            }
        }
        Ok(None)
    }

    /// Run a single trajectory to completion.
    pub async fn run_until_finished(&mut self, id: TrajectoryId) -> Result<()> {
        self.live.insert(id);
        if self.manager.get(id)?.active_step.is_none() && !self.manager.is_finished(id)? {
            self.manager.advance_with_policy(id, &self.policy).await?;
        }
        self.run_loop(Some(id)).await
    }

    /// Drive all live trajectories until every one is terminal (or max_ticks).
    pub async fn run_all_until_idle(&mut self) -> Result<()> {
        let ids: Vec<_> = self.live.iter().copied().collect();
        for id in ids {
            if self.manager.get(id)?.active_step.is_none() && !self.manager.is_finished(id)? {
                self.manager.advance_with_policy(id, &self.policy).await?;
            }
        }
        self.run_loop(None).await
    }

    async fn run_loop(&mut self, focus: Option<TrajectoryId>) -> Result<()> {
        let span = info_span!("driver.loop", ?focus);
        async {
            for tick_idx in 0..self.config.max_ticks {
                self.drain_tool_events().await?;
                self.maybe_unpin_under_pressure();

                if let Some(id) = focus {
                    if self.manager.is_finished(id)? {
                        self.cleanup_trajectory(id);
                        return Ok(());
                    }
                } else if self.all_live_finished()? {
                    return Ok(());
                }

                self.enqueue_all_ready()?;
                let tick = self.scheduler.tick();
                for decision in &tick.pin_decisions {
                    let _ = self.apply_pin_decision(decision);
                }
                debug!(
                    tick_idx,
                    actions = tick.actions.len(),
                    pins = tick.pin_decisions.len(),
                    ready = self.scheduler.ready_len(),
                    "scheduler tick"
                );

                if tick.actions.is_empty() {
                    // Prime idle trajectories that have no active step.
                    let ids: Vec<_> = self.live.iter().copied().collect();
                    let mut progressed = false;
                    for id in ids {
                        if self.manager.is_finished(id)? {
                            continue;
                        }
                        if self.manager.get(id)?.active_step.is_none() {
                            self.manager.advance_with_policy(id, &self.policy).await?;
                            progressed = true;
                        }
                    }
                    if !progressed && self.in_flight.is_empty() {
                        // Waiting on nothing — if focus still running, that's a stall.
                        if focus.is_some() {
                            tokio::task::yield_now().await;
                        } else {
                            tokio::task::yield_now().await;
                        }
                    } else {
                        tokio::task::yield_now().await;
                    }
                    continue;
                }

                for action in tick.actions {
                    self.execute_action(action).await?;
                }

                if let Some(id) = focus {
                    if self.manager.is_finished(id)? {
                        self.cleanup_trajectory(id);
                        return Ok(());
                    }
                }
            }

            Err(TrajectError::Other(format!(
                "driver exceeded max_ticks={} (focus={focus:?})",
                self.config.max_ticks
            )))
        }
        .instrument(span)
        .await
    }

    fn all_live_finished(&self) -> Result<bool> {
        for id in &self.live {
            if !self.manager.is_finished(*id)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn cleanup_trajectory(&mut self, id: TrajectoryId) {
        // Best-effort free of the leaf engine handle when the trajectory ends.
        if let Some(handle) = self.memory.engine_prefix_hint(id) {
            let session = self.memory.session_id(id).map(|s| s.to_string());
            self.spawn_engine_free(&handle, session);
        }
        self.memory.release_trajectory(id);
        self.live.remove(&id);
        self.external.remove(&id);
        self.cache_hits.remove(&id);
        self.tool_pin_started.remove(&id);
        self.in_flight.retain(|(t, _)| *t != id);
        self.decode_progress.retain(|(t, _), _| *t != id);
    }

    fn enqueue_all_ready(&mut self) -> Result<()> {
        let ids: Vec<_> = self.live.iter().copied().collect();
        for id in ids {
            self.enqueue_active(id)?;
        }
        Ok(())
    }

    fn enqueue_active(&mut self, id: TrajectoryId) -> Result<()> {
        if self.manager.is_finished(id)? {
            return Ok(());
        }
        let traj = self.manager.get(id)?;
        let Some(step) = traj.active_step.clone() else {
            return Ok(());
        };
        let key = (id, step.id());
        if self.in_flight.contains(&key) {
            return Ok(());
        }
        // Don't double-queue the same step while still sitting in ready.
        if self.scheduler.has_step(id, step.id()) {
            return Ok(());
        }

        let kind = match &step {
            Step::Generate { .. } => {
                let decoded = self.decode_progress.get(&key).copied().unwrap_or(0);
                if decoded > 0 {
                    SchedulableKind::ActiveDecode
                } else if traj.history.iter().any(|r| {
                    matches!(
                        r.outcome,
                        Some(StepOutcome::ToolDone { .. })
                    )
                }) && matches!(
                    traj.history.last().and_then(|r| r.outcome.as_ref()),
                    Some(StepOutcome::ToolDone { .. }) | None
                ) {
                    SchedulableKind::PostToolGenerate
                } else if traj.history.is_empty() {
                    if traj.priority.0 >= 10 {
                        SchedulableKind::HighPriorityNew
                    } else {
                        SchedulableKind::NormalNew
                    }
                } else {
                    SchedulableKind::NormalNew
                }
            }
            Step::Tool { .. } => SchedulableKind::Tool,
            Step::Control { .. } => SchedulableKind::NormalNew,
        };

        let decoded_so_far = self.decode_progress.get(&key).copied().unwrap_or(0);
        self.scheduler.push_ready(ReadyStep {
            trajectory_id: id,
            step,
            priority: SchedPriority::new(kind, traj.priority, traj.fairness_credit),
            decoded_so_far,
        });
        Ok(())
    }

    async fn execute_action(&mut self, action: TickAction) -> Result<()> {
        match action {
            TickAction::RunGenerateChunk {
                trajectory_id,
                step_id,
                chunk_tokens,
            } => {
                let key = (trajectory_id, step_id);
                let traj = self.manager.get(trajectory_id)?;
                let step = match traj.active_step.clone() {
                    Some(s) => s,
                    None => return Ok(()),
                };
                let Step::Generate {
                    delta,
                    constraints,
                    max_tokens,
                    id: active_id,
                    ..
                } = step
                else {
                    // Stale generate action after policy already advanced.
                    return Ok(());
                };
                if active_id != step_id {
                    return Ok(());
                }

                let decoded_so_far = self.decode_progress.get(&key).copied().unwrap_or(0);
                let now = now_ms();
                // Also append prompt tokens into the logical tree for radix sharing demos.
                if decoded_so_far == 0 {
                    if let Some(text) = &delta.text {
                        let toks: Vec<u32> = text.bytes().map(|b| b as u32).collect();
                        if !toks.is_empty() {
                            if let Ok(node) =
                                self.memory.append_tokens_at(trajectory_id, toks, now)
                            {
                                let _ =
                                    self.manager.with_mut(trajectory_id, |t| t.bind_prefix(node));
                            }
                        }
                    } else if !delta.token_ids.is_empty() {
                        if let Ok(node) = self.memory.append_tokens_at(
                            trajectory_id,
                            delta.token_ids.clone(),
                            now,
                        ) {
                            let _ = self.manager.with_mut(trajectory_id, |t| t.bind_prefix(node));
                        }
                    }
                }

                let traj = self.manager.get(trajectory_id)?;
                let session_id = self
                    .memory
                    .session_id(trajectory_id)
                    .map(|s| s.to_string())
                    .or_else(|| Some(trajectory_id.to_string()));
                let prefix_hint = self.memory.engine_prefix_hint(trajectory_id);
                let req = traject_inference::GenerateRequest {
                    trajectory_id,
                    step_id,
                    prefix: traj.current_prefix,
                    delta: delta.clone(),
                    constraints,
                    max_tokens,
                    session_id,
                    prefix_hint,
                };

                let chunk = self
                    .engine
                    .run_chunk(&req, chunk_tokens, decoded_so_far)
                    .instrument(info_span!("inference.chunk", %trajectory_id, %step_id))
                    .await?;

                let next = decoded_so_far.saturating_add(chunk.tokens_produced);
                self.decode_progress.insert(key, next);

                // Align MemoryManager with engine cache hits + produced tokens.
                if chunk.cache_hit_tokens > 0 {
                    let _ = self.memory.note_cache_hit(
                        trajectory_id,
                        chunk.cache_hit_tokens,
                        now,
                    );
                    let total = {
                        let e = self.cache_hits.entry(trajectory_id).or_insert(0);
                        *e = e.saturating_add(chunk.cache_hit_tokens);
                        *e
                    };
                    let last = chunk.cache_hit_tokens.to_string();
                    let total_s = total.to_string();
                    let _ = self.manager.with_mut(trajectory_id, |t| {
                        t.memory.set_slot("last_cache_hit_tokens", last);
                        t.memory.set_slot("total_cache_hit_tokens", total_s);
                    });
                }

                if !chunk.token_ids.is_empty() {
                    match self.memory.append_tokens_at(
                        trajectory_id,
                        chunk.token_ids.clone(),
                        now,
                    ) {
                        Ok(node) => {
                            let _ = self.manager.with_mut(trajectory_id, |t| t.bind_prefix(node));
                        }
                        Err(e) => warn!(%trajectory_id, error = %e, "prefix append failed"),
                    }
                }

                if chunk.finished {
                    self.decode_progress.remove(&key);
                    self.in_flight.remove(&key);
                    let finish_reason = chunk.finish_reason.unwrap_or(FinishReason::Stop);
                    let tool_call = chunk.tool_call.clone();
                    self.manager.complete_step(
                        trajectory_id,
                        StepOutcome::Generated {
                            step_id,
                            text: chunk.text,
                            token_ids: chunk.token_ids,
                            finish_reason,
                            tool_call: tool_call.clone(),
                        },
                    )?;
                    self.bump_fairness(trajectory_id, 1);

                    if tool_call.is_some() {
                        let now = now_ms();
                        self.tool_pin_started.insert(trajectory_id, now);
                        let tool_name = tool_call.as_ref().map(|c| c.name.as_str());
                        let p95 = self.tool_latency.p95_for(tool_name);
                        let traj = self.manager.get(trajectory_id)?;
                        let decision =
                            pin_for_tool_wait(&traj, now, self.scheduler.pin_policy(), p95);
                        self.apply_pin_decision(&decision)?;
                    }

                    if !self.external.contains(&trajectory_id) {
                        self.manager
                            .advance_with_policy(trajectory_id, &self.policy)
                            .await?;
                        if self.manager.is_finished(trajectory_id)? {
                            self.cleanup_trajectory(trajectory_id);
                        }
                    }
                } else {
                    // Still decoding — keep eligible as ActiveDecode via decode_progress.
                    self.in_flight.remove(&key);
                }
                Ok(())
            }
            TickAction::RunTool {
                trajectory_id,
                step_id,
            } => {
                let key = (trajectory_id, step_id);
                if !self.in_flight.insert(key) {
                    return Ok(());
                }
                let traj = self.manager.get(trajectory_id)?;
                let step = traj
                    .active_step
                    .clone()
                    .ok_or_else(|| TrajectError::StepNotFound(step_id))?;
                let Step::Tool {
                    call, timeout_ms, ..
                } = step
                else {
                    self.in_flight.remove(&key);
                    return Err(TrajectError::Other("expected tool step".into()));
                };
                debug!(%trajectory_id, %step_id, tool = %call.name, "spawning tool");
                self.tool_runtime.spawn(
                    trajectory_id,
                    step_id,
                    call,
                    Some(std::time::Duration::from_millis(timeout_ms)),
                );
                Ok(())
            }
            TickAction::RunControl {
                trajectory_id,
                step_id,
            } => {
                let traj = self.manager.get(trajectory_id)?;
                let step = traj
                    .active_step
                    .clone()
                    .ok_or_else(|| TrajectError::StepNotFound(step_id))?;
                let Step::Control { kind, payload, .. } = step else {
                    return Err(TrajectError::Other("expected control step".into()));
                };
                self.manager.complete_step(
                    trajectory_id,
                    StepOutcome::ControlDone {
                        step_id,
                        kind,
                        note: payload,
                    },
                )?;
                if !self.external.contains(&trajectory_id) {
                    self.manager
                        .advance_with_policy(trajectory_id, &self.policy)
                        .await?;
                }
                Ok(())
            }
        }
    }

    async fn drain_tool_events(&mut self) -> Result<()> {
        while let Ok(ev) = self.tool_rx.try_recv() {
            let key = (ev.trajectory_id, ev.step_id);
            self.in_flight.remove(&key);
            match ev.result {
                Ok(result) => {
                    let now = now_ms();
                    if let Some(start) = self.tool_pin_started.remove(&ev.trajectory_id) {
                        self.tool_latency
                            .record(&result.name, now.saturating_sub(start));
                    }
                    info!(
                        trajectory = %ev.trajectory_id,
                        tool = %result.name,
                        p95_ms = ?self.tool_latency.p95_for(Some(&result.name)),
                        "tool finished"
                    );
                    self.manager.complete_step(
                        ev.trajectory_id,
                        StepOutcome::ToolDone {
                            step_id: ev.step_id,
                            result,
                        },
                    )?;
                    self.manager
                        .with_mut(ev.trajectory_id, |t| t.pin.unpin())?;
                    if let Some(node) = self.memory.binding(ev.trajectory_id) {
                        let _ = self.memory.unpin_node(node);
                    }
                    // Soft prefetch pin for imminent post-tool Generate.
                    if let Ok(traj) = self.manager.get(ev.trajectory_id) {
                        let prefetch =
                            pin_for_prefetch(&traj, now, self.scheduler.pin_policy());
                        let _ = self.apply_pin_decision(&prefetch);
                    }
                    // Boost fairness so post-tool generate wins scheduling.
                    self.bump_fairness(ev.trajectory_id, 5);
                    if !self.external.contains(&ev.trajectory_id) {
                        self.manager
                            .advance_with_policy(ev.trajectory_id, &self.policy)
                            .await?;
                        if self.manager.is_finished(ev.trajectory_id)? {
                            self.cleanup_trajectory(ev.trajectory_id);
                        }
                    }
                }
                Err(e) => {
                    warn!(trajectory = %ev.trajectory_id, error = %e, "tool failed");
                    self.manager.fail(ev.trajectory_id, e.to_string())?;
                    self.cleanup_trajectory(ev.trajectory_id);
                }
            }
        }
        Ok(())
    }

    fn bump_fairness(&self, id: TrajectoryId, delta: i64) {
        let _ = self.manager.with_mut(id, |t| {
            t.fairness_credit = t.fairness_credit.saturating_add(delta);
        });
    }

    fn maybe_unpin_under_pressure(&mut self) {
        let stats = self.memory.stats();
        let high = stats.gpu_pressure >= self.config.eviction_pressure;
        let now = now_ms();
        if high {
            let ids: Vec<_> = self.live.iter().copied().collect();
            for id in ids {
                if let Ok(traj) = self.manager.get(id) {
                    if should_force_unpin(&traj.pin, now, true) {
                        let _ = self.manager.with_mut(id, |t| t.pin.unpin());
                        if let Some(node) = traj.current_prefix {
                            let _ = self.memory.unpin_node(node);
                        }
                    }
                }
            }
        }
        let (_n, handles) = self
            .memory
            .maybe_evict(now, self.config.eviction_pressure);
        for h in handles {
            self.spawn_engine_free(&h, None);
        }
    }

    fn spawn_engine_free(&self, prefix_id: &str, session_id: Option<String>) {
        let Some(client) = self.engine_prefix.clone() else {
            return;
        };
        let prefix_id = prefix_id.to_string();
        tokio::spawn(async move {
            client.free(&prefix_id, session_id.as_deref()).await;
        });
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Convenience: run a one-shot ReAct trajectory with the stub backend.
pub async fn run_simple_react(prompt: &str) -> Result<TrajectoryId> {
    let mut driver = Driver::new(DriverConfig::default());
    driver.policy = Arc::new(ReActPolicy::new(prompt));
    let id = driver.create_trajectory(TrajectoryConfig::default());
    driver.run_until_finished(id).await?;
    Ok(id)
}
