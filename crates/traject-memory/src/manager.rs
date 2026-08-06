use std::collections::HashMap;

use tracing::{debug, trace};
use traject_core::{PinInfo, PrefixNodeId, Result, TrajectError, TrajectoryId};

use crate::block_pool::BlockPool;
use crate::eviction::EvictionPolicy;
use crate::prefix_tree::PrefixTree;
use crate::tier::{MemoryTier, TierId};

#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    pub nodes: usize,
    pub blocks_allocated: usize,
    pub blocks_capacity: usize,
    pub gpu_pressure: f32,
    pub total_cache_hit_tokens: u32,
    pub pinned_nodes: usize,
}

/// Per-trajectory session binding for engine radix alignment.
#[derive(Debug, Clone)]
pub struct SessionBinding {
    pub session_id: String,
    pub total_cache_hit_tokens: u32,
    pub generate_count: u32,
}

/// Unified hierarchical memory: logical prefix tree + block pool + eviction.
///
/// Engine physical KV (sglang-lite radix / V4 cache) is addressed via
/// `PrefixNode.engine_handle`. Traject owns pin / ref / eviction decisions;
/// the handle is what gets sent as `prefix_id` on Generate.
pub struct MemoryManager {
    pub tree: PrefixTree,
    pub pool: BlockPool,
    pub gpu: MemoryTier,
    pub eviction: EvictionPolicy,
    /// Trajectory → currently bound leaf prefix.
    bindings: HashMap<TrajectoryId, PrefixNodeId>,
    /// Trajectory → agent session (stable across Generate/Tool turns).
    sessions: HashMap<TrajectoryId, SessionBinding>,
}

impl MemoryManager {
    pub fn new(block_capacity: usize) -> Self {
        Self {
            tree: PrefixTree::new(),
            pool: BlockPool::new(block_capacity),
            gpu: MemoryTier::new(TierId::Gpu, block_capacity),
            eviction: EvictionPolicy::default(),
            bindings: HashMap::new(),
            sessions: HashMap::new(),
        }
    }

    pub fn root_id(&self) -> PrefixNodeId {
        self.tree.root_id()
    }

    pub fn stats(&self) -> MemoryStats {
        let now = 0u64;
        let pinned_nodes = self
            .tree
            .nodes()
            .filter(|n| n.pin.is_pinned(now) || n.pin.pin_until_ms.is_some())
            .count();
        let total_cache_hit_tokens = self
            .sessions
            .values()
            .map(|s| s.total_cache_hit_tokens)
            .sum();
        MemoryStats {
            nodes: self.tree.len(),
            blocks_allocated: self.pool.allocated(),
            blocks_capacity: self.pool.capacity(),
            gpu_pressure: self.gpu.pressure(),
            total_cache_hit_tokens,
            pinned_nodes,
        }
    }

    /// Register a stable session id for a trajectory (defaults to traj id string).
    pub fn bind_session(&mut self, traj: TrajectoryId, session_id: impl Into<String>) {
        let session_id = session_id.into();
        self.sessions
            .entry(traj)
            .and_modify(|s| s.session_id = session_id.clone())
            .or_insert(SessionBinding {
                session_id,
                total_cache_hit_tokens: 0,
                generate_count: 0,
            });
    }

    pub fn session_id(&self, traj: TrajectoryId) -> Option<&str> {
        self.sessions.get(&traj).map(|s| s.session_id.as_str())
    }

    pub fn session_binding(&self, traj: TrajectoryId) -> Option<&SessionBinding> {
        self.sessions.get(&traj)
    }

    pub fn bind_trajectory(&mut self, traj: TrajectoryId, node: PrefixNodeId) -> Result<()> {
        if self.tree.get(node).is_none() {
            return Err(TrajectError::PrefixNotFound(node));
        }
        if let Some(old) = self.bindings.insert(traj, node) {
            if old != node {
                self.tree.release(old);
            }
        }
        self.tree.retain(node);
        self.tree.add_owner(node, traj);
        if !self.sessions.contains_key(&traj) {
            self.bind_session(traj, traj.to_string());
        }
        Ok(())
    }

    pub fn binding(&self, traj: TrajectoryId) -> Option<PrefixNodeId> {
        self.bindings.get(&traj).copied()
    }

    /// Opaque engine prefix key for the trajectory's current leaf (or session).
    pub fn engine_prefix_hint(&self, traj: TrajectoryId) -> Option<String> {
        let node = self.binding(traj)?;
        if let Some(h) = self.tree.engine_handle(node) {
            return Some(h.to_string());
        }
        // Fall back to logical node id so the engine can log / key by it.
        Some(node.to_string())
    }

    /// Bind an engine-side handle onto the current prefix leaf.
    pub fn set_engine_handle(
        &mut self,
        traj: TrajectoryId,
        handle: impl Into<String>,
    ) -> Result<()> {
        let node = self
            .binding(traj)
            .ok_or_else(|| TrajectError::Other("trajectory has no prefix binding".into()))?;
        self.tree.set_engine_handle(node, handle);
        Ok(())
    }

    /// Record engine cache-hit tokens against the current leaf + session totals.
    pub fn note_cache_hit(
        &mut self,
        traj: TrajectoryId,
        hits: u32,
        now_ms: u64,
    ) -> Result<()> {
        if hits == 0 {
            return Ok(());
        }
        if let Some(node) = self.binding(traj) {
            self.tree.add_cache_hits(node, hits);
            self.tree.touch(node, now_ms);
        }
        if let Some(s) = self.sessions.get_mut(&traj) {
            s.total_cache_hit_tokens = s.total_cache_hit_tokens.saturating_add(hits);
            s.generate_count = s.generate_count.saturating_add(1);
        }
        debug!(%traj, hits, "recorded engine cache hit tokens");
        Ok(())
    }

    pub fn total_cache_hits(&self, traj: TrajectoryId) -> u32 {
        self.sessions
            .get(&traj)
            .map(|s| s.total_cache_hit_tokens)
            .unwrap_or(0)
    }

    /// Insert tokens under the trajectory's current prefix (radix-shared).
    pub fn append_tokens(
        &mut self,
        traj: TrajectoryId,
        tokens: Vec<u32>,
    ) -> Result<PrefixNodeId> {
        self.append_tokens_at(traj, tokens, 0)
    }

    pub fn append_tokens_at(
        &mut self,
        traj: TrajectoryId,
        tokens: Vec<u32>,
        now_ms: u64,
    ) -> Result<PrefixNodeId> {
        if tokens.is_empty() {
            return self
                .binding(traj)
                .ok_or_else(|| TrajectError::Other("trajectory has no prefix binding".into()));
        }
        let parent = self
            .binding(traj)
            .unwrap_or_else(|| self.tree.root_id());

        self.ensure_capacity(1, now_ms)?;

        let child = self
            .tree
            .insert_tokens(parent, &tokens, traj)
            .ok_or_else(|| TrajectError::PrefixNotFound(parent))?;

        // Inherit / set engine handle: keep parent handle when fully shared, else leaf id.
        if self.tree.engine_handle(child).is_none() {
            let handle = self
                .tree
                .engine_handle(parent)
                .map(|s| s.to_string())
                .unwrap_or_else(|| child.to_string());
            self.tree.set_engine_handle(child, handle);
        }
        self.tree.touch(child, now_ms);

        if let Some(node) = self.tree.get_mut(child) {
            if node.blocks.is_empty() {
                if let Some(bid) = self.pool.alloc() {
                    node.blocks.push(bid);
                    self.gpu.used_blocks = self.pool.allocated();
                }
            }
        }

        self.bind_trajectory(traj, child)?;
        debug!(%traj, %child, n_tokens = tokens.len(), "appended tokens to prefix");
        Ok(child)
    }

    pub fn pin_node(&mut self, node: PrefixNodeId, pin: PinInfo) -> Result<()> {
        let n = self
            .tree
            .get_mut(node)
            .ok_or(TrajectError::PrefixNotFound(node))?;
        n.pin = pin;
        Ok(())
    }

    /// Pin the trajectory's current prefix for a tool gap.
    pub fn pin_trajectory(
        &mut self,
        traj: TrajectoryId,
        pin: PinInfo,
    ) -> Result<PrefixNodeId> {
        let node = self
            .binding(traj)
            .ok_or_else(|| TrajectError::Other("trajectory has no prefix binding".into()))?;
        self.pin_node(node, pin)?;
        debug!(%traj, %node, "pinned prefix for tool wait");
        Ok(node)
    }

    pub fn unpin_node(&mut self, node: PrefixNodeId) -> Result<()> {
        let n = self
            .tree
            .get_mut(node)
            .ok_or(TrajectError::PrefixNotFound(node))?;
        n.pin.unpin();
        Ok(())
    }

    pub fn unpin_trajectory(&mut self, traj: TrajectoryId) -> Result<()> {
        if let Some(node) = self.binding(traj) {
            self.unpin_node(node)?;
        }
        Ok(())
    }

    pub fn release_trajectory(&mut self, traj: TrajectoryId) {
        if let Some(node) = self.bindings.remove(&traj) {
            self.tree.remove_owner(node, traj);
            self.tree.release(node);
            trace!(%traj, %node, "released trajectory binding");
        }
        self.sessions.remove(&traj);
    }

    /// Evict unreferenced, unpinned nodes under pressure.
    /// Returns `(node_count, engine_handles_to_free)`.
    pub fn maybe_evict(
        &mut self,
        now_ms: u64,
        pressure_threshold: f32,
    ) -> (usize, Vec<String>) {
        if self.gpu.pressure() < pressure_threshold {
            return (0, Vec::new());
        }
        let mut freed = 0;
        let mut handles = Vec::new();
        while self.gpu.pressure() >= pressure_threshold {
            let candidates: Vec<_> = self.tree.nodes().cloned().collect();
            let Some(pick) = self.eviction.pick(candidates.iter(), now_ms) else {
                break;
            };
            match self.free_node(pick.node_id, now_ms) {
                Some(h) => {
                    freed += 1;
                    if let Some(handle) = h {
                        handles.push(handle);
                    }
                }
                None => break,
            }
        }
        if freed > 0 {
            debug!(
                freed,
                engine_frees = handles.len(),
                pressure = self.gpu.pressure(),
                "evicted prefix nodes"
            );
        }
        (freed, handles)
    }

    fn ensure_capacity(&mut self, need: usize, now_ms: u64) -> Result<()> {
        if self.pool.allocated() + need <= self.pool.capacity() {
            return Ok(());
        }
        let _ = self.maybe_evict(now_ms, 0.85);
        if self.pool.allocated() + need <= self.pool.capacity() {
            Ok(())
        } else {
            Err(TrajectError::MemoryPressure(format!(
                "need {need} blocks, allocated {}/{}",
                self.pool.allocated(),
                self.pool.capacity()
            )))
        }
    }

    /// Free a node. Returns `Some(engine_handle)` if freed, `None` if not eligible.
    ///
    /// Eligible when unpinned and either ref_count==0 or no remaining owners
    /// (refcounts can lag after trajectory release).
    fn free_node(&mut self, id: PrefixNodeId, now_ms: u64) -> Option<Option<String>> {
        if id == self.tree.root_id() {
            return None;
        }
        let node = self.tree.get(id).cloned()?;
        if node.pin.is_pinned(now_ms) {
            return None;
        }
        if node.ref_count > 0 && !node.owners.is_empty() {
            return None;
        }
        for b in &node.blocks {
            self.pool.free(*b);
        }
        self.gpu.used_blocks = self.pool.allocated();
        let handle = node.engine_handle.clone();
        if let Some(ref h) = handle {
            debug!(%id, handle = %h, "evicted engine-aligned prefix node");
        }
        self.tree.remove_node(id);
        Some(handle)
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use traject_core::{PinReason};

    #[test]
    fn shared_prefix_reuses_node() {
        let mut mem = MemoryManager::new(128);
        let t1 = TrajectoryId::new();
        let t2 = TrajectoryId::new();
        mem.bind_trajectory(t1, mem.root_id()).unwrap();
        mem.bind_trajectory(t2, mem.root_id()).unwrap();
        let a = mem.append_tokens(t1, vec![1, 2, 3]).unwrap();
        mem.bind_trajectory(t2, mem.root_id()).unwrap();
        let b = mem.append_tokens(t2, vec![1, 2, 3]).unwrap();
        assert_eq!(a, b);
        assert!(mem.tree.get(a).unwrap().ref_count >= 2);
    }

    #[test]
    fn engine_handle_and_cache_hit_track() {
        let mut mem = MemoryManager::new(128);
        let t = TrajectoryId::new();
        mem.bind_trajectory(t, mem.root_id()).unwrap();
        mem.bind_session(t, "sess-1");
        let node = mem.append_tokens_at(t, vec![1, 2], 1000).unwrap();
        assert!(mem.engine_prefix_hint(t).is_some());
        mem.set_engine_handle(t, "engine-prefix-abc").unwrap();
        assert_eq!(mem.tree.engine_handle(node), Some("engine-prefix-abc"));
        mem.note_cache_hit(t, 42, 2000).unwrap();
        assert_eq!(mem.total_cache_hits(t), 42);
        assert_eq!(mem.tree.get(node).unwrap().cache_hit_tokens, 42);
    }

    #[test]
    fn pin_protects_from_eviction() {
        let mut mem = MemoryManager::new(2);
        let t = TrajectoryId::new();
        mem.bind_trajectory(t, mem.root_id()).unwrap();
        let n = mem.append_tokens(t, vec![1]).unwrap();
        // Force high pressure by filling pool.
        let _ = mem.append_tokens(t, vec![2]);
        mem.pin_node(
            n,
            PinInfo::pin_until(u64::MAX, PinReason::WaitingTool, 1),
        )
        .unwrap();
        // Release binding so ref_count can drop, but pin should block free.
        mem.release_trajectory(t);
        let (freed, handles) = mem.maybe_evict(0, 0.0);
        // Root + pinned path may still free other nodes; pinned node must remain.
        assert!(
            mem.tree.get(n).is_some(),
            "pinned node must survive, freed={freed} handles={handles:?}"
        );
    }

    #[test]
    fn eviction_returns_engine_handles() {
        let mut mem = MemoryManager::new(1);
        let t = TrajectoryId::new();
        mem.bind_trajectory(t, mem.root_id()).unwrap();
        let n = mem.append_tokens(t, vec![1, 2, 3]).unwrap();
        mem.set_engine_handle(t, "eng-h-1").unwrap();
        assert_eq!(mem.tree.engine_handle(n), Some("eng-h-1"));
        mem.release_trajectory(t);
        // Force pressure by lowering capacity accounting: pool already full.
        let (freed, handles) = mem.maybe_evict(0, 0.0);
        assert!(freed >= 1, "expected free, got {freed}");
        assert!(
            handles.iter().any(|h| h == "eng-h-1"),
            "handles={handles:?}"
        );
    }
}
