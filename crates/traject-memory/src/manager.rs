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
}

/// Unified hierarchical memory: logical prefix tree + block pool + eviction.
pub struct MemoryManager {
    pub tree: PrefixTree,
    pub pool: BlockPool,
    pub gpu: MemoryTier,
    pub eviction: EvictionPolicy,
    /// Trajectory → currently bound leaf prefix.
    bindings: HashMap<TrajectoryId, PrefixNodeId>,
}

impl MemoryManager {
    pub fn new(block_capacity: usize) -> Self {
        Self {
            tree: PrefixTree::new(),
            pool: BlockPool::new(block_capacity),
            gpu: MemoryTier::new(TierId::Gpu, block_capacity),
            eviction: EvictionPolicy::default(),
            bindings: HashMap::new(),
        }
    }

    pub fn root_id(&self) -> PrefixNodeId {
        self.tree.root_id()
    }

    pub fn stats(&self) -> MemoryStats {
        MemoryStats {
            nodes: self.tree.len(),
            blocks_allocated: self.pool.allocated(),
            blocks_capacity: self.pool.capacity(),
            gpu_pressure: self.gpu.pressure(),
        }
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
        Ok(())
    }

    pub fn binding(&self, traj: TrajectoryId) -> Option<PrefixNodeId> {
        self.bindings.get(&traj).copied()
    }

    /// Insert tokens under the trajectory's current prefix (radix-shared).
    pub fn append_tokens(
        &mut self,
        traj: TrajectoryId,
        tokens: Vec<u32>,
    ) -> Result<PrefixNodeId> {
        if tokens.is_empty() {
            return self
                .binding(traj)
                .ok_or_else(|| TrajectError::Other("trajectory has no prefix binding".into()));
        }
        let parent = self
            .binding(traj)
            .unwrap_or_else(|| self.tree.root_id());

        // Ensure capacity for a crude 1-block-per-node accounting.
        self.ensure_capacity(1)?;

        let child = self
            .tree
            .insert_tokens(parent, &tokens, traj)
            .ok_or_else(|| TrajectError::PrefixNotFound(parent))?;

        // Allocate a block if this is a newly created physical edge with no blocks yet.
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

    pub fn unpin_node(&mut self, node: PrefixNodeId) -> Result<()> {
        let n = self
            .tree
            .get_mut(node)
            .ok_or(TrajectError::PrefixNotFound(node))?;
        n.pin.unpin();
        Ok(())
    }

    pub fn release_trajectory(&mut self, traj: TrajectoryId) {
        if let Some(node) = self.bindings.remove(&traj) {
            self.tree.remove_owner(node, traj);
            self.tree.release(node);
            trace!(%traj, %node, "released trajectory binding");
        }
    }

    /// Evict unreferenced, unpinned nodes under pressure. Returns freed node count.
    pub fn maybe_evict(&mut self, now_ms: u64, pressure_threshold: f32) -> usize {
        if self.gpu.pressure() < pressure_threshold {
            return 0;
        }
        let mut freed = 0;
        while self.gpu.pressure() >= pressure_threshold {
            let candidates: Vec<_> = self.tree.nodes().cloned().collect();
            let Some(pick) = self.eviction.pick(candidates.iter(), now_ms) else {
                break;
            };
            if !self.free_node(pick.node_id, now_ms) {
                break;
            }
            freed += 1;
        }
        if freed > 0 {
            debug!(freed, pressure = self.gpu.pressure(), "evicted prefix nodes");
        }
        freed
    }

    fn ensure_capacity(&mut self, need: usize) -> Result<()> {
        if self.pool.allocated() + need <= self.pool.capacity() {
            return Ok(());
        }
        let now = 0; // caller may pass real time later via maybe_evict beforehand
        self.maybe_evict(now, 0.85);
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

    fn free_node(&mut self, id: PrefixNodeId, now_ms: u64) -> bool {
        if id == self.tree.root_id() {
            return false;
        }
        let Some(node) = self.tree.get(id).cloned() else {
            return false;
        };
        if node.ref_count > 0 || node.pin.is_pinned(now_ms) {
            return false;
        }
        for b in &node.blocks {
            self.pool.free(*b);
        }
        self.gpu.used_blocks = self.pool.allocated();
        self.tree.remove_node(id);
        true
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

    #[test]
    fn shared_prefix_reuses_node() {
        let mut mem = MemoryManager::new(128);
        let t1 = TrajectoryId::new();
        let t2 = TrajectoryId::new();
        mem.bind_trajectory(t1, mem.root_id()).unwrap();
        mem.bind_trajectory(t2, mem.root_id()).unwrap();
        let a = mem.append_tokens(t1, vec![1, 2, 3]).unwrap();
        // Reset t2 to root and append same tokens → should share.
        mem.bind_trajectory(t2, mem.root_id()).unwrap();
        let b = mem.append_tokens(t2, vec![1, 2, 3]).unwrap();
        assert_eq!(a, b);
        assert!(mem.tree.get(a).unwrap().ref_count >= 2);
    }
}
