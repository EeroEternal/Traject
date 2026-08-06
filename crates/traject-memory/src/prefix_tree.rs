use serde::{Deserialize, Serialize};
use traject_core::{PinInfo, PrefixNodeId, TrajectoryId};

use crate::{BlockId, TierId};

/// One node in the logical radix / prefix tree.
///
/// Physical KV lives in the engine (sglang-lite radix / V4 prefix cache).
/// `engine_handle` is the opaque key the engine uses for that materialization
/// so Traject can pin / reuse / evict in lockstep with MemoryManager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrefixNode {
    pub id: PrefixNodeId,
    pub parent: Option<PrefixNodeId>,
    pub children: Vec<PrefixNodeId>,
    /// Tokens represented by this edge from parent.
    pub tokens: Vec<u32>,
    pub ref_count: u32,
    pub share_score: f32,
    pub pin: PinInfo,
    pub owners: Vec<TrajectoryId>,
    pub tier: TierId,
    pub blocks: Vec<BlockId>,
    /// Engine-side radix / prefix-cache handle (session-stable key).
    pub engine_handle: Option<String>,
    /// Last touch time for LRU eviction under pressure.
    pub last_access_ms: u64,
    /// Cumulative cache-hit tokens reported by the engine for this node.
    pub cache_hit_tokens: u32,
}

impl PrefixNode {
    pub fn root() -> Self {
        Self {
            id: PrefixNodeId::new(),
            parent: None,
            children: Vec::new(),
            tokens: Vec::new(),
            ref_count: 0,
            share_score: 0.0,
            pin: PinInfo::default(),
            owners: Vec::new(),
            tier: TierId::Gpu,
            blocks: Vec::new(),
            engine_handle: None,
            last_access_ms: 0,
            cache_hit_tokens: 0,
        }
    }
}

/// In-memory prefix tree with radix sharing on insert.
#[derive(Debug, Default)]
pub struct PrefixTree {
    nodes: std::collections::HashMap<PrefixNodeId, PrefixNode>,
    root: Option<PrefixNodeId>,
}

impl PrefixTree {
    pub fn new() -> Self {
        let mut tree = Self::default();
        let root = PrefixNode::root();
        let id = root.id;
        tree.nodes.insert(id, root);
        tree.root = Some(id);
        tree
    }

    pub fn root_id(&self) -> PrefixNodeId {
        self.root.expect("prefix tree always has a root")
    }

    pub fn get(&self, id: PrefixNodeId) -> Option<&PrefixNode> {
        self.nodes.get(&id)
    }

    pub fn get_mut(&mut self, id: PrefixNodeId) -> Option<&mut PrefixNode> {
        self.nodes.get_mut(&id)
    }

    pub fn nodes(&self) -> impl Iterator<Item = &PrefixNode> {
        self.nodes.values()
    }

    /// Append without sharing (always new child). Prefer `insert_tokens`.
    pub fn append(
        &mut self,
        parent: PrefixNodeId,
        tokens: Vec<u32>,
        owner: TrajectoryId,
    ) -> Option<PrefixNodeId> {
        self.alloc_child(parent, tokens, owner)
    }

    /// Radix insert: reuse longest matching child edge, split on divergence.
    pub fn insert_tokens(
        &mut self,
        parent: PrefixNodeId,
        tokens: &[u32],
        owner: TrajectoryId,
    ) -> Option<PrefixNodeId> {
        if tokens.is_empty() {
            self.retain(parent);
            self.add_owner(parent, owner);
            return Some(parent);
        }

        let child_ids = self.nodes.get(&parent)?.children.clone();
        for cid in child_ids {
            let child_tokens = self.nodes.get(&cid)?.tokens.clone();
            let common = common_prefix_len(tokens, &child_tokens);
            if common == 0 {
                continue;
            }

            if common == child_tokens.len() {
                // Fully consume this edge; recurse with remainder.
                let rest = &tokens[common..];
                if rest.is_empty() {
                    self.retain(cid);
                    self.add_owner(cid, owner);
                    self.bump_share(cid);
                    return Some(cid);
                }
                return self.insert_tokens(cid, rest, owner);
            }

            if common == tokens.len() {
                // New path is a strict prefix of existing edge → split.
                let split = self.split_edge(cid, common)?;
                self.retain(split);
                self.add_owner(split, owner);
                self.bump_share(split);
                return Some(split);
            }

            // Partial overlap → split existing, then branch remainder.
            let split = self.split_edge(cid, common)?;
            let rest = tokens[common..].to_vec();
            let leaf = self.alloc_child(split, rest, owner)?;
            self.bump_share(split);
            return Some(leaf);
        }

        // No matching child — allocate fresh.
        self.alloc_child(parent, tokens.to_vec(), owner)
    }

    fn split_edge(&mut self, node_id: PrefixNodeId, at: usize) -> Option<PrefixNodeId> {
        let node = self.nodes.get(&node_id)?.clone();
        if at == 0 || at >= node.tokens.len() {
            return Some(node_id);
        }
        let parent_id = node.parent?;
        let head = node.tokens[..at].to_vec();
        let tail = node.tokens[at..].to_vec();

        let mid_id = PrefixNodeId::new();
        let mid = PrefixNode {
            id: mid_id,
            parent: Some(parent_id),
            children: vec![node_id],
            tokens: head,
            ref_count: node.ref_count,
            share_score: node.share_score,
            pin: PinInfo::default(),
            owners: node.owners.clone(),
            tier: node.tier,
            blocks: Vec::new(),
            engine_handle: None,
            last_access_ms: node.last_access_ms,
            cache_hit_tokens: 0,
        };

        // Rewire parent: replace node_id with mid_id.
        if let Some(p) = self.nodes.get_mut(&parent_id) {
            for c in &mut p.children {
                if *c == node_id {
                    *c = mid_id;
                }
            }
        }

        // Update original node to hold the tail.
        if let Some(n) = self.nodes.get_mut(&node_id) {
            n.parent = Some(mid_id);
            n.tokens = tail;
        }

        self.nodes.insert(mid_id, mid);
        Some(mid_id)
    }

    fn alloc_child(
        &mut self,
        parent: PrefixNodeId,
        tokens: Vec<u32>,
        owner: TrajectoryId,
    ) -> Option<PrefixNodeId> {
        let child_id = PrefixNodeId::new();
        {
            let parent_node = self.nodes.get_mut(&parent)?;
            parent_node.children.push(child_id);
        }
        let node = PrefixNode {
            id: child_id,
            parent: Some(parent),
            children: Vec::new(),
            tokens,
            ref_count: 1,
            share_score: 1.0,
            pin: PinInfo::default(),
            owners: vec![owner],
            tier: TierId::Gpu,
            blocks: Vec::new(),
            engine_handle: None,
            last_access_ms: 0,
            cache_hit_tokens: 0,
        };
        self.nodes.insert(child_id, node);
        Some(child_id)
    }

    pub fn touch(&mut self, id: PrefixNodeId, now_ms: u64) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.last_access_ms = now_ms;
        }
    }

    pub fn set_engine_handle(&mut self, id: PrefixNodeId, handle: impl Into<String>) -> bool {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.engine_handle = Some(handle.into());
            true
        } else {
            false
        }
    }

    pub fn engine_handle(&self, id: PrefixNodeId) -> Option<&str> {
        self.nodes
            .get(&id)
            .and_then(|n| n.engine_handle.as_deref())
    }

    pub fn add_cache_hits(&mut self, id: PrefixNodeId, hits: u32) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.cache_hit_tokens = n.cache_hit_tokens.saturating_add(hits);
            n.share_score += hits as f32 * 0.01;
        }
    }

    fn bump_share(&mut self, id: PrefixNodeId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.share_score += 1.0;
        }
    }

    pub fn retain(&mut self, id: PrefixNodeId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.ref_count = n.ref_count.saturating_add(1);
        }
    }

    pub fn release(&mut self, id: PrefixNodeId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.ref_count = n.ref_count.saturating_sub(1);
        }
    }

    pub fn add_owner(&mut self, id: PrefixNodeId, owner: TrajectoryId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            if !n.owners.contains(&owner) {
                n.owners.push(owner);
            }
        }
    }

    pub fn remove_owner(&mut self, id: PrefixNodeId, owner: TrajectoryId) {
        if let Some(n) = self.nodes.get_mut(&id) {
            n.owners.retain(|o| *o != owner);
        }
    }

    pub fn remove_node(&mut self, id: PrefixNodeId) -> bool {
        if Some(id) == self.root {
            return false;
        }
        let Some(node) = self.nodes.remove(&id) else {
            return false;
        };
        if let Some(parent) = node.parent {
            if let Some(p) = self.nodes.get_mut(&parent) {
                p.children.retain(|c| *c != id);
            }
        }
        // Orphan children are re-parented to grandparent if possible, else dropped links.
        for child in node.children {
            if let Some(c) = self.nodes.get_mut(&child) {
                c.parent = node.parent;
            }
            if let Some(parent) = node.parent {
                if let Some(p) = self.nodes.get_mut(&parent) {
                    if !p.children.contains(&child) {
                        p.children.push(child);
                    }
                }
            }
        }
        true
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_child() {
        let mut tree = PrefixTree::new();
        let root = tree.root_id();
        let child = tree
            .append(root, vec![1, 2, 3], TrajectoryId::new())
            .unwrap();
        assert_eq!(tree.len(), 2);
        assert_eq!(tree.get(child).unwrap().tokens, vec![1, 2, 3]);
    }

    #[test]
    fn radix_share_identical() {
        let mut tree = PrefixTree::new();
        let root = tree.root_id();
        let t1 = TrajectoryId::new();
        let t2 = TrajectoryId::new();
        let a = tree.insert_tokens(root, &[10, 20, 30], t1).unwrap();
        let b = tree.insert_tokens(root, &[10, 20, 30], t2).unwrap();
        assert_eq!(a, b);
        assert!(tree.get(a).unwrap().ref_count >= 2);
    }

    #[test]
    fn radix_split_on_divergence() {
        let mut tree = PrefixTree::new();
        let root = tree.root_id();
        let t1 = TrajectoryId::new();
        let t2 = TrajectoryId::new();
        let a = tree.insert_tokens(root, &[1, 2, 3, 4], t1).unwrap();
        let b = tree.insert_tokens(root, &[1, 2, 9, 9], t2).unwrap();
        assert_ne!(a, b);
        // Mid node for [1,2] should exist.
        let mid = tree.get(a).unwrap().parent.unwrap();
        assert_eq!(tree.get(mid).unwrap().tokens, vec![1, 2]);
        assert_eq!(tree.get(b).unwrap().parent, Some(mid));
    }
}
