use serde::{Deserialize, Serialize};
use traject_core::PrefixNodeId;

use crate::PrefixNode;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionCandidate {
    pub node_id: PrefixNodeId,
    pub score: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvictionPolicy {
    /// Weight on share_score (higher share → harder to evict).
    pub share_weight: f32,
    /// Weight on ref_count.
    pub ref_weight: f32,
}

impl Default for EvictionPolicy {
    fn default() -> Self {
        Self {
            share_weight: 2.0,
            ref_weight: 1.0,
        }
    }
}

impl EvictionPolicy {
    /// Lower score = better eviction candidate. Pinned nodes are skipped.
    pub fn score(&self, node: &PrefixNode, now_ms: u64) -> Option<f32> {
        if node.pin.is_pinned(now_ms) {
            return None;
        }
        if node.ref_count > 0 {
            return None;
        }
        Some(self.share_weight * node.share_score + self.ref_weight * node.ref_count as f32)
    }

    pub fn pick<'a>(
        &self,
        nodes: impl Iterator<Item = &'a PrefixNode>,
        now_ms: u64,
    ) -> Option<EvictionCandidate> {
        nodes
            .filter_map(|n| {
                self.score(n, now_ms).map(|score| EvictionCandidate {
                    node_id: n.id,
                    score,
                })
            })
            .min_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
    }
}
