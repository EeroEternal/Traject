//! Logical prefix tree with radix sharing + physical paged blocks.

mod block_pool;
mod eviction;
mod manager;
mod prefix_tree;
mod tier;

pub use block_pool::{BlockId, BlockPool};
pub use eviction::{EvictionCandidate, EvictionPolicy};
pub use manager::{MemoryManager, MemoryStats};
pub use prefix_tree::{PrefixNode, PrefixTree};
pub use tier::{MemoryTier, TierId};
