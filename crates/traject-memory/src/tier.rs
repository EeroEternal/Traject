use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TierId {
    Gpu,
    Cpu,
    Remote,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTier {
    pub id: TierId,
    pub capacity_blocks: usize,
    pub used_blocks: usize,
}

impl MemoryTier {
    pub fn new(id: TierId, capacity_blocks: usize) -> Self {
        Self {
            id,
            capacity_blocks,
            used_blocks: 0,
        }
    }

    pub fn pressure(&self) -> f32 {
        if self.capacity_blocks == 0 {
            return 1.0;
        }
        self.used_blocks as f32 / self.capacity_blocks as f32
    }
}
