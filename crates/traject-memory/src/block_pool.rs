use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BlockId(pub u64);

/// Simple fixed-size block pool (Phase 0 stub — no real device memory).
#[derive(Debug)]
pub struct BlockPool {
    next_id: u64,
    free: Vec<BlockId>,
    capacity: usize,
    allocated: usize,
}

impl BlockPool {
    pub fn new(capacity: usize) -> Self {
        Self {
            next_id: 1,
            free: Vec::new(),
            capacity,
            allocated: 0,
        }
    }

    pub fn alloc(&mut self) -> Option<BlockId> {
        if let Some(id) = self.free.pop() {
            self.allocated += 1;
            return Some(id);
        }
        if self.allocated >= self.capacity {
            return None;
        }
        let id = BlockId(self.next_id);
        self.next_id += 1;
        self.allocated += 1;
        Some(id)
    }

    pub fn free(&mut self, id: BlockId) {
        self.free.push(id);
        self.allocated = self.allocated.saturating_sub(1);
    }

    pub fn allocated(&self) -> usize {
        self.allocated
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}
