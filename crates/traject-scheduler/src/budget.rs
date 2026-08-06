use serde::{Deserialize, Serialize};

/// Dual budget that bounds one scheduler tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerBudget {
    pub tokens: TokenBudget,
    pub tools: ToolConcurrencyBudget,
}

impl Default for SchedulerBudget {
    fn default() -> Self {
        Self {
            tokens: TokenBudget::new(2048),
            tools: ToolConcurrencyBudget::new(64),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBudget {
    pub capacity: u32,
    pub remaining: u32,
}

impl TokenBudget {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            remaining: capacity,
        }
    }

    pub fn try_consume(&mut self, n: u32) -> bool {
        if self.remaining >= n {
            self.remaining -= n;
            true
        } else {
            false
        }
    }

    pub fn refill(&mut self) {
        self.remaining = self.capacity;
    }

    /// Runtime tweak from decode count / memory / latency feedback.
    pub fn adjust_capacity(&mut self, new_capacity: u32) {
        let used = self.capacity.saturating_sub(self.remaining);
        self.capacity = new_capacity;
        self.remaining = new_capacity.saturating_sub(used);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConcurrencyBudget {
    pub capacity: u32,
    pub in_flight: u32,
}

impl ToolConcurrencyBudget {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            in_flight: 0,
        }
    }

    pub fn try_acquire(&mut self) -> bool {
        if self.in_flight < self.capacity {
            self.in_flight += 1;
            true
        } else {
            false
        }
    }

    pub fn release(&mut self) {
        self.in_flight = self.in_flight.saturating_sub(1);
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub token_remaining: u32,
    pub token_capacity: u32,
    pub tools_in_flight: u32,
    pub tools_capacity: u32,
}

impl From<&SchedulerBudget> for BudgetSnapshot {
    fn from(b: &SchedulerBudget) -> Self {
        Self {
            token_remaining: b.tokens.remaining,
            token_capacity: b.tokens.capacity,
            tools_in_flight: b.tools.in_flight,
            tools_capacity: b.tools.capacity,
        }
    }
}
