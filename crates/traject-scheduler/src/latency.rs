//! Rolling tool-latency histogram for pin TTL estimation.

use std::collections::HashMap;

/// Keeps a fixed-size ring of recent samples (ms) and answers p95.
#[derive(Debug, Clone)]
pub struct LatencyHistogram {
    samples: Vec<u64>,
    cap: usize,
    next: usize,
    count: usize,
}

impl LatencyHistogram {
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(8);
        Self {
            samples: vec![0; cap],
            cap,
            next: 0,
            count: 0,
        }
    }

    pub fn record(&mut self, ms: u64) {
        self.samples[self.next] = ms;
        self.next = (self.next + 1) % self.cap;
        self.count = (self.count + 1).min(self.cap);
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Approximate percentile in \[0, 100\]. Returns None if no samples.
    pub fn percentile(&self, pct: u8) -> Option<u64> {
        if self.count == 0 {
            return None;
        }
        let mut v: Vec<u64> = if self.count < self.cap {
            self.samples[..self.count].to_vec()
        } else {
            self.samples.clone()
        };
        v.sort_unstable();
        let pct = pct.min(100) as usize;
        let idx = ((pct * (v.len().saturating_sub(1))) / 100).min(v.len() - 1);
        Some(v[idx])
    }

    pub fn p95(&self) -> Option<u64> {
        self.percentile(95)
    }

    pub fn p50(&self) -> Option<u64> {
        self.percentile(50)
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new(64)
    }
}

/// Per-tool + global latency tracker.
#[derive(Debug, Default)]
pub struct ToolLatencyTracker {
    by_tool: HashMap<String, LatencyHistogram>,
    global: LatencyHistogram,
}

impl ToolLatencyTracker {
    pub fn new() -> Self {
        Self {
            by_tool: HashMap::new(),
            global: LatencyHistogram::default(),
        }
    }

    pub fn record(&mut self, tool: &str, ms: u64) {
        self.global.record(ms);
        self.by_tool
            .entry(tool.to_string())
            .or_default()
            .record(ms);
    }

    /// Prefer tool-specific p95, else global p95.
    pub fn p95_for(&self, tool: Option<&str>) -> Option<u64> {
        if let Some(name) = tool {
            if let Some(h) = self.by_tool.get(name) {
                if let Some(p) = h.p95() {
                    return Some(p);
                }
            }
        }
        self.global.p95()
    }

    pub fn global_p95(&self) -> Option<u64> {
        self.global.p95()
    }

    pub fn sample_count(&self) -> usize {
        self.global.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_from_samples() {
        let mut h = LatencyHistogram::new(32);
        for i in 1..=20 {
            h.record(i * 10);
        }
        let p95 = h.p95().unwrap();
        assert!(p95 >= 180, "p95={p95}");
        assert!(p95 <= 200, "p95={p95}");
    }

    #[test]
    fn tracker_prefers_tool() {
        let mut t = ToolLatencyTracker::new();
        for _ in 0..20 {
            t.record("Glob", 100);
            t.record("Bash", 5_000);
        }
        assert!(t.p95_for(Some("Glob")).unwrap() < 500);
        assert!(t.p95_for(Some("Bash")).unwrap() > 4_000);
    }
}
