use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque handle into the logical prefix tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixNodeId(pub Uuid);

impl PrefixNodeId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for PrefixNodeId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for PrefixNodeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Short-term structured state attached to a Trajectory.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMemory {
    pub scratchpad: Scratchpad,
    /// Opaque key/value bag for policy-specific state.
    pub slots: Vec<(String, String)>,
}

impl AgentMemory {
    pub fn set_slot(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        if let Some((_, v)) = self.slots.iter_mut().find(|(k, _)| *k == key) {
            *v = value.into();
        } else {
            self.slots.push((key, value.into()));
        }
    }

    pub fn get_slot(&self, key: &str) -> Option<&str> {
        self.slots
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

/// Free-form working notes the policy / agent can mutate between steps.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Scratchpad {
    pub notes: Vec<String>,
}

impl Scratchpad {
    pub fn push(&mut self, note: impl Into<String>) {
        self.notes.push(note.into());
    }
}

/// Pin lifecycle for a prefix (or trajectory-bound KV).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PinInfo {
    /// Absolute deadline; `None` means unpinned.
    pub pin_until_ms: Option<u64>,
    /// Why this pin exists (tool wait, prefetch, manual, …).
    pub reason: PinReason,
    /// Soft priority among pins when memory pressure forces unpin.
    pub strength: u8,
}

impl Default for PinInfo {
    fn default() -> Self {
        Self {
            pin_until_ms: None,
            reason: PinReason::None,
            strength: 0,
        }
    }
}

impl PinInfo {
    pub fn is_pinned(&self, now_ms: u64) -> bool {
        self.pin_until_ms
            .map(|until| until > now_ms)
            .unwrap_or(false)
    }

    pub fn pin_until(until_ms: u64, reason: PinReason, strength: u8) -> Self {
        Self {
            pin_until_ms: Some(until_ms),
            reason,
            strength,
        }
    }

    pub fn unpin(&mut self) {
        self.pin_until_ms = None;
        self.reason = PinReason::None;
        self.strength = 0;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PinReason {
    #[default]
    None,
    WaitingTool,
    Prefetch,
    Manual,
    Affinity,
}
