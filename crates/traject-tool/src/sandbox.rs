use serde::{Deserialize, Serialize};

/// Placeholder sandbox policy (Phase 0: allow-all).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxPolicy {
    pub allow_network: bool,
    pub allow_fs: bool,
    pub max_cpu_ms: u64,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            allow_network: false,
            allow_fs: false,
            max_cpu_ms: 5_000,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sandbox {
    pub policy: SandboxPolicy,
}

impl Sandbox {
    pub fn check_allowed(&self, tool_name: &str) -> bool {
        let _ = tool_name;
        true
    }
}
