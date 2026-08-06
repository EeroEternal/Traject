//! Minimal HF `config.json` subset used by the local runner.

use std::fs;
use std::path::Path;

use serde::Deserialize;
use traject_core::{Result, TrajectError};

#[derive(Debug, Clone, Deserialize)]
pub struct HfModelConfig {
    #[serde(default = "default_vocab")]
    pub vocab_size: u32,
    #[serde(default = "default_hidden")]
    pub hidden_size: u32,
    #[serde(default)]
    pub num_attention_heads: Option<u32>,
    #[serde(default)]
    pub num_key_value_heads: Option<u32>,
    #[serde(default)]
    pub head_dim: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
}

fn default_vocab() -> u32 {
    32000
}
fn default_hidden() -> u32 {
    4096
}

impl HfModelConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let raw = fs::read_to_string(&path).map_err(|e| {
            TrajectError::Other(format!("read {}: {e}", path.display()))
        })?;
        serde_json::from_str(&raw).map_err(|e| {
            TrajectError::Other(format!("parse {}: {e}", path.display()))
        })
    }
}
