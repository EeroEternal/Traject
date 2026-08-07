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
    pub num_hidden_layers: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub model_type: Option<String>,
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    /// DeepSeek-V4 o_proj grouping (e.g. 8).
    #[serde(default)]
    pub o_groups: Option<u32>,
    /// DeepSeek-V4 o_proj LoRA rank per group (e.g. 1024).
    #[serde(default)]
    pub o_lora_rank: Option<u32>,
    /// MoE top-k (e.g. 6).
    #[serde(default)]
    pub num_experts_per_tok: Option<u32>,
    /// Multiplier on routed expert sum (e.g. 1.5).
    #[serde(default)]
    pub routed_scaling_factor: Option<f32>,
    /// RoPE dims on the last slice of each head (V4 Flash: 64).
    #[serde(default)]
    pub qk_rope_head_dim: Option<u32>,
    /// Base RoPE theta (default 10000).
    #[serde(default)]
    pub rope_theta: Option<f32>,
    /// Hyper-Connection stream count (V4 Flash: 4).
    #[serde(default)]
    pub hc_mult: Option<u32>,
    /// Sinkhorn iterations for HC comb matrix (default 20).
    #[serde(default)]
    pub hc_sinkhorn_iters: Option<u32>,
    /// HC epsilon (default 1e-6).
    #[serde(default)]
    pub hc_eps: Option<f32>,
    /// Sliding-window size for pure SWA layers (V4 Flash: 128).
    #[serde(default, alias = "window_size")]
    pub sliding_window: Option<u32>,
    /// Per-layer KV compress ratios (0 = pure SWA; >0 = compress + YaRN rope).
    #[serde(default)]
    pub compress_ratios: Option<Vec<u32>>,
    /// RoPE base for compressed layers (V4 Flash: 160000).
    #[serde(default)]
    pub compress_rope_theta: Option<f32>,
    /// YaRN / rope scaling block from config.json.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Indexer head count (V4 Flash: 64).
    #[serde(default)]
    pub index_n_heads: Option<u32>,
    /// Indexer per-head dim (V4 Flash: 128).
    #[serde(default)]
    pub index_head_dim: Option<u32>,
    /// Indexer top-k over compress pool (V4 Flash: 512).
    #[serde(default)]
    pub index_topk: Option<u32>,
}

/// Subset of HF `rope_scaling` used for YaRN frequency interpolation.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<u32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
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
