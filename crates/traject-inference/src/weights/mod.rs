//! Weight loading for in-process runners (safetensors / HF sharded checkpoints).

mod catalog;
mod dtype;
mod hf_config;

pub use catalog::{
    load_embed_head_norm, load_layer0_attn, Layer0AttnWeights, SafetensorCatalog, TensorF32,
};
pub use hf_config::HfModelConfig;
