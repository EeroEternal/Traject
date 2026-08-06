//! Weight loading for in-process runners (safetensors / HF sharded checkpoints).

mod catalog;
mod dtype;
mod hf_config;

pub use catalog::{load_embed_head_norm, SafetensorCatalog, TensorF32};
pub use hf_config::HfModelConfig;
