//! Weight loading for in-process runners (safetensors / HF sharded checkpoints).

mod catalog;
mod dtype;
mod hf_config;

pub use catalog::{
    load_embed_head_norm, load_hc_head, load_layer0_attn, load_layer0_routed_moe,
    load_layer0_shared_ffn, load_layer_attn, load_layer_hc, load_layer_routed_moe,
    load_layer_shared_ffn, load_layer_stack, ExpertF32, ExpertPacked, HcBranchWeights,
    HcHeadWeights, Layer0AttnWeights, Layer0RoutedMoe, Layer0SharedFfn, LayerBlock,
    CompressorWeights, LayerHcWeights, LinearMat, PackedFp4Mat, PackedFp8Mat, RopeParams,
    SafetensorCatalog, TensorF32,
};
pub use hf_config::{HfModelConfig, RopeScalingConfig};
