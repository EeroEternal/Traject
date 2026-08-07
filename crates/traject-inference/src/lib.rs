//! Inference core — forward, sample, constrained decode.
//!
//! Backends: stub, OpenAI HTTP, sglang-lite engine, in-process
//! [`LocalWeightRunner`] (physical paged KV + toy/weights), KernelBackend
//! smoke (CPU ref; FlashInfer via `--features flashinfer`).

mod backend;
mod chunked;
mod engine;
mod gpu;
mod kernel;
mod tokenizer;
mod weights;

pub use weights::{
    load_embed_head_norm, load_hc_head, load_layer0_attn, load_layer0_routed_moe,
    load_layer0_shared_ffn, load_layer_attn, load_layer_hc, load_layer_routed_moe,
    load_layer_shared_ffn, load_layer_stack, ExpertF32, ExpertPacked, HcBranchWeights,
    HcHeadWeights, HfModelConfig, Layer0AttnWeights, Layer0RoutedMoe, Layer0SharedFfn, LayerBlock,
    LayerHcWeights, LinearMat, PackedFp4Mat, PackedFp8Mat, SafetensorCatalog, TensorF32,
};
pub use tokenizer::HfTokenizer;

pub use backend::{
    EnginePrefixClient, HttpOpenAiBackend, KernelSmokeBackend, LocalEngineConfig,
    LocalEngineHandle, LocalWeightConfig, LocalWeightRunner, PagedKvPool, SglangLiteEngineBackend,
};
pub use chunked::{ChunkRequest, ChunkResult};
pub use engine::{
    GenerateRequest, GenerateResult, InferenceBackend, InferenceEngine, StubBackend, StubMode,
};
pub use gpu::GpuCapabilities;
pub use kernel::{
    CpuRefKernel, DecodeRequest, DecodeResult, KernelBackend, KvLayout, PrefillRequest,
    PrefillResult, SampleRequest, SampleResult,
};
#[cfg(feature = "flashinfer")]
pub use kernel::{discover_site_packages, FlashInferKernel, FlashInferKernelConfig};
