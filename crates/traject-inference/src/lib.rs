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
    load_embed_head_norm, load_layer0_attn, HfModelConfig, Layer0AttnWeights, SafetensorCatalog,
    TensorF32,
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
