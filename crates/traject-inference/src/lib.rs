//! Inference core — forward, sample, constrained decode.
//!
//! Backends: stub, OpenAI HTTP, sglang-lite engine, in-process KernelBackend
//! (CPU ref always; FlashInfer via `--features flashinfer`).

mod backend;
mod chunked;
mod engine;
mod gpu;
mod kernel;

pub use backend::{
    HttpOpenAiBackend, KernelSmokeBackend, LocalEngineConfig, LocalEngineHandle,
    SglangLiteEngineBackend,
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
pub use kernel::{FlashInferKernel, FlashInferKernelConfig};
