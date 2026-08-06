//! In-process GPU kernel layer.
//!
//! This is the boundary between Trajectory scheduling and device execution:
//! Prefill / Decode / Sample operate on tensors and paged KV that live in the
//! same process as Traject (not over HTTP).

mod cpu_ref;
#[cfg(feature = "flashinfer")]
mod flashinfer_py;
mod types;

pub use cpu_ref::CpuRefKernel;
#[cfg(feature = "flashinfer")]
pub use flashinfer_py::{discover_site_packages, FlashInferKernel, FlashInferKernelConfig};
pub use types::{
    DecodeRequest, DecodeResult, KernelBackend, KvLayout, PrefillRequest, PrefillResult,
    SampleRequest, SampleResult,
};
