//! Pluggable inference backends.

pub mod http_openai;
pub mod kernel_smoke;
pub mod local_engine;
pub mod sglang_lite;

pub use http_openai::HttpOpenAiBackend;
pub use kernel_smoke::KernelSmokeBackend;
pub use local_engine::{LocalEngineConfig, LocalEngineHandle};
pub use sglang_lite::SglangLiteEngineBackend;
