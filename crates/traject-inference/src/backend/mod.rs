//! Pluggable inference backends.

pub mod engine_prefix;
pub mod http_openai;
pub mod kernel_smoke;
pub mod local_engine;
pub mod local_runner;
pub mod sglang_lite;

pub use engine_prefix::EnginePrefixClient;
pub use http_openai::HttpOpenAiBackend;
pub use kernel_smoke::KernelSmokeBackend;
pub use local_engine::{LocalEngineConfig, LocalEngineHandle};
pub use local_runner::{LocalWeightConfig, LocalWeightRunner, PagedKvPool};
pub use sglang_lite::SglangLiteEngineBackend;
