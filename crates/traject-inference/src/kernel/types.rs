use async_trait::async_trait;
use traject_core::Result;

/// Layout for KV tensors passed into kernels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KvLayout {
    #[default]
    Nhd,
    Hnd,
}

impl KvLayout {
    pub fn as_flashinfer(self) -> &'static str {
        match self {
            KvLayout::Nhd => "NHD",
            KvLayout::Hnd => "HND",
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefillRequest {
    /// Flattened Q: [tokens * heads * dim] f16/bf16 bytes interpreted by backend.
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub num_tokens: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub layout: KvLayout,
}

#[derive(Debug, Clone)]
pub struct PrefillResult {
    pub o: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct DecodeRequest {
    pub q: Vec<f32>,
    pub k_cache: Vec<f32>,
    pub v_cache: Vec<f32>,
    pub seq_len: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub layout: KvLayout,
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub o: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct SampleRequest {
    pub logits: Vec<f32>,
    pub temperature: f32,
    pub top_p: f32,
}

#[derive(Debug, Clone)]
pub struct SampleResult {
    pub token_id: u32,
}

/// Device-side ops owned by Traject's process.
#[async_trait]
pub trait KernelBackend: Send + Sync {
    fn name(&self) -> &str;

    async fn prefill(&self, req: PrefillRequest) -> Result<PrefillResult>;
    async fn decode(&self, req: DecodeRequest) -> Result<DecodeResult>;
    async fn sample(&self, req: SampleRequest) -> Result<SampleResult>;
}
