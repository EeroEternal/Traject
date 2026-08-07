//! InferenceBackend that drives in-process KernelBackend for Generate steps.
//!
//! This does **not** load a full LLM. It proves the Trajectory → Kernel path:
//! each Generate chunk runs FlashInfer/CPU attention on synthetic QKV, then
//! samples a token. Full weight loading plugs in behind the same KernelBackend.

use std::sync::Arc;

use async_trait::async_trait;
use traject_core::{FinishReason, Result, TrajectError};

use crate::kernel::{
    CpuRefKernel, DecodeRequest, KernelBackend, KvLayout, SampleRequest,
};
use crate::{ChunkRequest, ChunkResult, InferenceBackend};

pub struct KernelSmokeBackend {
    kernel: Arc<dyn KernelBackend>,
    num_heads: u32,
    head_dim: u32,
    /// Growing KV cache length per step (toy).
    cache_len: std::sync::Mutex<u32>,
}

impl KernelSmokeBackend {
    pub fn cpu() -> Self {
        Self {
            kernel: Arc::new(CpuRefKernel),
            num_heads: 4,
            head_dim: 32,
            cache_len: std::sync::Mutex::new(8),
        }
    }

    pub fn with_kernel(kernel: Arc<dyn KernelBackend>) -> Self {
        Self {
            kernel,
            num_heads: 4,
            head_dim: 32,
            cache_len: std::sync::Mutex::new(8),
        }
    }

    fn synth_qkv(&self, seq_len: u32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = self.num_heads as usize;
        let d = self.head_dim as usize;
        let s = seq_len as usize;
        let q = vec![0.05f32; h * d];
        let mut k = Vec::with_capacity(s * h * d);
        let mut v = Vec::with_capacity(s * h * d);
        for i in 0..s {
            let f = (i as f32 + 1.0) * 0.01;
            k.extend(std::iter::repeat(f).take(h * d));
            v.extend(std::iter::repeat(f * 2.0).take(h * d));
        }
        (q, k, v)
    }
}

#[async_trait]
impl InferenceBackend for KernelSmokeBackend {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult> {
        let seq_len = {
            let mut n = self
                .cache_len
                .lock()
                .map_err(|_| TrajectError::Other("cache lock".into()))?;
            *n = (*n + 1).min(64);
            *n
        };
        let (q, k, v) = self.synth_qkv(seq_len);
        let dec = self
            .kernel
            .decode(DecodeRequest {
                q,
                k_cache: k,
                v_cache: v,
                seq_len,
                num_heads: self.num_heads,
                head_dim: self.head_dim,
                layout: KvLayout::Nhd,
                attn_sink: None,
            })
            .await?;

        // Toy logits from attention output energy → sample a byte-ish token.
        let energy: f32 = dec.o.iter().map(|x| x.abs()).sum();
        let mut logits = vec![0.0f32; 256];
        let idx = ((energy * 1000.0) as u32 % 200) + 32;
        logits[idx as usize] = 10.0;
        let sampled = self
            .kernel
            .sample(SampleRequest {
                logits,
                temperature: req.constraints.temperature.unwrap_or(0.0),
                top_p: req.constraints.top_p.unwrap_or(1.0),
            })
            .await?;

        let ch = char::from_u32(sampled.token_id).unwrap_or('?');
        let text = format!("{ch}");
        let produced = 1u32;
        let finished = req.decoded_so_far + produced >= req.chunk_tokens.min(req.max_tokens).max(1)
            || req.decoded_so_far + produced >= 8;

        Ok(ChunkResult {
            text,
            token_ids: vec![sampled.token_id],
            tokens_produced: produced,
            finished,
            finish_reason: if finished {
                Some(FinishReason::Stop)
            } else {
                None
            },
            tool_call: None,
            new_prefix: None,
            cache_hit_tokens: 0,
        })
    }
}
