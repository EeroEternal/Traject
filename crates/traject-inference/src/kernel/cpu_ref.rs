//! CPU reference attention — correctness oracle / CI without GPU.

use async_trait::async_trait;
use traject_core::{Result, TrajectError};

use super::{
    DecodeRequest, DecodeResult, KernelBackend, PrefillRequest, PrefillResult, SampleRequest,
    SampleResult,
};

pub struct CpuRefKernel;

fn softmax_attn(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    // q: [1, H, D], k/v: [S, H, D] in NHD
    if q.len() != num_heads * head_dim {
        return Err(TrajectError::Inference(format!(
            "q len {} != H*D {}",
            q.len(),
            num_heads * head_dim
        )));
    }
    if k.len() != seq_len * num_heads * head_dim || v.len() != k.len() {
        return Err(TrajectError::Inference("k/v shape mismatch".into()));
    }
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; num_heads * head_dim];
    for h in 0..num_heads {
        let qh = &q[h * head_dim..(h + 1) * head_dim];
        let mut scores = vec![0.0f32; seq_len];
        let mut max_s = f32::NEG_INFINITY;
        for s in 0..seq_len {
            let kh = &k[(s * num_heads + h) * head_dim..(s * num_heads + h + 1) * head_dim];
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += qh[d] * kh[d];
            }
            scores[s] = dot * scale;
            max_s = max_s.max(scores[s]);
        }
        let mut sum = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for s in scores.iter_mut() {
            *s /= sum;
        }
        let oh = &mut out[h * head_dim..(h + 1) * head_dim];
        for s in 0..seq_len {
            let vh = &v[(s * num_heads + h) * head_dim..(s * num_heads + h + 1) * head_dim];
            for d in 0..head_dim {
                oh[d] += scores[s] * vh[d];
            }
        }
    }
    Ok(out)
}

#[async_trait]
impl KernelBackend for CpuRefKernel {
    fn name(&self) -> &str {
        "cpu-ref"
    }

    async fn prefill(&self, req: PrefillRequest) -> Result<PrefillResult> {
        // Causal-naive: for each token attend to previous+self via decode-style.
        let h = req.num_heads as usize;
        let d = req.head_dim as usize;
        let t = req.num_tokens as usize;
        let mut o = Vec::with_capacity(t * h * d);
        for i in 0..t {
            let q = &req.q[i * h * d..(i + 1) * h * d];
            let k = &req.k[..(i + 1) * h * d];
            let v = &req.v[..(i + 1) * h * d];
            let oi = softmax_attn(q, k, v, i + 1, h, d)?;
            o.extend_from_slice(&oi);
        }
        Ok(PrefillResult { o })
    }

    async fn decode(&self, req: DecodeRequest) -> Result<DecodeResult> {
        let o = softmax_attn(
            &req.q,
            &req.k_cache,
            &req.v_cache,
            req.seq_len as usize,
            req.num_heads as usize,
            req.head_dim as usize,
        )?;
        Ok(DecodeResult { o })
    }

    async fn sample(&self, req: SampleRequest) -> Result<SampleResult> {
        if req.logits.is_empty() {
            return Err(TrajectError::Inference("empty logits".into()));
        }
        let temp = if req.temperature <= 0.0 {
            1.0
        } else {
            req.temperature
        };
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &v) in req.logits.iter().enumerate() {
            let s = v / temp;
            if s > best_v {
                best_v = s;
                best = i;
            }
        }
        Ok(SampleResult {
            token_id: best as u32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn decode_smoke() {
        let k = CpuRefKernel;
        let h = 2usize;
        let d = 4usize;
        let s = 3usize;
        let q = vec![0.1f32; h * d];
        let kc = vec![0.2f32; s * h * d];
        let vc = vec![0.3f32; s * h * d];
        let out = k
            .decode(DecodeRequest {
                q,
                k_cache: kc,
                v_cache: vc,
                seq_len: s as u32,
                num_heads: h as u32,
                head_dim: d as u32,
                layout: Default::default(),
            })
            .await
            .unwrap();
        assert_eq!(out.o.len(), h * d);
    }
}
