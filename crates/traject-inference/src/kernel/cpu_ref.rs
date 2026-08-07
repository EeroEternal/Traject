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
    attn_sink: Option<&[f32]>,
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
    use rayon::prelude::*;
    let scale = 1.0 / (head_dim as f32).sqrt();
    // Parallel over heads (independent softmax + V weighted sum).
    let head_outs: Vec<Vec<f32>> = (0..num_heads)
        .into_par_iter()
        .map(|h| {
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
            let sink = attn_sink.and_then(|s| s.get(h).copied());
            if let Some(sk) = sink {
                max_s = max_s.max(sk);
            }
            let mut sum = 0.0f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum += *s;
            }
            // Attention sink: absorb probability mass, no value contribution.
            if let Some(sk) = sink {
                sum += (sk - max_s).exp();
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for s in scores.iter_mut() {
                *s *= inv;
            }
            let mut oh = vec![0.0f32; head_dim];
            for s in 0..seq_len {
                let vh = &v[(s * num_heads + h) * head_dim..(s * num_heads + h + 1) * head_dim];
                for d in 0..head_dim {
                    oh[d] += scores[s] * vh[d];
                }
            }
            oh
        })
        .collect();
    let mut out = Vec::with_capacity(num_heads * head_dim);
    for oh in head_outs {
        out.extend_from_slice(&oh);
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
            let oi = softmax_attn(q, k, v, i + 1, h, d, None)?;
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
            req.attn_sink.as_deref(),
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
                attn_sink: None,
            })
            .await
            .unwrap();
        assert_eq!(out.o.len(), h * d);
    }

    #[tokio::test]
    async fn decode_with_sink_reduces_output_magnitude() {
        let k = CpuRefKernel;
        let h = 1usize;
        let d = 2usize;
        let s = 1usize;
        let q = vec![1.0f32, 0.0];
        let kc = vec![1.0f32, 0.0];
        let vc = vec![2.0f32, 0.0];
        let base = k
            .decode(DecodeRequest {
                q: q.clone(),
                k_cache: kc.clone(),
                v_cache: vc.clone(),
                seq_len: s as u32,
                num_heads: h as u32,
                head_dim: d as u32,
                layout: Default::default(),
                attn_sink: None,
            })
            .await
            .unwrap();
        let sunk = k
            .decode(DecodeRequest {
                q,
                k_cache: kc,
                v_cache: vc,
                seq_len: s as u32,
                num_heads: h as u32,
                head_dim: d as u32,
                layout: Default::default(),
                // Large sink → most mass absorbed → smaller o
                attn_sink: Some(vec![20.0]),
            })
            .await
            .unwrap();
        assert!(sunk.o[0].abs() < base.o[0].abs());
    }
}
