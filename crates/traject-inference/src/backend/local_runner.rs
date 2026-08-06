//! In-process weight runner with **physical paged KV** owned by Traject.
//!
//! This is the Phase-1 endgame shape (not full MoE parity yet):
//! - Tokenize (byte-hash toy or caller-provided ids)
//! - Embed + single-layer attention via [`KernelBackend`]
//! - Sample next token
//! - Store K/V in [`PagedKvPool`] keyed by MemoryManager-style prefix handles
//! - [`InferenceBackend::free_prefix`] zeros and drops physical pages
//!
//! Full DeepSeek-V4 weights still run in sglang-lite; this runner proves the
//! same-process ownership path MemoryManager + Driver expect.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use tracing::{debug, info};
use traject_core::{FinishReason, Result, TrajectError, TrajectoryId};

use crate::kernel::{
    CpuRefKernel, DecodeRequest, KernelBackend, KvLayout, PrefillRequest, SampleRequest,
};
use crate::{ChunkRequest, ChunkResult, InferenceBackend};

/// One physical KV page (mirrors engine block_size pages).
#[derive(Debug, Clone)]
struct KvPage {
    /// K flattened: tokens * heads * dim
    k: Vec<f32>,
    v: Vec<f32>,
    tokens: u32,
}

impl KvPage {
    fn empty(cap_tokens: usize, heads: usize, dim: usize) -> Self {
        let n = cap_tokens * heads * dim;
        Self {
            k: vec![0.0; n],
            v: vec![0.0; n],
            tokens: 0,
        }
    }
}

/// Physical paged KV store. Free zeros memory before drop.
#[derive(Debug, Default)]
pub struct PagedKvPool {
    page_tokens: usize,
    num_heads: usize,
    head_dim: usize,
    pages: HashMap<u64, KvPage>,
    next_id: u64,
    /// prefix_handle → ordered page ids for that sequence
    by_prefix: HashMap<String, Vec<u64>>,
    /// trajectory → active prefix handle
    by_traj: HashMap<TrajectoryId, String>,
}

impl PagedKvPool {
    pub fn new(page_tokens: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            page_tokens: page_tokens.max(1),
            num_heads: num_heads.max(1),
            head_dim: head_dim.max(1),
            pages: HashMap::new(),
            next_id: 1,
            by_prefix: HashMap::new(),
            by_traj: HashMap::new(),
        }
    }

    pub fn pages_allocated(&self) -> usize {
        self.pages.len()
    }

    fn alloc_page(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.pages.insert(
            id,
            KvPage::empty(self.page_tokens, self.num_heads, self.head_dim),
        );
        id
    }

    fn bind_prefix(&mut self, traj: TrajectoryId, prefix: String) {
        self.by_traj.insert(traj, prefix);
    }

    fn pages_for_prefix_mut(&mut self, prefix: &str) -> &mut Vec<u64> {
        self.by_prefix.entry(prefix.to_string()).or_default()
    }

    /// Append one token's K/V (size heads*dim each).
    fn append_kv(&mut self, prefix: &str, k: &[f32], v: &[f32]) {
        let need = self.num_heads * self.head_dim;
        assert_eq!(k.len(), need);
        assert_eq!(v.len(), need);
        let page_tokens = self.page_tokens;
        let ids = self.pages_for_prefix_mut(prefix);
        if ids.is_empty() {
            let id = {
                // alloc without double borrow
                let id = self.next_id;
                self.next_id += 1;
                self.pages.insert(
                    id,
                    KvPage::empty(page_tokens, self.num_heads, self.head_dim),
                );
                id
            };
            self.pages_for_prefix_mut(prefix).push(id);
        }
        let last = *self.pages_for_prefix_mut(prefix).last().unwrap();
        let page = self.pages.get_mut(&last).unwrap();
        if page.tokens as usize >= page_tokens {
            let id = self.alloc_page();
            self.pages_for_prefix_mut(prefix).push(id);
            let page = self.pages.get_mut(&id).unwrap();
            let off = 0;
            page.k[off..off + need].copy_from_slice(k);
            page.v[off..off + need].copy_from_slice(v);
            page.tokens = 1;
        } else {
            let off = page.tokens as usize * need;
            page.k[off..off + need].copy_from_slice(k);
            page.v[off..off + need].copy_from_slice(v);
            page.tokens += 1;
        }
    }

    fn materialize_kv(&self, prefix: &str) -> (Vec<f32>, Vec<f32>, u32) {
        let Some(ids) = self.by_prefix.get(prefix) else {
            return (Vec::new(), Vec::new(), 0);
        };
        let need = self.num_heads * self.head_dim;
        let mut k_all = Vec::new();
        let mut v_all = Vec::new();
        let mut tokens = 0u32;
        for id in ids {
            let Some(p) = self.pages.get(id) else {
                continue;
            };
            let n = p.tokens as usize * need;
            k_all.extend_from_slice(&p.k[..n]);
            v_all.extend_from_slice(&p.v[..n]);
            tokens += p.tokens;
        }
        (k_all, v_all, tokens)
    }

    /// Physical free: zero pages then drop. Returns pages freed.
    pub fn free_prefix(&mut self, prefix: &str) -> usize {
        let Some(ids) = self.by_prefix.remove(prefix) else {
            return 0;
        };
        let mut n = 0;
        for id in ids {
            if let Some(mut page) = self.pages.remove(&id) {
                // Explicit zero before drop (physical free contract).
                for x in &mut page.k {
                    *x = 0.0;
                }
                for x in &mut page.v {
                    *x = 0.0;
                }
                page.tokens = 0;
                n += 1;
            }
        }
        self.by_traj.retain(|_, p| p != prefix);
        debug!(%prefix, pages = n, "local runner freed physical KV pages");
        n
    }
}

/// Toy embedding + output projection (in-process "weights").
struct ToyWeights {
    vocab: u32,
    hidden: usize,
    /// vocab * hidden
    embed: Vec<f32>,
    /// hidden * vocab (output)
    unembed: Vec<f32>,
}

impl ToyWeights {
    fn new(vocab: u32, hidden: usize) -> Self {
        let v = vocab as usize;
        let mut embed = Vec::with_capacity(v * hidden);
        let mut unembed = Vec::with_capacity(v * hidden);
        for i in 0..v {
            for j in 0..hidden {
                let e = (((i * 131 + j * 17) % 1000) as f32) * 0.001 - 0.5;
                embed.push(e);
                unembed.push(e * 0.5);
            }
        }
        Self {
            vocab,
            hidden,
            embed,
            unembed,
        }
    }

    fn embed_token(&self, tid: u32) -> Vec<f32> {
        let i = (tid % self.vocab) as usize;
        let s = i * self.hidden;
        self.embed[s..s + self.hidden].to_vec()
    }

    fn logits(&self, h: &[f32]) -> Vec<f32> {
        let v = self.vocab as usize;
        let mut out = vec![0.0f32; v];
        for i in 0..v {
            let row = &self.unembed[i * self.hidden..(i + 1) * self.hidden];
            let mut s = 0.0;
            for (a, b) in h.iter().zip(row.iter()) {
                s += a * b;
            }
            out[i] = s;
        }
        out
    }
}

#[derive(Debug, Clone)]
pub struct LocalWeightConfig {
    pub vocab_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub page_tokens: usize,
    pub max_new_tokens_default: u32,
}

impl Default for LocalWeightConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            num_heads: 4,
            head_dim: 32,
            page_tokens: 16,
            max_new_tokens_default: 32,
        }
    }
}

/// In-process runner: physical KV + toy weights + KernelBackend attention.
pub struct LocalWeightRunner {
    kernel: Arc<dyn KernelBackend>,
    weights: ToyWeights,
    kv: Mutex<PagedKvPool>,
    cfg: LocalWeightConfig,
}

impl LocalWeightRunner {
    pub fn new(cfg: LocalWeightConfig) -> Self {
        let hidden = (cfg.num_heads * cfg.head_dim) as usize;
        Self {
            kernel: Arc::new(CpuRefKernel),
            weights: ToyWeights::new(cfg.vocab_size, hidden),
            kv: Mutex::new(PagedKvPool::new(
                cfg.page_tokens,
                cfg.num_heads as usize,
                cfg.head_dim as usize,
            )),
            cfg,
        }
    }

    pub fn with_kernel(cfg: LocalWeightConfig, kernel: Arc<dyn KernelBackend>) -> Self {
        let mut s = Self::new(cfg);
        s.kernel = kernel;
        s
    }

    pub fn pages_allocated(&self) -> usize {
        self.kv.lock().pages_allocated()
    }

    fn tokenize(text: &str, vocab: u32) -> Vec<u32> {
        if text.is_empty() {
            return vec![1];
        }
        text.bytes()
            .map(|b| (b as u32) % vocab.max(2))
            .collect()
    }

    fn detokenize(ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|t| {
                let b = (*t).min(255) as u8;
                if (32..127).contains(&b) {
                    Some(b as char)
                } else {
                    Some('.')
                }
            })
            .collect()
    }
}

#[async_trait]
impl InferenceBackend for LocalWeightRunner {
    async fn generate_chunk(&self, req: ChunkRequest) -> Result<ChunkResult> {
        let prefix = req
            .prefix_hint
            .clone()
            .or_else(|| req.prefix.map(|p| p.to_string()))
            .unwrap_or_else(|| req.trajectory_id.to_string());

        let prompt_ids = if !req.delta.token_ids.is_empty() {
            req.delta.token_ids.clone()
        } else {
            Self::tokenize(
                req.delta.text.as_deref().unwrap_or("?"),
                self.cfg.vocab_size,
            )
        };

        {
            let mut kv = self.kv.lock();
            kv.bind_prefix(req.trajectory_id, prefix.clone());
        }

        // Prefill: embed each prompt token and append to physical KV.
        let heads = self.cfg.num_heads as usize;
        let dim = self.cfg.head_dim as usize;
        let hidden = heads * dim;

        if req.decoded_so_far == 0 {
            for &tid in &prompt_ids {
                let emb = self.weights.embed_token(tid);
                // Toy: use embedding as K and V projection.
                let k = emb.clone();
                let v = emb.iter().map(|x| x * 0.5).collect::<Vec<_>>();
                self.kv.lock().append_kv(&prefix, &k[..hidden], &v[..hidden]);
            }
            // Prefill attention smoke on last token Q.
            if let Some(&last) = prompt_ids.last() {
                let q = self.weights.embed_token(last);
                let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
                if seq_len > 0 {
                    let _ = self
                        .kernel
                        .prefill(PrefillRequest {
                            q: q[..hidden].to_vec(),
                            k: k_cache.clone(),
                            v: v_cache.clone(),
                            num_tokens: 1,
                            num_heads: self.cfg.num_heads,
                            head_dim: self.cfg.head_dim,
                            layout: KvLayout::Nhd,
                        })
                        .await;
                }
            }
        }

        // Decode one (or few) tokens for this chunk.
        let budget = req
            .chunk_tokens
            .min(req.max_tokens.saturating_sub(req.decoded_so_far))
            .max(1)
            .min(8);
        let mut out_ids = Vec::new();
        let mut out_text = String::new();

        for _ in 0..budget {
            let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
            if seq_len == 0 {
                break;
            }
            // Q from last token embedding (approx).
            let last_tid = out_ids
                .last()
                .copied()
                .or_else(|| prompt_ids.last().copied())
                .unwrap_or(1);
            let q = self.weights.embed_token(last_tid);
            let dec = self
                .kernel
                .decode(DecodeRequest {
                    q: q[..hidden].to_vec(),
                    k_cache,
                    v_cache,
                    seq_len,
                    num_heads: self.cfg.num_heads,
                    head_dim: self.cfg.head_dim,
                    layout: KvLayout::Nhd,
                })
                .await
                .map_err(|e| TrajectError::Inference(format!("local decode: {e}")))?;

            let logits = self.weights.logits(&dec.o);
            let sampled = self
                .kernel
                .sample(SampleRequest {
                    logits,
                    temperature: req.constraints.temperature.unwrap_or(0.0),
                    top_p: req.constraints.top_p.unwrap_or(1.0),
                })
                .await?;

            let tid = sampled.token_id % self.cfg.vocab_size;
            // Append new KV for generated token.
            let emb = self.weights.embed_token(tid);
            let k = emb.clone();
            let v = emb.iter().map(|x| x * 0.5).collect::<Vec<_>>();
            self.kv.lock().append_kv(&prefix, &k[..hidden], &v[..hidden]);

            out_ids.push(tid);
            out_text.push_str(&Self::detokenize(&[tid]));

            // Stop on printable period-ish toy EOS.
            if tid % 97 == 0 && !out_ids.is_empty() {
                break;
            }
        }

        let produced = out_ids.len() as u32;
        let finished = req.decoded_so_far + produced >= req.max_tokens
            || produced < budget
            || req.decoded_so_far + produced >= req.chunk_tokens;

        info!(
            trajectory = %req.trajectory_id,
            prefix = %prefix,
            produced,
            pages = self.pages_allocated(),
            "local weight runner chunk"
        );

        Ok(ChunkResult {
            text: out_text,
            token_ids: out_ids,
            tokens_produced: produced.max(1),
            finished,
            finish_reason: if finished {
                Some(FinishReason::Stop)
            } else {
                None
            },
            tool_call: None,
            new_prefix: req.prefix,
            cache_hit_tokens: if req.decoded_so_far > 0 {
                req.decoded_so_far.min(16)
            } else {
                0
            },
        })
    }

    async fn free_prefix(&self, prefix_id: &str, _session_id: Option<&str>) -> Result<()> {
        let n = self.kv.lock().free_prefix(prefix_id);
        info!(%prefix_id, pages_zeroed = n, "local runner physical KV free");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ChunkRequest;
    use traject_core::{Constraints, GenerateDelta, StepId, TrajectoryId};

    #[tokio::test]
    async fn local_runner_generate_and_free() {
        let runner = LocalWeightRunner::new(LocalWeightConfig::default());
        let traj = TrajectoryId::new();
        let prefix = "pfx-local-1".to_string();
        let req = ChunkRequest {
            trajectory_id: traj,
            step_id: StepId::new(),
            prefix: None,
            delta: GenerateDelta::from_text("hello world"),
            constraints: Constraints::default(),
            chunk_tokens: 4,
            decoded_so_far: 0,
            max_tokens: 8,
            session_id: Some("sess".into()),
            prefix_hint: Some(prefix.clone()),
        };
        let out = runner.generate_chunk(req).await.unwrap();
        assert!(out.tokens_produced >= 1);
        assert!(runner.pages_allocated() >= 1);
        runner.free_prefix(&prefix, None).await.unwrap();
        assert_eq!(runner.pages_allocated(), 0);
    }
}
