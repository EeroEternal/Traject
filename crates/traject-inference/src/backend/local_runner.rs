//! In-process weight runner with **physical paged KV** owned by Traject.
//!
//! This is the Phase-1 endgame shape (not full MoE parity yet):
//! - Tokenize via HF `tokenizer.json` (or byte-hash toy fallback)
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
use tracing::{debug, info, warn};
use traject_core::{FinishReason, Result, TrajectError, TrajectoryId};

use crate::kernel::{
    CpuRefKernel, DecodeRequest, KernelBackend, KvLayout, SampleRequest,
};
use crate::tokenizer::HfTokenizer;
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

    /// Overwrite the last stored token's K/V (used to replace placeholder / crude KV
    /// with values from the full residual path during decode).
    fn overwrite_last_kv(&mut self, prefix: &str, k: &[f32], v: &[f32]) -> bool {
        let need = self.num_heads * self.head_dim;
        if k.len() < need || v.len() < need {
            return false;
        }
        let Some(ids) = self.by_prefix.get(prefix) else {
            return false;
        };
        let Some(&last) = ids.last() else {
            return false;
        };
        let Some(page) = self.pages.get_mut(&last) else {
            return false;
        };
        if page.tokens == 0 {
            return false;
        }
        let off = (page.tokens as usize - 1) * need;
        page.k[off..off + need].copy_from_slice(&k[..need]);
        page.v[off..off + need].copy_from_slice(&v[..need]);
        true
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

/// In-process model weights (toy or real safetensors + optional layer stack).
struct ModelWeights {
    vocab: u32,
    /// Model hidden size (e.g. 4096 for DeepSeek-V4).
    hidden: usize,
    /// Attention projection size = num_heads * head_dim (may be << hidden).
    attn_dim: usize,
    /// [vocab, hidden] row-major
    embed: Vec<f32>,
    /// [vocab, hidden] row-major (lm head)
    head: Vec<f32>,
    /// Optional final RMSNorm weights [hidden]
    norm: Option<Vec<f32>>,
    /// Fixed down-projection hidden → attn_dim (fallback when no layers)
    w_down: Vec<f32>,
    /// Up-projection attn_dim → hidden
    w_up: Vec<f32>,
    /// Loaded transformer blocks (layer 0..N-1).
    layers: Vec<crate::weights::LayerBlock>,
    source: String,
    eos_token_id: Option<u32>,
    /// Base RoPE theta (V4 Flash: 10000 for pure SWA layers).
    rope_theta: f32,
    /// Final Hyper-Connection collapse before lm_head.
    hc_head: Option<crate::weights::HcHeadWeights>,
}

impl ModelWeights {
    fn toy(vocab: u32, hidden: usize, attn_dim: usize) -> Self {
        let v = vocab as usize;
        let mut embed = Vec::with_capacity(v * hidden);
        let mut head = Vec::with_capacity(v * hidden);
        for i in 0..v {
            for j in 0..hidden {
                let e = (((i * 131 + j * 17) % 1000) as f32) * 0.001 - 0.5;
                embed.push(e);
                head.push(e * 0.5);
            }
        }
        let (w_down, w_up) = random_projections(hidden, attn_dim, 42);
        Self {
            vocab,
            hidden,
            attn_dim,
            embed,
            head,
            norm: None,
            w_down,
            w_up,
            layers: Vec::new(),
            source: "toy".into(),
            eos_token_id: Some(1),
            rope_theta: 10000.0,
            hc_head: None,
        }
    }

    fn from_safetensors(
        model_dir: &std::path::Path,
        attn_heads: u32,
        attn_dim_per_head: u32,
    ) -> Result<Self> {
        use crate::weights::{load_embed_head_norm, load_hc_head, load_layer_stack, HfModelConfig};

        let cfg = HfModelConfig::load(model_dir).ok();
        let (embed_t, head_t, norm_t, embed_key) = load_embed_head_norm(model_dir)?;
        if embed_t.shape.len() != 2 {
            return Err(TrajectError::Other(format!(
                "embed shape {:?} want [vocab, hidden]",
                embed_t.shape
            )));
        }
        let vocab = embed_t.shape[0] as u32;
        let hidden = embed_t.shape[1];
        if head_t.shape != embed_t.shape && !(head_t.shape.len() == 2 && head_t.shape[1] == hidden) {
            if head_t.rows() != vocab as usize || head_t.cols() != hidden {
                return Err(TrajectError::Other(format!(
                    "head shape {:?} incompatible with embed {:?}",
                    head_t.shape, embed_t.shape
                )));
            }
        }
        let mut attn_dim = (attn_heads * attn_dim_per_head) as usize;
        let n_layers = local_layer_count(cfg.as_ref());
        let layers = match load_layer_stack(model_dir, n_layers) {
            Ok(stack) => {
                if let Some(first) = stack.first() {
                    if first.attn.kv_dim() > 0 {
                        attn_dim = first.attn.kv_dim();
                    }
                }
                stack
            }
            Err(e) => {
                warn!(error = %e, "layer stack not loaded; using random projections");
                Vec::new()
            }
        };
        let (w_down, w_up) = random_projections(hidden, attn_dim, 7);
        let norm = norm_t.map(|t| t.data);
        let eos = cfg.as_ref().and_then(|c| c.eos_token_id);
        info!(
            dir = %model_dir.display(),
            vocab,
            hidden,
            embed_key = %embed_key,
            has_norm = norm.is_some(),
            n_layers = layers.len(),
            has_layer0 = !layers.is_empty(),
            has_layer0_ffn = layers.first().and_then(|l| l.ffn.as_ref()).is_some(),
            has_layer0_moe = layers.first().and_then(|l| l.moe.as_ref()).is_some(),
            attn_dim,
            model_type = ?cfg.as_ref().and_then(|c| c.model_type.clone()),
            "loaded real safetensors embed/head for local runner"
        );
        let rope_theta = cfg
            .as_ref()
            .and_then(|c| c.rope_theta)
            .unwrap_or(10000.0);
        let hc_head = match load_hc_head(model_dir) {
            Ok(h) => Some(h),
            Err(e) => {
                warn!(error = %e, "hc_head not loaded; will mean-collapse multi-stream if HC");
                None
            }
        };
        let has_hc = layers.iter().any(|l| l.hc.is_some());
        info!(
            has_hc,
            has_hc_head = hc_head.is_some(),
            hc_mult = layers
                .first()
                .and_then(|l| l.hc.as_ref().map(|h| h.hc_mult)),
            "Hyper-Connection status for local runner"
        );
        Ok(Self {
            vocab: cfg.as_ref().map(|c| c.vocab_size).unwrap_or(vocab).max(vocab),
            hidden,
            attn_dim,
            embed: embed_t.data,
            head: head_t.data,
            norm,
            w_down,
            w_up,
            layers,
            source: format!("safetensors:{}", model_dir.display()),
            eos_token_id: eos.or(Some(1)),
            rope_theta,
            hc_head,
        })
    }

    fn embed_token(&self, tid: u32) -> Vec<f32> {
        let i = (tid % self.vocab) as usize;
        let s = i * self.hidden;
        self.embed[s..s + self.hidden].to_vec()
    }

    fn rms_norm_with(&self, x: &[f32], gamma: Option<&[f32]>) -> Vec<f32> {
        let mut ss = 0.0f32;
        for v in x {
            ss += v * v;
        }
        let scale = (ss / x.len() as f32 + 1e-6).sqrt().recip();
        let mut out: Vec<f32> = x.iter().map(|v| v * scale).collect();
        if let Some(w) = gamma {
            for (o, wi) in out.iter_mut().zip(w.iter()) {
                *o *= *wi;
            }
        }
        out
    }

    fn rms_norm(&self, x: &[f32]) -> Vec<f32> {
        self.rms_norm_with(x, self.norm.as_deref())
    }

    /// Official Attention expects **already attn_normed** `x`.
    ///
    /// `q_lora = q_norm(wq_a(x))` — used by indexer as `qr`.
    fn project_q_lora(&self, layer: &crate::weights::Layer0AttnWeights, x_normed: &[f32]) -> Vec<f32> {
        let mut q = layer.wq_a.matvec(x_normed);
        if let Some(ref qn) = layer.q_norm {
            q = self.rms_norm_with(&q, Some(qn));
        }
        q
    }

    /// Multi-head Q from attn_normed `x` (no RoPE; caller applies position).
    fn project_q_layer(
        &self,
        layer: &crate::weights::Layer0AttnWeights,
        x_normed: &[f32],
    ) -> Vec<f32> {
        let q = self.project_q_lora(layer, x_normed);
        if let Some(ref wq_b) = layer.wq_b {
            let q_full = wq_b.matvec(&q);
            // Keep multi-head Q: first n_q_heads × head_dim (no mean-pool).
            let n_heads = layer.n_heads.unwrap_or(1).max(1);
            let head_dim = layer.kv_dim().max(1);
            let use_h = attn_heads_to_use(n_heads);
            let need = use_h * head_dim;
            let mut out = q_full;
            if out.len() > need {
                out.truncate(need);
            } else {
                out.resize(need, 0.0);
            }
            // Per-head RMSNorm after wq_b (official Attention.forward).
            for hi in 0..use_h {
                let sl = &mut out[hi * head_dim..(hi + 1) * head_dim];
                let mut ss = 0.0f32;
                for v in sl.iter() {
                    ss += v * v;
                }
                let inv = (ss / head_dim as f32 + 1e-6).sqrt().recip();
                for v in sl.iter_mut() {
                    *v *= inv;
                }
            }
            return out;
        }
        let mut out = q;
        out.resize(self.attn_dim, 0.0);
        out
    }

    /// Shared MQA latent (K=V) from attn_normed `x`. RoPE applied by caller.
    fn project_kv_layer(
        &self,
        layer: &crate::weights::Layer0AttnWeights,
        x_normed: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut kv = layer.wkv.matvec(x_normed);
        if let Some(ref kn) = layer.kv_norm {
            kv = self.rms_norm_with(&kv, Some(kn));
        }
        kv.resize(layer.kv_dim().max(1), 0.0);
        // Official V4: single latent is both K and V for sparse MLA.
        let v = kv.clone();
        (kv, v)
    }

    fn project_q(&self, h: &[f32]) -> Vec<f32> {
        if let Some(block) = self.layers.first() {
            let hn = self.rms_norm_with(h, Some(&block.attn.attn_norm));
            return self.project_q_layer(&block.attn, &hn);
        }
        matvec(&self.w_down, self.attn_dim, self.hidden, h)
    }

    fn project_kv(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if let Some(block) = self.layers.first() {
            let hn = self.rms_norm_with(h, Some(&block.attn.attn_norm));
            return self.project_kv_layer(&block.attn, &hn);
        }
        let a = matvec(&self.w_down, self.attn_dim, self.hidden, h);
        let v = a.iter().map(|x| x * 0.5).collect();
        (a, v)
    }

    fn attn_to_hidden_layer(
        &self,
        layer: &crate::weights::Layer0AttnWeights,
        attn_o: &[f32],
        _h_resid: &[f32],
    ) -> Vec<f32> {
        if let Some(ref wo_b) = layer.wo_b {
            let inter = layer.o_intermediate();
            let mut mid = vec![0.0f32; inter];
            let g = layer.o_groups.max(1);
            let lor = layer.o_lora_rank.max(1);
            let head_dim = layer.kv_dim().max(1);
            let n_heads_full = layer.n_heads.unwrap_or(1).max(1);
            let n_heads = if attn_o.len() >= head_dim && attn_o.len() % head_dim == 0 {
                (attn_o.len() / head_dim).min(n_heads_full).max(1)
            } else {
                1
            };
            // Official: o.view(groups, heads_per_group * head_dim) @ wo_a[g]
            // (concat heads in group — not mean-pool).
            let hpg = (n_heads_full / g).max(1);
            let group_in = hpg * head_dim;
            if let Some(ref wo_a) = layer.wo_a {
                if wo_a.rows() == inter && wo_a.cols() == group_in && n_heads >= hpg {
                    let groups_used = (n_heads / hpg).min(g).max(1);
                    for gi in 0..groups_used {
                        let gbase = gi * hpg * head_dim;
                        if gbase + group_in > attn_o.len() {
                            break;
                        }
                        let group = &attn_o[gbase..gbase + group_in];
                        for r in 0..lor {
                            mid[gi * lor + r] = wo_a.row_dot(gi * lor + r, group);
                        }
                    }
                } else if n_heads > 1 && attn_o.len() >= n_heads * head_dim {
                    // Fallback: mean-pool heads when group dims don't match (partial heads).
                    let hpg_use = (n_heads / g).max(1);
                    for gi in 0..g {
                        let mut pooled = vec![0.0f32; head_dim];
                        let mut count = 0usize;
                        for hi in 0..hpg_use {
                            let h = gi * hpg_use + hi;
                            if h >= n_heads {
                                break;
                            }
                            let base = h * head_dim;
                            for d in 0..head_dim {
                                pooled[d] += attn_o[base + d];
                            }
                            count += 1;
                        }
                        if count > 0 {
                            let inv = 1.0 / count as f32;
                            for d in pooled.iter_mut() {
                                *d *= inv;
                            }
                        }
                        let base = gi * lor;
                        let take = head_dim.min(lor);
                        for d in 0..take {
                            mid[base + d] += pooled[d];
                        }
                    }
                } else {
                    let take = attn_o.len().min(lor);
                    let scale = 1.0 / (g as f32).sqrt();
                    for gi in 0..g {
                        let base = gi * lor;
                        for d in 0..take {
                            mid[base + d] += attn_o[d] * scale;
                        }
                    }
                }
            } else {
                // No wo_a: inject raw o into o_lora slots.
                let take = attn_o.len().min(lor);
                let scale = 1.0 / (g as f32).sqrt();
                for gi in 0..g {
                    let base = gi * lor;
                    for d in 0..take {
                        mid[base + d] += attn_o[d] * scale;
                    }
                }
            }
            return wo_b.matvec(&mid);
        }
        matvec(&self.w_up, self.hidden, self.attn_dim, attn_o)
    }

    fn attn_to_hidden(&self, attn_o: &[f32], h_resid: &[f32]) -> Vec<f32> {
        if let Some(block) = self.layers.first() {
            return self.attn_to_hidden_layer(&block.attn, attn_o, h_resid);
        }
        matvec(&self.w_up, self.hidden, self.attn_dim, attn_o)
    }

    /// Shared-expert SwiGLU on **already ffn_normed** activations.
    fn shared_ffn_delta_normed(
        &self,
        ffn: &crate::weights::Layer0SharedFfn,
        n: &[f32],
    ) -> Vec<f32> {
        let mut u = ffn.w1.matvec(n);
        let mut g = ffn.w3.matvec(n);
        let limit = ffn.swiglu_limit;
        if limit > 0.0 {
            for v in &mut u {
                *v = v.min(limit);
            }
            for v in &mut g {
                *v = (*v).clamp(-limit, limit);
            }
        }
        let mut gated = vec![0.0f32; ffn.intermediate];
        for i in 0..ffn.intermediate {
            let x = u.get(i).copied().unwrap_or(0.0);
            let y = g.get(i).copied().unwrap_or(0.0);
            gated[i] = x * (1.0 / (1.0 + (-x).exp())) * y; // silu(u) * g
        }
        ffn.w2.matvec(&gated)
    }

    /// Pure shared-expert SwiGLU (applies ffn_norm then SwiGLU).
    fn shared_ffn_delta(&self, ffn: &crate::weights::Layer0SharedFfn, h: &[f32]) -> Vec<f32> {
        let n = self.rms_norm_with(h, Some(&ffn.ffn_norm));
        self.shared_ffn_delta_normed(ffn, &n)
    }

    fn shared_ffn_residual_block(
        &self,
        ffn: &crate::weights::Layer0SharedFfn,
        h: &[f32],
    ) -> Vec<f32> {
        let delta = self.shared_ffn_delta(ffn, h);
        let mut out = h.to_vec();
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += *d;
        }
        out
    }

    /// Pure routed MoE sum (no residual). Weights already include `route_scale`.
    ///
    /// Top-k expert SwiGLUs run in parallel (rayon); results are reduced serially.
    fn routed_moe_delta(
        &self,
        moe: &crate::weights::Layer0RoutedMoe,
        ffn_norm: Option<&[f32]>,
        h: &[f32],
        already_normed: bool,
        token_id: Option<u32>,
    ) -> Vec<f32> {
        use rayon::prelude::*;
        let n = if already_normed {
            h.to_vec()
        } else if let Some(g) = ffn_norm {
            self.rms_norm_with(h, Some(g))
        } else {
            h.to_vec()
        };
        let routes = moe.route(&n, token_id);
        let limit = moe.swiglu_limit;
        let parts: Vec<(usize, f32, traject_core::Result<Vec<f32>>)> = routes
            .into_par_iter()
            .map(|(eid, w)| {
                let out = moe.expert(eid).and_then(|ex| ex.swiglu(&n, limit));
                (eid, w, out)
            })
            .collect();
        let mut delta = vec![0.0f32; moe.hidden];
        for (eid, w, res) in parts {
            match res {
                Ok(d) => {
                    for (o, x) in delta.iter_mut().zip(d.iter()) {
                        *o += w * *x;
                    }
                }
                Err(e) => {
                    warn!(expert = eid, error = %e, "routed expert load/swiglu failed; skip");
                }
            }
        }
        delta
    }

    fn routed_moe_residual_block(
        &self,
        moe: &crate::weights::Layer0RoutedMoe,
        ffn_norm: Option<&[f32]>,
        h: &[f32],
        token_id: Option<u32>,
    ) -> Vec<f32> {
        let delta = self.routed_moe_delta(moe, ffn_norm, h, false, token_id);
        let mut out = h.to_vec();
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += *d;
        }
        out
    }

    /// Official MoE: **one** `ffn_norm`, then shared + routed on the same tensor.
    fn moe_block_delta(
        &self,
        ffn: Option<&crate::weights::Layer0SharedFfn>,
        moe: Option<&crate::weights::Layer0RoutedMoe>,
        h: &[f32],
        token_id: Option<u32>,
    ) -> Vec<f32> {
        let mut y = vec![0.0f32; h.len()];
        if let Some(ffn) = ffn {
            // Official Block: x = ffn_norm(x); y = shared(x) + routed(x)
            let n = self.rms_norm_with(h, Some(&ffn.ffn_norm));
            let d = self.shared_ffn_delta_normed(ffn, &n);
            for (o, x) in y.iter_mut().zip(d.iter()) {
                *o += *x;
            }
            if let Some(moe) = moe {
                let d = self.routed_moe_delta(moe, None, &n, true, token_id);
                for (o, x) in y.iter_mut().zip(d.iter()) {
                    *o += *x;
                }
            }
        } else if let Some(moe) = moe {
            y = self.routed_moe_delta(moe, None, h, false, token_id);
        }
        y
    }

    fn has_hc(&self) -> bool {
        self.layers.iter().any(|l| l.hc.is_some())
    }

    fn has_layer0(&self) -> bool {
        !self.layers.is_empty()
    }

    fn has_layer0_ffn(&self) -> bool {
        self.layers.first().and_then(|l| l.ffn.as_ref()).is_some()
    }

    fn has_layer0_moe(&self) -> bool {
        self.layers.first().and_then(|l| l.moe.as_ref()).is_some()
    }

    fn n_layers(&self) -> usize {
        self.layers.len()
    }

    fn logits(&self, h: &[f32]) -> Vec<f32> {
        use rayon::prelude::*;
        let h = self.rms_norm(h);
        let v = self.vocab as usize;
        let hidden = self.hidden;
        let head = &self.head;
        // head is [vocab, hidden] — logits[i] = dot(head[i], h); parallel over vocab.
        (0..v)
            .into_par_iter()
            .map(|i| {
                let row = &head[i * hidden..(i + 1) * hidden];
                let mut s = 0.0f32;
                for (a, b) in h.iter().zip(row.iter()) {
                    s += a * b;
                }
                s
            })
            .collect()
    }
}

fn random_projections(hidden: usize, attn_dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut state = seed;
    let mut rnd = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1);
        ((state >> 33) as f32 / u32::MAX as f32) * 0.02 - 0.01
    };
    let mut w_down = Vec::with_capacity(attn_dim * hidden);
    for _ in 0..attn_dim * hidden {
        w_down.push(rnd());
    }
    let mut w_up = Vec::with_capacity(hidden * attn_dim);
    for _ in 0..hidden * attn_dim {
        w_up.push(rnd());
    }
    (w_down, w_up)
}


/// Q heads for KernelBackend (`TRAJECT_ATTN_HEADS`).
///
/// Default = **all model heads** (V4 Flash: 64). Set `TRAJECT_ATTN_HEADS=8` etc.
/// for faster CPU smoke.
fn attn_heads_to_use(model_heads: usize) -> usize {
    let env = std::env::var("TRAJECT_ATTN_HEADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let want = env.unwrap_or(model_heads.max(1)).max(1);
    want.min(model_heads.max(1))
}

/// Resolve multi-head MQA layout from loaded weights.
///
/// Returns `(multihead, n_q_heads, head_dim, pool_heads, pool_dim)`.
fn resolve_attn_layout(
    weights: &ModelWeights,
    cfg: &LocalWeightConfig,
) -> (bool, usize, usize, usize, usize) {
    if let Some(block) = weights.layers.first() {
        if block.attn.has_q_expand() {
            let head_dim = block.attn.kv_dim().max(1);
            let model_h = block.attn.n_heads.unwrap_or(1).max(1);
            let n_q = attn_heads_to_use(model_h);
            // Store compressed KV only (1 × head_dim).
            return (true, n_q, head_dim, 1, head_dim);
        }
    }
    let heads = cfg.num_heads.max(1) as usize;
    let dim = (weights.attn_dim / heads).max(1);
    (false, heads, dim, heads, dim)
}

/// Expand compressed KV `[seq * head_dim]` → MQA `[seq * n_q_heads * head_dim]`.
fn expand_kv_mqa(kv: &[f32], seq: usize, n_q_heads: usize, head_dim: usize) -> Vec<f32> {
    let need = seq * head_dim;
    let src = if kv.len() >= need {
        &kv[..need]
    } else {
        kv
    };
    let mut out = vec![0.0f32; seq * n_q_heads * head_dim];
    for s in 0..seq {
        let base_in = s * head_dim;
        if base_in + head_dim > src.len() {
            break;
        }
        for h in 0..n_q_heads {
            let base_out = (s * n_q_heads + h) * head_dim;
            out[base_out..base_out + head_dim]
                .copy_from_slice(&src[base_in..base_in + head_dim]);
        }
    }
    out
}

/// RoPE on multi-head layout `[H * D]` using per-layer [`RopeParams`] (base or YaRN).
fn apply_rope_heads(
    x: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    rope: &crate::weights::RopeParams,
    pos: usize,
    inverse: bool,
) {
    if n_heads == 0 || head_dim == 0 {
        return;
    }
    for h in 0..n_heads {
        let base = h * head_dim;
        if base + head_dim > x.len() {
            break;
        }
        rope.apply_slice(&mut x[base..base + head_dim], pos, inverse);
    }
}

/// RoPE on a single MQA latent of length `head_dim`.
fn apply_rope_latent(
    x: &mut [f32],
    head_dim: usize,
    rope: &crate::weights::RopeParams,
    pos: usize,
) {
    if x.len() < head_dim {
        return;
    }
    rope.apply_slice(&mut x[..head_dim], pos, false);
}

/// Official attention KV path: RoPE then FP8 QAT on **no-RoPE** dims
/// (`act_quant(kv[..., :-rd], 64, ue8m0, inplace)`).
fn finalize_attn_kv_latent(
    x: &mut [f32],
    head_dim: usize,
    rope: &crate::weights::RopeParams,
    pos: usize,
) {
    apply_rope_latent(x, head_dim, rope, pos);
    let d = head_dim.min(x.len());
    if d == 0 {
        return;
    }
    let rd = rope.rope_dim.min(d);
    crate::weights::dtype::fp8_act_quant_nope_inplace(&mut x[..d], rd, 64);
}

/// Store layer K/V for absolute position `abs_pos`.
///
/// - `cache_len == abs_pos` → **append** (new token, like official write at start_pos)
/// - `cache_len == abs_pos + 1` → **overwrite last** (refresh residual for that token)
/// - otherwise → append (best-effort recovery)
///
/// Returns `(seq_len_after, did_append)`.
fn store_layer_kv_at_pos(
    kv: &Mutex<PagedKvPool>,
    pfx: &str,
    k: &[f32],
    v: &[f32],
    abs_pos: usize,
) -> (u32, bool) {
    let mut guard = kv.lock();
    let prev = guard.materialize_kv(pfx).2 as usize;
    if prev == abs_pos {
        guard.append_kv(pfx, k, v);
        ((prev + 1) as u32, true)
    } else if prev == abs_pos + 1 {
        let ok = guard.overwrite_last_kv(pfx, k, v);
        if !ok {
            guard.append_kv(pfx, k, v);
            return ((prev + 1) as u32, true);
        }
        (prev as u32, false)
    } else if prev == 0 {
        guard.append_kv(pfx, k, v);
        (1, true)
    } else {
        // Cache ahead/behind abs_pos — append to avoid stalling decode.
        guard.append_kv(pfx, k, v);
        ((prev + 1) as u32, true)
    }
}

/// First `n_q` entries of layer `attn_sink`, if present.
fn sink_for_heads(layer: &crate::weights::Layer0AttnWeights, n_q: usize) -> Option<Vec<f32>> {
    layer.attn_sink.as_ref().map(|s| {
        let mut out = s[..s.len().min(n_q)].to_vec();
        out.resize(n_q, 0.0);
        out
    })
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Expand single hidden → `hc_mult` identical streams (layout `[hc][d]` flat).
fn expand_hc_streams(h: &[f32], hc_mult: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(hc_mult * h.len());
    for _ in 0..hc_mult.max(1) {
        out.extend_from_slice(h);
    }
    out
}

/// Official `hc_split_sinkhorn` (CPU): pre / post / comb from mixes.
fn hc_split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    hc: usize,
    iters: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pre = vec![0.0f32; hc];
    let mut post = vec![0.0f32; hc];
    let s0 = scale.first().copied().unwrap_or(1.0);
    let s1 = scale.get(1).copied().unwrap_or(1.0);
    let s2 = scale.get(2).copied().unwrap_or(1.0);
    for j in 0..hc {
        pre[j] = sigmoid(mixes[j] * s0 + base[j]) + eps;
        post[j] = 2.0 * sigmoid(mixes[j + hc] * s1 + base.get(j + hc).copied().unwrap_or(0.0));
    }
    let mut comb = vec![0.0f32; hc * hc];
    for j in 0..hc {
        for k in 0..hc {
            let idx = j * hc + k + hc * 2;
            let b = base.get(idx).copied().unwrap_or(0.0);
            comb[j * hc + k] = mixes.get(idx).copied().unwrap_or(0.0) * s2 + b;
        }
    }
    // softmax over last dim + eps
    for j in 0..hc {
        let row = &mut comb[j * hc..(j + 1) * hc];
        let mut max_v = f32::NEG_INFINITY;
        for &v in row.iter() {
            max_v = max_v.max(v);
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
        for v in row.iter_mut() {
            *v = *v * inv + eps;
        }
    }
    let col_normalize = |comb: &mut [f32]| {
        let mut col_sum = vec![0.0f32; hc];
        for j in 0..hc {
            for k in 0..hc {
                col_sum[k] += comb[j * hc + k];
            }
        }
        for j in 0..hc {
            for k in 0..hc {
                comb[j * hc + k] /= col_sum[k] + eps;
            }
        }
    };
    let row_normalize = |comb: &mut [f32]| {
        for j in 0..hc {
            let mut s = 0.0f32;
            for k in 0..hc {
                s += comb[j * hc + k];
            }
            for k in 0..hc {
                comb[j * hc + k] /= s + eps;
            }
        }
    };
    col_normalize(&mut comb);
    for _ in 0..iters.saturating_sub(1) {
        row_normalize(&mut comb);
        col_normalize(&mut comb);
    }
    (pre, post, comb)
}

/// HC pre: multi-stream → single stream + post/comb for post.
fn hc_pre(
    streams: &[f32],
    branch: &crate::weights::HcBranchWeights,
    hc_mult: usize,
    hidden: usize,
    sinkhorn_iters: usize,
    eps: f32,
    norm_eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let hc_dim = hc_mult * hidden;
    let mix_hc = (2 + hc_mult) * hc_mult;
    let x = if streams.len() >= hc_dim {
        &streams[..hc_dim]
    } else {
        streams
    };
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let rsqrt = (ss / x.len().max(1) as f32 + norm_eps).sqrt().recip();
    // mixes = (fn_w @ x) * rsqrt ; fn_w: [mix_hc, hc_dim]
    let mut mixes = vec![0.0f32; mix_hc];
    let cols = branch.fn_w.cols().max(1);
    for i in 0..mix_hc.min(branch.fn_w.rows()) {
        let row = &branch.fn_w.data[i * cols..(i + 1) * cols];
        let mut s = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            s += a * b;
        }
        mixes[i] = s * rsqrt;
    }
    let (pre, post, comb) = hc_split_sinkhorn(
        &mixes,
        &branch.scale,
        &branch.base,
        hc_mult,
        sinkhorn_iters,
        eps,
    );
    // y = sum_i pre[i] * stream[i]
    let mut y = vec![0.0f32; hidden];
    for i in 0..hc_mult {
        let base = i * hidden;
        if base + hidden > streams.len() {
            break;
        }
        let p = pre[i];
        for d in 0..hidden {
            y[d] += p * streams[base + d];
        }
    }
    (y, post, comb)
}

/// HC post: single stream + residual multi-stream → multi-stream.
///
/// `y[j] = post[j] * x + residual[j] * sum_i comb[i,j]` (matches official dims).
fn hc_post(
    x: &[f32],
    residual: &[f32],
    post: &[f32],
    comb: &[f32],
    hc_mult: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut col_sum = vec![0.0f32; hc_mult];
    for i in 0..hc_mult {
        for j in 0..hc_mult {
            col_sum[j] += comb.get(i * hc_mult + j).copied().unwrap_or(0.0);
        }
    }
    let mut out = vec![0.0f32; hc_mult * hidden];
    for j in 0..hc_mult {
        let p = post.get(j).copied().unwrap_or(0.0);
        let c = col_sum[j];
        let rbase = j * hidden;
        let obase = j * hidden;
        for d in 0..hidden {
            let xv = x.get(d).copied().unwrap_or(0.0);
            let rv = residual.get(rbase + d).copied().unwrap_or(0.0);
            out[obase + d] = p * xv + c * rv;
        }
    }
    out
}

/// Collapse multi-stream → single via `hc_head_*`.
fn hc_head_collapse(streams: &[f32], head: &crate::weights::HcHeadWeights) -> Vec<f32> {
    let hc = head.hc_mult.max(1);
    let hidden = head.hidden.max(1);
    let hc_dim = hc * hidden;
    let x = if streams.len() >= hc_dim {
        &streams[..hc_dim]
    } else {
        streams
    };
    let mut ss = 0.0f32;
    for v in x {
        ss += v * v;
    }
    let rsqrt = (ss / x.len().max(1) as f32 + 1e-6).sqrt().recip();
    let cols = head.fn_w.cols().max(1);
    let mut mixes = vec![0.0f32; hc];
    for i in 0..hc.min(head.fn_w.rows()) {
        let row = &head.fn_w.data[i * cols..(i + 1) * cols];
        let mut s = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            s += a * b;
        }
        mixes[i] = s * rsqrt;
    }
    let mut pre = vec![0.0f32; hc];
    for i in 0..hc {
        let b = head.base.get(i).copied().unwrap_or(0.0);
        pre[i] = sigmoid(mixes[i] * head.scale + b) + head.eps;
    }
    let mut y = vec![0.0f32; hidden];
    for i in 0..hc {
        let base = i * hidden;
        if base + hidden > streams.len() {
            break;
        }
        let p = pre[i];
        for d in 0..hidden {
            y[d] += p * streams[base + d];
        }
    }
    y
}

/// Mean-collapse multi-stream when hc_head is missing.
fn mean_collapse_streams(streams: &[f32], hc_mult: usize, hidden: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; hidden];
    let n = hc_mult.max(1) as f32;
    for i in 0..hc_mult {
        let base = i * hidden;
        if base + hidden > streams.len() {
            break;
        }
        for d in 0..hidden {
            y[d] += streams[base + d];
        }
    }
    for v in y.iter_mut() {
        *v /= n;
    }
    y
}

/// How many transformer layers to load for the local runner.
///
/// - `TRAJECT_LOCAL_LAYERS` — requested count (default **4**, min 1)
/// - `TRAJECT_LOCAL_LAYERS_MAX` — hard cap (default **43** = full V4 Flash depth,
///   never above model `num_hidden_layers`)
///
/// Packed FP8 dense is ~0.13 GiB/layer; 43 layers ~6 GiB dense (+ embed/head).
fn local_layer_count(cfg: Option<&crate::weights::HfModelConfig>) -> usize {
    let env = std::env::var("TRAJECT_LOCAL_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let n = env.unwrap_or(4).max(1);
    let max_model = cfg
        .and_then(|c| c.num_hidden_layers)
        .map(|x| x as usize)
        .unwrap_or(43);
    let cap = std::env::var("TRAJECT_LOCAL_LAYERS_MAX")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(max_model.max(1))
        .max(1)
        .min(max_model);
    n.min(cap)
}

/// Sliding-window size (`sliding_window` / `window_size` in config, default 128).
///
/// `TRAJECT_SLIDING_WINDOW=0` disables windowing (full context).
fn resolve_sliding_window(cfg: Option<&crate::weights::HfModelConfig>) -> usize {
    if let Ok(s) = std::env::var("TRAJECT_SLIDING_WINDOW") {
        if let Ok(v) = s.parse::<usize>() {
            return v; // 0 = disabled
        }
    }
    cfg.and_then(|c| c.sliding_window)
        .map(|x| x as usize)
        .unwrap_or(128)
        .max(0)
}

/// Keep only the last `window` tokens of a NHD-style cache `[S * width]`.
///
/// `width` is `heads * head_dim` (expanded) or `head_dim` (compressed MQA).
fn slice_kv_window(
    k: Vec<f32>,
    v: Vec<f32>,
    seq_len: u32,
    width: usize,
    window: usize,
) -> (Vec<f32>, Vec<f32>, u32) {
    let s = seq_len as usize;
    if window == 0 || s <= window || width == 0 {
        return (k, v, seq_len);
    }
    let keep = window;
    let start = s - keep;
    let byte_start = start * width;
    let need = keep * width;
    let k_out = if k.len() >= byte_start + need {
        k[byte_start..byte_start + need].to_vec()
    } else {
        k
    };
    let v_out = if v.len() >= byte_start + need {
        v[byte_start..byte_start + need].to_vec()
    } else {
        v
    };
    (k_out, v_out, keep as u32)
}

/// Max strided compress tokens to keep (matches V4 `index_topk` order of magnitude).
fn max_compress_tokens() -> usize {
    std::env::var("TRAJECT_COMPRESS_TOPK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512)
        .max(1)
}

/// Indices for sliding-window tokens + strided history (compress layers).
///
/// Official path uses a learned compressor + indexer; this keeps a **strided
/// full-token** stand-in for compressed slots so long-range tokens remain
/// visible when `compress_ratio > 1`.
fn build_sparse_kv_indices(
    seq_len: usize,
    window: usize,
    compress_ratio: usize,
    max_compress: usize,
) -> Vec<usize> {
    if seq_len == 0 {
        return Vec::new();
    }
    let win = if window == 0 {
        seq_len
    } else {
        window.min(seq_len)
    };
    let win_start = seq_len - win;
    let mut idxs = Vec::with_capacity(win + max_compress.min(win_start / compress_ratio.max(1)));
    if compress_ratio > 1 && win_start > 0 {
        // One token per compress block in the pre-window history (block end).
        let mut p = compress_ratio - 1;
        let mut n = 0usize;
        while p < win_start && n < max_compress {
            idxs.push(p);
            p += compress_ratio;
            n += 1;
        }
    }
    for i in win_start..seq_len {
        idxs.push(i);
    }
    idxs
}

/// Gather selected token slots from `[S * width]` K/V caches.
fn gather_kv_by_idx(
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    width: usize,
    idxs: &[usize],
) -> (Vec<f32>, Vec<f32>, u32) {
    let width = width.max(1);
    let n = idxs.len();
    let mut ko = vec![0.0f32; n * width];
    let mut vo = vec![0.0f32; n * width];
    for (o, &i) in idxs.iter().enumerate() {
        if i >= seq_len {
            continue;
        }
        let src = i * width;
        if src + width <= k.len() {
            ko[o * width..(o + 1) * width].copy_from_slice(&k[src..src + width]);
        }
        if src + width <= v.len() {
            vo[o * width..(o + 1) * width].copy_from_slice(&v[src..src + width]);
        }
    }
    (ko, vo, n as u32)
}

/// Window (+ learned compress pool or strided history) then MQA expand.
///
/// `compress_pool`: optional `(k, v, len)` from `{prefix}:L{i}:C` when a learned
/// compressor is loaded. When present, window fine tokens are concatenated with
/// the compress pool (official layout stand-in: window ‖ compressed).
fn prepare_kv_for_decode(
    multihead: bool,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    seq_len: u32,
    n_q: usize,
    head_dim: usize,
    window: usize,
    compress_ratio: usize,
    compress_pool: Option<(Vec<f32>, Vec<f32>, u32)>,
) -> (Vec<f32>, Vec<f32>, u32) {
    let width = if multihead {
        head_dim.max(1)
    } else {
        (n_q * head_dim).max(1)
    };
    let s = seq_len as usize;
    let (k, v, sl) = if let Some((ck, cv, cl)) = compress_pool {
        if cl > 0 {
            let (mut wk, mut wv, wl) = slice_kv_window(k_cache, v_cache, seq_len, width, window);
            let cneed = cl as usize * width;
            if ck.len() >= cneed && cv.len() >= cneed {
                wk.extend_from_slice(&ck[..cneed]);
                wv.extend_from_slice(&cv[..cneed]);
                (wk, wv, wl + cl)
            } else {
                (wk, wv, wl)
            }
        } else if compress_ratio > 1 && s > window.max(1) && window > 0 {
            let idxs = build_sparse_kv_indices(s, window, compress_ratio, max_compress_tokens());
            gather_kv_by_idx(&k_cache, &v_cache, s, width, &idxs)
        } else {
            slice_kv_window(k_cache, v_cache, seq_len, width, window)
        }
    } else if compress_ratio > 1 && s > window.max(1) && window > 0 {
        let idxs = build_sparse_kv_indices(s, window, compress_ratio, max_compress_tokens());
        gather_kv_by_idx(&k_cache, &v_cache, s, width, &idxs)
    } else {
        slice_kv_window(k_cache, v_cache, seq_len, width, window)
    };
    if multihead {
        (
            expand_kv_mqa(&k, sl as usize, n_q, head_dim),
            expand_kv_mqa(&v, sl as usize, n_q, head_dim),
            sl,
        )
    } else {
        (k, v, sl)
    }
}

/// Per-layer incremental compressor state (official `kv_state` / `score_state`).
#[derive(Debug, Clone)]
struct CompressLayerState {
    /// Flat `[slots * out_dim]` for kv projections in the open block.
    kv_state: Vec<f32>,
    score_state: Vec<f32>,
    tokens_seen: usize,
}

impl CompressLayerState {
    fn new(comp: &crate::weights::CompressorWeights) -> Self {
        let slots = if comp.overlap {
            2 * comp.ratio
        } else {
            comp.ratio
        };
        let out_dim = comp.out_dim();
        let n = slots * out_dim;
        Self {
            kv_state: vec![0.0; n],
            score_state: vec![f32::NEG_INFINITY; n],
            tokens_seen: 0,
        }
    }
}

/// Push one token into the compressor; when a block completes return `head_dim` latent.
fn compressor_push(
    comp: &crate::weights::CompressorWeights,
    state: &mut CompressLayerState,
    x: &[f32],
    pos: usize,
    rope: &crate::weights::RopeParams,
) -> Option<Vec<f32>> {
    let ratio = comp.ratio.max(1);
    let d = comp.head_dim.max(1);
    let out_dim = comp.out_dim();
    let mut kv = comp.wkv.matvec(x);
    let mut score = comp.wgate.matvec(x);
    kv.resize(out_dim, 0.0);
    score.resize(out_dim, 0.0);

    if !comp.overlap {
        let r = pos % ratio;
        // ape[r]
        let ape_off = r * out_dim;
        for j in 0..out_dim {
            let a = comp.ape.get(ape_off + j).copied().unwrap_or(0.0);
            score[j] += a;
        }
        let base = r * out_dim;
        state.kv_state[base..base + out_dim].copy_from_slice(&kv);
        state.score_state[base..base + out_dim].copy_from_slice(&score);
        state.tokens_seen = pos + 1;
        if (pos + 1) % ratio != 0 {
            return None;
        }
        // Softmax over ratio for each dim, weighted sum of kv → [d] (out_dim==d).
        let mut out = vec![0.0f32; d];
        for j in 0..d.min(out_dim) {
            let mut max_s = f32::NEG_INFINITY;
            for r in 0..ratio {
                max_s = max_s.max(state.score_state[r * out_dim + j]);
            }
            let mut sum = 0.0f32;
            let mut acc = 0.0f32;
            for r in 0..ratio {
                let e = (state.score_state[r * out_dim + j] - max_s).exp();
                sum += e;
                acc += e * state.kv_state[r * out_dim + j];
            }
            out[j] = if sum > 0.0 { acc / sum } else { 0.0 };
        }
        // RMSNorm with compressor.norm
        let mut ss = 0.0f32;
        for v in &out {
            ss += v * v;
        }
        let inv = (ss / d as f32 + 1e-6).sqrt().recip();
        for (j, v) in out.iter_mut().enumerate() {
            let g = comp.norm.get(j).copied().unwrap_or(1.0);
            *v = *v * inv * g;
        }
        // RoPE at last token of the block
        rope.apply_slice(&mut out, pos, false);
        // Official: rotate=True → Hadamard+FP4; else FP8 QAT on no-RoPE dims.
        if comp.rotate {
            crate::weights::dtype::indexer_qk_qat_inplace(&mut out, 32);
        } else {
            let rd = rope.rope_dim.min(out.len());
            crate::weights::dtype::fp8_act_quant_nope_inplace(&mut out, rd, 64);
        }
        return Some(out);
    }

    // Overlap path (ratio==4): store current half at slots [ratio + pos%ratio].
    let r = pos % ratio;
    let ape_off = r * out_dim;
    for j in 0..out_dim {
        let a = comp.ape.get(ape_off + j).copied().unwrap_or(0.0);
        score[j] += a;
    }
    let slot = ratio + r;
    let base = slot * out_dim;
    if base + out_dim <= state.kv_state.len() {
        state.kv_state[base..base + out_dim].copy_from_slice(&kv);
        state.score_state[base..base + out_dim].copy_from_slice(&score);
    }
    state.tokens_seen = pos + 1;
    if (pos + 1) % ratio != 0 {
        return None;
    }
    // cat: first half dims from slots[:ratio], second half from slots[ratio:]
    // → [2*ratio, d], then softmax over 2*ratio and sum.
    let n_pool = 2 * ratio;
    let mut pool_kv = vec![0.0f32; n_pool * d];
    let mut pool_sc = vec![f32::NEG_INFINITY; n_pool * d];
    for i in 0..ratio {
        let src0 = i * out_dim;
        let src1 = (ratio + i) * out_dim;
        for j in 0..d {
            // first half of out_dim from early slots, second half from late slots
            pool_kv[i * d + j] = state.kv_state.get(src0 + j).copied().unwrap_or(0.0);
            pool_sc[i * d + j] = state.score_state.get(src0 + j).copied().unwrap_or(f32::NEG_INFINITY);
            pool_kv[(ratio + i) * d + j] = state
                .kv_state
                .get(src1 + d + j)
                .copied()
                .unwrap_or(0.0);
            pool_sc[(ratio + i) * d + j] = state
                .score_state
                .get(src1 + d + j)
                .copied()
                .unwrap_or(f32::NEG_INFINITY);
        }
    }
    let mut out = vec![0.0f32; d];
    for j in 0..d {
        let mut max_s = f32::NEG_INFINITY;
        for i in 0..n_pool {
            max_s = max_s.max(pool_sc[i * d + j]);
        }
        let mut sum = 0.0f32;
        let mut acc = 0.0f32;
        for i in 0..n_pool {
            let e = (pool_sc[i * d + j] - max_s).exp();
            sum += e;
            acc += e * pool_kv[i * d + j];
        }
        out[j] = if sum > 0.0 { acc / sum } else { 0.0 };
    }
    // Shift: slots[:ratio] = previous current window (slots[ratio:])
    for i in 0..ratio {
        let dst = i * out_dim;
        let src = (ratio + i) * out_dim;
        if src + out_dim <= state.kv_state.len() {
            let kv_slice = state.kv_state[src..src + out_dim].to_vec();
            let sc_slice = state.score_state[src..src + out_dim].to_vec();
            state.kv_state[dst..dst + out_dim].copy_from_slice(&kv_slice);
            state.score_state[dst..dst + out_dim].copy_from_slice(&sc_slice);
        }
    }
    let mut ss = 0.0f32;
    for v in &out {
        ss += v * v;
    }
    let inv = (ss / d as f32 + 1e-6).sqrt().recip();
    for (j, v) in out.iter_mut().enumerate() {
        let g = comp.norm.get(j).copied().unwrap_or(1.0);
        *v = *v * inv * g;
    }
    rope.apply_slice(&mut out, pos, false);
    if comp.rotate {
        crate::weights::dtype::indexer_qk_qat_inplace(&mut out, 32);
    } else {
        let rd = rope.rope_dim.min(out.len());
        crate::weights::dtype::fp8_act_quant_nope_inplace(&mut out, rd, 64);
    }
    Some(out)
}

fn maybe_append_compress(
    kv: &Mutex<PagedKvPool>,
    compress_states: &Mutex<HashMap<String, CompressLayerState>>,
    index_kv: &Mutex<HashMap<String, (Vec<f32>, usize)>>,
    pfx: &str,
    block: &crate::weights::LayerBlock,
    x: &[f32],
    pos: usize,
    kv_width: usize,
) {
    if let Some(ref comp) = block.attn.compressor {
        let ckey = format!("{pfx}:C");
        let mut map = compress_states.lock();
        let st = map
            .entry(ckey.clone())
            .or_insert_with(|| CompressLayerState::new(comp));
        if let Some(mut ckv) = compressor_push(comp, st, x, pos, &block.attn.rope) {
            ckv.resize(kv_width, 0.0);
            let v = ckv.clone();
            kv.lock().append_kv(&ckey, &ckv, &v);
        }
    }
    // Parallel index-compressor stream (ratio-4 layers with indexer).
    if let Some(ref ix) = block.attn.indexer {
        let ikey = format!("{pfx}:I");
        let mut map = compress_states.lock();
        let st = map
            .entry(ikey.clone())
            .or_insert_with(|| CompressLayerState::new(&ix.compressor));
        if let Some(ckv) = compressor_push(&ix.compressor, st, x, pos, &ix.rope) {
            let mut store = index_kv.lock();
            let e = store.entry(ikey).or_insert_with(|| (Vec::new(), 0));
            e.0.extend_from_slice(&ckv);
            e.1 += 1;
        }
    }
}

fn materialize_compress_pool(
    kv: &Mutex<PagedKvPool>,
    pfx: &str,
) -> Option<(Vec<f32>, Vec<f32>, u32)> {
    let ckey = format!("{pfx}:C");
    let (k, v, n) = kv.lock().materialize_kv(&ckey);
    if n == 0 {
        None
    } else {
        Some((k, v, n))
    }
}

/// Score compress tokens with the learned indexer and keep top-k.
///
/// `ck/cv` are main compress latents (`head_dim` each). `ik` is the parallel
/// index-compress cache (`index_head_dim` each). Returns filtered `ck/cv`.
fn topk_compress_by_indexer(
    ix: &crate::weights::IndexerWeights,
    qr: &[f32],
    x: &[f32],
    pos: usize,
    ck: &[f32],
    cv: &[f32],
    cl: usize,
    main_dim: usize,
    ik: &[f32],
    il: usize,
) -> (Vec<f32>, Vec<f32>, u32) {
    let main_dim = main_dim.max(1);
    let id = ix.head_dim.max(1);
    let nh = ix.n_heads.max(1);
    let n = cl.min(il);
    if n == 0 {
        return (Vec::new(), Vec::new(), 0);
    }
    let k = ix.topk.min(n).max(1);
    // q: [H * D_idx] — RoPE then Hadamard + FP4 QAT (official Indexer.forward)
    let mut q = ix.wq_b.matvec(qr);
    q.resize(nh * id, 0.0);
    for h in 0..nh {
        let qh = &mut q[h * id..(h + 1) * id];
        ix.rope.apply_slice(qh, pos, false);
        crate::weights::dtype::indexer_qk_qat_inplace(qh, 32);
    }
    // head weights: weights_proj @ x  * scale
    let cols = ix.weights_proj.cols().max(1);
    let mut weights = vec![0.0f32; nh];
    let scale = (id as f32).sqrt().recip() * (nh as f32).sqrt().recip();
    for h in 0..nh.min(ix.weights_proj.rows()) {
        let r = &ix.weights_proj.data[h * cols..(h + 1) * cols];
        let mut s = 0.0f32;
        for (a, b) in r.iter().zip(x.iter()) {
            s += a * b;
        }
        weights[h] = s * scale;
    }
    let mut scores = vec![0.0f32; n];
    for t in 0..n {
        let k_base = t * id;
        if k_base + id > ik.len() {
            break;
        }
        let kt = &ik[k_base..k_base + id];
        let mut s = 0.0f32;
        for h in 0..nh {
            let qh = &q[h * id..(h + 1) * id];
            let mut dot = 0.0f32;
            for d in 0..id {
                dot += qh[d] * kt[d];
            }
            s += dot.max(0.0) * weights[h]; // relu
        }
        scores[t] = s;
    }
    // top-k indices
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    order.truncate(k);
    order.sort_unstable(); // restore temporal order for stability
    let mut ok = Vec::with_capacity(order.len() * main_dim);
    let mut ov = Vec::with_capacity(order.len() * main_dim);
    for &t in &order {
        let base = t * main_dim;
        if base + main_dim <= ck.len() && base + main_dim <= cv.len() {
            ok.extend_from_slice(&ck[base..base + main_dim]);
            ov.extend_from_slice(&cv[base..base + main_dim]);
        }
    }
    let n_out = (ok.len() / main_dim) as u32;
    (ok, ov, n_out)
}

fn filter_compress_pool_for_block(
    block: &crate::weights::LayerBlock,
    qr: Option<&[f32]>,
    x: Option<&[f32]>,
    pos: usize,
    pool: Option<(Vec<f32>, Vec<f32>, u32)>,
    index_kv: &Mutex<HashMap<String, (Vec<f32>, usize)>>,
    pfx: &str,
    main_dim: usize,
) -> Option<(Vec<f32>, Vec<f32>, u32)> {
    let Some((ck, cv, cl)) = pool else {
        return None;
    };
    if cl == 0 {
        return Some((ck, cv, cl));
    }
    let Some(ref ix) = block.attn.indexer else {
        return Some((ck, cv, cl));
    };
    let (Some(qr), Some(x)) = (qr, x) else {
        return Some((ck, cv, cl));
    };
    let ikey = format!("{pfx}:I");
    let guard = index_kv.lock();
    let Some((ref ik, il)) = guard.get(&ikey) else {
        return Some((ck, cv, cl));
    };
    if *il == 0 {
        return Some((ck, cv, cl));
    }
    Some(topk_compress_by_indexer(
        ix,
        qr,
        x,
        pos,
        &ck,
        &cv,
        cl as usize,
        main_dim,
        ik,
        *il,
    ))
}

fn matvec(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut s = 0.0;
        for (a, b) in row.iter().zip(x.iter()) {
            s += a * b;
        }
        y[i] = s;
    }
    y
}

/// SwiGLU: `w2( silu(w1 x) ⊙ w3 x )`.
#[allow(dead_code)] // kept for unit tests / f32-only fallbacks
fn swiglu_delta(
    x: &[f32],
    w1: &[f32],
    w2: &[f32],
    w3: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let u = matvec(w1, intermediate, hidden, x);
    let g = matvec(w3, intermediate, hidden, x);
    let mut gated = vec![0.0f32; intermediate];
    for i in 0..intermediate {
        let ui = u[i];
        let silu = ui / (1.0 + (-ui).exp());
        gated[i] = silu * g[i];
    }
    matvec(w2, hidden, intermediate, &gated)
}

/// Mean-pool multi-head Q `[n_heads * head_dim]` down to `attn_dim` (= head_dim for MQA KV).
/// Mean-pool multi-head Q → single attn vector (legacy / unit tests).
#[cfg_attr(not(test), allow(dead_code))]
fn pool_heads_to_attn(q_full: &[f32], n_heads: Option<usize>, attn_dim: usize) -> Vec<f32> {
    let n = q_full.len();
    if n == 0 {
        return vec![0.0; attn_dim];
    }
    // Prefer explicit head count when rows % heads == 0 and head_dim == attn_dim.
    if let Some(h) = n_heads {
        if h > 0 && n % h == 0 {
            let head_dim = n / h;
            if head_dim == attn_dim {
                let mut out = vec![0.0f32; attn_dim];
                for hi in 0..h {
                    let base = hi * head_dim;
                    for d in 0..head_dim {
                        out[d] += q_full[base + d];
                    }
                }
                let inv = 1.0 / h as f32;
                for x in &mut out {
                    *x *= inv;
                }
                return out;
            }
        }
    }
    // Fallback: average successive chunks of attn_dim.
    if n >= attn_dim && n % attn_dim == 0 {
        let chunks = n / attn_dim;
        let mut out = vec![0.0f32; attn_dim];
        for c in 0..chunks {
            let base = c * attn_dim;
            for d in 0..attn_dim {
                out[d] += q_full[base + d];
            }
        }
        let inv = 1.0 / chunks as f32;
        for x in &mut out {
            *x *= inv;
        }
        return out;
    }
    let mut out = q_full.to_vec();
    out.resize(attn_dim, 0.0);
    out
}

/// Pick FlashInfer when feature + prefer + import succeed; otherwise CPU ref.
fn select_kernel(prefer_flashinfer: bool) -> Arc<dyn KernelBackend> {
    #[cfg(feature = "flashinfer")]
    {
        if prefer_flashinfer {
            use crate::kernel::{FlashInferKernel, FlashInferKernelConfig};
            if let Some(k) = FlashInferKernel::try_new(FlashInferKernelConfig::default()) {
                return Arc::new(k);
            }
        }
    }
    #[cfg(not(feature = "flashinfer"))]
    {
        let _ = prefer_flashinfer;
    }
    Arc::new(CpuRefKernel)
}

#[derive(Debug, Clone)]
pub struct LocalWeightConfig {
    pub vocab_size: u32,
    pub num_heads: u32,
    pub head_dim: u32,
    pub page_tokens: usize,
    pub max_new_tokens_default: u32,
    /// HF model directory with config.json + sharded safetensors.
    pub model_dir: Option<std::path::PathBuf>,
    /// Prefer FlashInfer when the `flashinfer` cargo feature is enabled.
    /// Default true; set false to force CPU ref (tests / no GPU).
    pub prefer_flashinfer: bool,
}

impl Default for LocalWeightConfig {
    fn default() -> Self {
        Self {
            vocab_size: 512,
            num_heads: 4,
            head_dim: 32,
            page_tokens: 16,
            max_new_tokens_default: 32,
            model_dir: None,
            prefer_flashinfer: true,
        }
    }
}

/// In-process runner: physical KV + weights (toy or real safetensors) + KernelBackend.
pub struct LocalWeightRunner {
    kernel: Arc<dyn KernelBackend>,
    weights: ModelWeights,
    kv: Mutex<PagedKvPool>,
    cfg: LocalWeightConfig,
    /// Official HF tokenizer when `model_dir/tokenizer.json` is present.
    tokenizer: Option<HfTokenizer>,
    /// Multi-head Q / MQA path (when wq_b present).
    multihead: bool,
    /// Q heads used for KernelBackend (capped by TRAJECT_ATTN_HEADS).
    n_q_heads: usize,
    /// Per-head dim (= kv_lora when multihead).
    head_dim: usize,
    /// Sliding-window size for attention (0 = full context).
    sliding_window: usize,
    /// Incremental compressor state keyed by `{prefix}:L{i}:C` / `:I`.
    compress_states: Mutex<HashMap<String, CompressLayerState>>,
    /// Indexer compress latents keyed by `{prefix}:L{i}:I` (flat `[n * index_head_dim]`).
    index_kv: Mutex<HashMap<String, (Vec<f32>, usize)>>,
}

impl LocalWeightRunner {
    pub fn new(cfg: LocalWeightConfig) -> Self {
        let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
        let tokenizer = cfg.model_dir.as_ref().and_then(|dir| {
            match HfTokenizer::from_model_dir(dir) {
                Ok(t) => Some(t),
                Err(e) => {
                    warn!(error = %e, dir = %dir.display(), "HF tokenizer not loaded; using toy encode");
                    None
                }
            }
        });
        let weights = if let Some(dir) = &cfg.model_dir {
            match ModelWeights::from_safetensors(dir, cfg.num_heads, cfg.head_dim) {
                Ok(mut w) => {
                    // Prefer tokenizer EOS when weights config lacked it.
                    if w.eos_token_id.is_none() {
                        if let Some(t) = &tokenizer {
                            w.eos_token_id = t.eos_token_id();
                        }
                    }
                    w
                }
                Err(e) => {
                    tracing::warn!(error = %e, "safetensors load failed; falling back to toy weights");
                    ModelWeights::toy(cfg.vocab_size, attn_dim.max(64), attn_dim)
                }
            }
        } else {
            ModelWeights::toy(cfg.vocab_size, attn_dim, attn_dim)
        };
        // Multi-head MQA: store compressed KV (1 head × kv_lora); expand for Q heads at kernel time.
        let (multihead, n_q_heads, head_dim, pool_heads, pool_dim) =
            resolve_attn_layout(&weights, &cfg);
        let sliding_window = {
            use crate::weights::HfModelConfig;
            let hf = cfg.model_dir.as_ref().and_then(|d| HfModelConfig::load(d).ok());
            resolve_sliding_window(hf.as_ref())
        };
        let kernel = select_kernel(cfg.prefer_flashinfer);
        info!(
            kernel = kernel.name(),
            multihead,
            n_q_heads,
            head_dim,
            pool_heads,
            pool_dim,
            sliding_window,
            "LocalWeightRunner attention kernel"
        );
        Self {
            kernel,
            weights,
            kv: Mutex::new(PagedKvPool::new(cfg.page_tokens, pool_heads, pool_dim)),
            cfg,
            tokenizer,
            multihead,
            n_q_heads,
            head_dim,
            sliding_window,
            compress_states: Mutex::new(HashMap::new()),
            index_kv: Mutex::new(HashMap::new()),
        }
    }

    /// Load real embed/head/norm (+ tokenizer) from a HuggingFace model directory.
    pub fn from_model_dir(model_dir: impl Into<std::path::PathBuf>) -> Result<Self> {
        let model_dir = model_dir.into();
        let mut cfg = LocalWeightConfig::default();
        if let Ok(hc) = crate::weights::HfModelConfig::load(&model_dir) {
            cfg.vocab_size = hc.vocab_size;
            // Keep small attention dims for CPU path; full 64*512 is huge.
            cfg.num_heads = 8;
            cfg.head_dim = 64;
        }
        cfg.model_dir = Some(model_dir);
        Ok(Self::new(cfg))
    }

    pub fn with_kernel(cfg: LocalWeightConfig, kernel: Arc<dyn KernelBackend>) -> Self {
        let mut s = Self::new(LocalWeightConfig {
            prefer_flashinfer: false,
            ..cfg
        });
        s.kernel = kernel;
        info!(kernel = s.kernel.name(), "LocalWeightRunner using explicit kernel");
        s
    }

    pub fn weight_source(&self) -> &str {
        &self.weights.source
    }

    /// Whether an official HF tokenizer was loaded.
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Whether real layer-0 attention projections were loaded.
    pub fn has_layer0_attn(&self) -> bool {
        self.weights.has_layer0()
    }

    /// Whether layer-0 shared-expert FFN was loaded.
    pub fn has_layer0_ffn(&self) -> bool {
        self.weights.has_layer0_ffn()
    }

    /// Whether MLA Q expand (`wq_b`) was loaded.
    pub fn has_layer0_q_expand(&self) -> bool {
        self.weights
            .layers
            .first()
            .map(|l| l.attn.has_q_expand())
            .unwrap_or(false)
    }

    /// Whether real `wo_b` output projection was loaded.
    pub fn has_layer0_o_proj(&self) -> bool {
        self.weights
            .layers
            .first()
            .map(|l| l.attn.has_o_proj())
            .unwrap_or(false)
    }

    pub fn n_layers(&self) -> usize {
        self.weights.n_layers()
    }

    /// Whether routed MoE gate was loaded.
    pub fn has_layer0_moe(&self) -> bool {
        self.weights.has_layer0_moe()
    }

    /// Name of the active attention kernel (`cpu-ref` or `flashinfer-py`).
    pub fn kernel_name(&self) -> &str {
        self.kernel.name()
    }

    pub fn pages_allocated(&self) -> usize {
        self.kv.lock().pages_allocated()
    }

    fn tokenize(&self, text: &str) -> Vec<u32> {
        if let Some(tok) = &self.tokenizer {
            match tok.encode(text, false) {
                Ok(ids) if !ids.is_empty() => return ids,
                Ok(_) => {
                    warn!("tokenizer returned empty ids; falling back");
                }
                Err(e) => {
                    warn!(error = %e, "tokenizer encode failed; falling back to toy");
                }
            }
        }
        Self::toy_tokenize(text, self.weights.vocab.max(self.cfg.vocab_size))
    }

    fn detokenize(&self, ids: &[u32]) -> String {
        if let Some(tok) = &self.tokenizer {
            match tok.decode(ids, true) {
                Ok(s) => return s,
                Err(e) => {
                    warn!(error = %e, "tokenizer decode failed; printing ids");
                }
            }
        }
        format!(
            "[{}]",
            ids.iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
        )
    }

    fn toy_tokenize(text: &str, vocab: u32) -> Vec<u32> {
        if text.is_empty() {
            return vec![1];
        }
        text.chars()
            .map(|c| (c as u32) % vocab.max(2))
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
            self.tokenize(req.delta.text.as_deref().unwrap_or("?"))
        };

        {
            let mut kv = self.kv.lock();
            kv.bind_prefix(req.trajectory_id, prefix.clone());
        }

        let n_q = self.n_q_heads.max(1);
        let head_dim = self.head_dim.max(1);
        // Multi-head MQA stores compressed KV (1×head_dim); toy/legacy stores full H×D.
        let kv_width = if self.multihead {
            head_dim
        } else {
            n_q * head_dim
        };
        let q_width = n_q * head_dim;
        let n_layers = self.weights.n_layers().max(1);
        let rope_theta = self.weights.rope_theta;
        let rope_dim = self
            .weights
            .layers
            .first()
            .map(|b| b.attn.rope_head_dim)
            .unwrap_or(64)
            .min(head_dim.max(64));
        let toy_rope = crate::weights::RopeParams::base(rope_dim.max(2), rope_theta);

        let hc_mult = self
            .weights
            .layers
            .first()
            .and_then(|l| l.hc.as_ref().map(|h| h.hc_mult))
            .unwrap_or(1)
            .max(1);

        // Prefill prompt through the full layer stack (per-layer KV under prefix:L{i}).
        if req.decoded_so_far == 0 {
            for (pos, &tid) in prompt_ids.iter().enumerate() {
                let emb = self.weights.embed_token(tid);
                if self.weights.layers.is_empty() {
                    let (mut kk, mut vv) = self.weights.project_kv(&emb);
                    kk.resize(kv_width, 0.0);
                    vv.resize(kv_width, 0.0);
                    self.kv.lock().append_kv(&prefix, &kk, &vv);
                } else {
                    let mut streams = if self.weights.has_hc() {
                        expand_hc_streams(&emb, hc_mult)
                    } else {
                        emb
                    };
                    for (li, block) in self.weights.layers.iter().enumerate() {
                        let pfx = format!("{prefix}:L{li}");
                        if let Some(ref hc) = block.hc {
                            let residual = streams.clone();
                            let (x, post, comb) = hc_pre(
                                &streams,
                                &hc.attn,
                                hc.hc_mult,
                                hc.hidden,
                                hc.sinkhorn_iters,
                                hc.eps,
                                hc.norm_eps,
                            );
                            // Official: single attn_norm shared by Q/KV/compressor/indexer.
                            let xn = self.weights.rms_norm_with(&x, Some(&block.attn.attn_norm));
                            let (mut kk, _) = self.weights.project_kv_layer(&block.attn, &xn);
                            kk.resize(kv_width, 0.0);
                            finalize_attn_kv_latent(&mut kk, head_dim, &block.attn.rope, pos);
                            let vv = kk.clone();
                            self.kv.lock().append_kv(&pfx, &kk, &vv);
                            maybe_append_compress(
                                &self.kv,
                                &self.compress_states,
                                &self.index_kv,
                                &pfx,
                                block,
                                &xn,
                                pos,
                                kv_width,
                            );
                            let mut qq = self.weights.project_q_layer(&block.attn, &xn);
                            qq.resize(q_width, 0.0);
                            apply_rope_heads(&mut qq, n_q, head_dim, &block.attn.rope, pos, false);
                            let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                            let mut attn_out = vec![0.0f32; self.weights.hidden];
                            if seq_len > 0 {
                                let (k_exp, v_exp, seq_len) = prepare_kv_for_decode(
                                    self.multihead,
                                    k_cache,
                                    v_cache,
                                    seq_len,
                                    n_q,
                                    head_dim,
                                    self.sliding_window,
                                    block.attn.rope.compress_ratio,
                                    {
                                        let qr = self.weights.project_q_lora(&block.attn, &xn);
                                        filter_compress_pool_for_block(
                                            block,
                                            Some(&qr),
                                            Some(&xn),
                                            pos,
                                            materialize_compress_pool(&self.kv, &pfx),
                                            &self.index_kv,
                                            &pfx,
                                            head_dim,
                                        )
                                    },
                                );
                                if let Ok(dec) = self
                                    .kernel
                                    .decode(DecodeRequest {
                                        q: qq,
                                        k_cache: k_exp,
                                        v_cache: v_exp,
                                        seq_len,
                                        num_heads: n_q as u32,
                                        head_dim: head_dim as u32,
                                        layout: KvLayout::Nhd,
                                        attn_sink: sink_for_heads(&block.attn, n_q),
                                    })
                                    .await
                                {
                                    let mut attn_o = dec.o;
                                    attn_o.resize(q_width, 0.0);
                                    apply_rope_heads(&mut attn_o, n_q, head_dim, &block.attn.rope, pos, true);
                                    attn_out = self
                                        .weights
                                        .attn_to_hidden_layer(&block.attn, &attn_o, &x);
                                }
                            }
                            streams = hc_post(
                                &attn_out,
                                &residual,
                                &post,
                                &comb,
                                hc.hc_mult,
                                hc.hidden,
                            );
                            let residual = streams.clone();
                            let (x, post, comb) = hc_pre(
                                &streams,
                                &hc.ffn,
                                hc.hc_mult,
                                hc.hidden,
                                hc.sinkhorn_iters,
                                hc.eps,
                                hc.norm_eps,
                            );
                            let y = self.weights.moe_block_delta(
                                block.ffn.as_ref(),
                                block.moe.as_ref(),
                                &x,
                                Some(tid as u32),
                            );
                            streams =
                                hc_post(&y, &residual, &post, &comb, hc.hc_mult, hc.hidden);
                        } else {
                            // Simple residual path (no HC).
                            let mut h = if streams.len() == self.weights.hidden {
                                streams.clone()
                            } else {
                                mean_collapse_streams(&streams, hc_mult, self.weights.hidden)
                            };
                            // Official: single attn_norm shared by Q/KV/compressor/indexer.
                            let xn = self.weights.rms_norm_with(&h, Some(&block.attn.attn_norm));
                            let (mut kk, _) = self.weights.project_kv_layer(&block.attn, &xn);
                            kk.resize(kv_width, 0.0);
                            finalize_attn_kv_latent(&mut kk, head_dim, &block.attn.rope, pos);
                            let vv = kk.clone();
                            self.kv.lock().append_kv(&pfx, &kk, &vv);
                            maybe_append_compress(
                                &self.kv,
                                &self.compress_states,
                                &self.index_kv,
                                &pfx,
                                block,
                                &xn,
                                pos,
                                kv_width,
                            );
                            let mut qq = self.weights.project_q_layer(&block.attn, &xn);
                            qq.resize(q_width, 0.0);
                            apply_rope_heads(&mut qq, n_q, head_dim, &block.attn.rope, pos, false);
                            let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                            if seq_len > 0 {
                                let (k_exp, v_exp, seq_len) = prepare_kv_for_decode(
                                    self.multihead,
                                    k_cache,
                                    v_cache,
                                    seq_len,
                                    n_q,
                                    head_dim,
                                    self.sliding_window,
                                    block.attn.rope.compress_ratio,
                                    {
                                        let qr = self.weights.project_q_lora(&block.attn, &xn);
                                        filter_compress_pool_for_block(
                                            block,
                                            Some(&qr),
                                            Some(&xn),
                                            pos,
                                            materialize_compress_pool(&self.kv, &pfx),
                                            &self.index_kv,
                                            &pfx,
                                            head_dim,
                                        )
                                    },
                                );
                                if let Ok(dec) = self
                                    .kernel
                                    .decode(DecodeRequest {
                                        q: qq,
                                        k_cache: k_exp,
                                        v_cache: v_exp,
                                        seq_len,
                                        num_heads: n_q as u32,
                                        head_dim: head_dim as u32,
                                        layout: KvLayout::Nhd,
                                        attn_sink: sink_for_heads(&block.attn, n_q),
                                    })
                                    .await
                                {
                                    let mut attn_o = dec.o;
                                    attn_o.resize(q_width, 0.0);
                                    apply_rope_heads(&mut attn_o, n_q, head_dim, &block.attn.rope, pos, true);
                                    let delta = self
                                        .weights
                                        .attn_to_hidden_layer(&block.attn, &attn_o, &h);
                                    if block.attn.has_o_proj() {
                                        for (a, b) in h.iter_mut().zip(delta.iter()) {
                                            *a += *b;
                                        }
                                    } else {
                                        for (a, b) in h.iter_mut().zip(delta.iter()) {
                                            *a = 0.5 * *a + 0.5 * *b;
                                        }
                                    }
                                }
                            }
                            if let Some(ref ffn) = block.ffn {
                                h = self.weights.shared_ffn_residual_block(ffn, &h);
                            }
                            if let Some(ref moe) = block.moe {
                                let norm = block.ffn.as_ref().map(|f| f.ffn_norm.as_slice());
                                h = self.weights.routed_moe_residual_block(moe, norm, &h, Some(tid as u32));
                            }
                            streams = h;
                        }
                    }
                }
            }
        }

        // Decode tokens for this chunk (one new token at a time through stack).
        let budget = req
            .chunk_tokens
            .min(req.max_tokens.saturating_sub(req.decoded_so_far))
            .max(1)
            .min(8);
        let mut out_ids = Vec::new();
        let mut out_text = String::new();
        let eos = self.weights.eos_token_id;

        for _ in 0..budget {
            let last_tid = out_ids
                .last()
                .copied()
                .or_else(|| prompt_ids.last().copied())
                .unwrap_or(1);
            // Absolute position of `last_tid` in the sequence (0-based).
            // Prefill fills 0..prompt_len-1; generated tokens start at prompt_len.
            let abs_pos = if out_ids.is_empty() {
                prompt_ids.len().saturating_sub(1)
            } else {
                prompt_ids.len() + out_ids.len() - 1
            };
            let emb = self.weights.embed_token(last_tid);
            let mut streams = if self.weights.has_hc() && !self.weights.layers.is_empty() {
                expand_hc_streams(&emb, hc_mult)
            } else {
                emb
            };

            if self.weights.layers.is_empty() {
                let mut h = streams;
                let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
                if seq_len == 0 {
                    break;
                }
                let q_pos = (seq_len as usize).saturating_sub(1);
                let mut q = self.weights.project_q(&h);
                q.resize(q_width, 0.0);
                apply_rope_heads(&mut q, n_q, head_dim, &toy_rope, q_pos, false);
                let (k_exp, v_exp, seq_len) = prepare_kv_for_decode(
                    self.multihead,
                    k_cache,
                    v_cache,
                    seq_len,
                    n_q,
                    head_dim,
                    self.sliding_window,
                    0,
                    None,
                );
                let dec = self
                    .kernel
                    .decode(DecodeRequest {
                        q,
                        k_cache: k_exp,
                        v_cache: v_exp,
                        seq_len,
                        num_heads: n_q as u32,
                        head_dim: head_dim as u32,
                        layout: KvLayout::Nhd,
                        attn_sink: None,
                    })
                    .await
                    .map_err(|e| TrajectError::Inference(format!("local decode: {e}")))?;
                let mut attn_o = dec.o;
                attn_o.resize(q_width, 0.0);
                apply_rope_heads(&mut attn_o, n_q, head_dim, &toy_rope, q_pos, true);
                let attn_h = self.weights.attn_to_hidden(&attn_o, &h);
                for (a, b) in h.iter_mut().zip(attn_h.iter()) {
                    *a = 0.5 * *a + 0.5 * *b;
                }
                streams = h;
            } else {
                for (li, block) in self.weights.layers.iter().enumerate() {
                    let pfx = format!("{prefix}:L{li}");
                    if let Some(ref hc) = block.hc {
                        let residual = streams.clone();
                        let (x, post, comb) = hc_pre(
                            &streams,
                            &hc.attn,
                            hc.hc_mult,
                            hc.hidden,
                            hc.sinkhorn_iters,
                            hc.eps,
                            hc.norm_eps,
                        );
                        // Write K/V at abs_pos from full residual (append new or refresh).
                        // Official: single attn_norm for Q/KV/compressor/indexer.
                        let xn = self.weights.rms_norm_with(&x, Some(&block.attn.attn_norm));
                        let (mut kk, _) = self.weights.project_kv_layer(&block.attn, &xn);
                        kk.resize(kv_width, 0.0);
                        finalize_attn_kv_latent(&mut kk, head_dim, &block.attn.rope, abs_pos);
                        let vv = kk.clone();
                        let (seq_len, did_append) =
                            store_layer_kv_at_pos(&self.kv, &pfx, &kk, &vv, abs_pos);
                        if did_append {
                            maybe_append_compress(
                                &self.kv,
                                &self.compress_states,
                                &self.index_kv,
                                &pfx,
                                block,
                                &xn,
                                abs_pos,
                                kv_width,
                            );
                        }
                        if seq_len == 0 {
                            streams = residual;
                            continue;
                        }
                        let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                        let q_pos = abs_pos.min((seq_len as usize).saturating_sub(1));
                        let mut qq = self.weights.project_q_layer(&block.attn, &xn);
                        qq.resize(q_width, 0.0);
                        apply_rope_heads(&mut qq, n_q, head_dim, &block.attn.rope, q_pos, false);
                        let (k_exp, v_exp, seq_len) = prepare_kv_for_decode(
                            self.multihead,
                            k_cache,
                            v_cache,
                            seq_len,
                            n_q,
                            head_dim,
                            self.sliding_window,
                            block.attn.rope.compress_ratio,
                            {
                                let qr = self.weights.project_q_lora(&block.attn, &xn);
                                filter_compress_pool_for_block(
                                    block,
                                    Some(&qr),
                                    Some(&xn),
                                    q_pos,
                                    materialize_compress_pool(&self.kv, &pfx),
                                    &self.index_kv,
                                    &pfx,
                                    head_dim,
                                )
                            },
                        );
                        let dec = self
                            .kernel
                            .decode(DecodeRequest {
                                q: qq,
                                k_cache: k_exp,
                                v_cache: v_exp,
                                seq_len,
                                num_heads: n_q as u32,
                                head_dim: head_dim as u32,
                                layout: KvLayout::Nhd,
                                attn_sink: sink_for_heads(&block.attn, n_q),
                            })
                            .await
                            .map_err(|e| {
                                TrajectError::Inference(format!("local decode L{li}: {e}"))
                            })?;
                        let mut attn_o = dec.o;
                        attn_o.resize(q_width, 0.0);
                        apply_rope_heads(&mut attn_o, n_q, head_dim, &block.attn.rope, q_pos, true);
                        let attn_out =
                            self.weights.attn_to_hidden_layer(&block.attn, &attn_o, &x);
                        streams =
                            hc_post(&attn_out, &residual, &post, &comb, hc.hc_mult, hc.hidden);
                        let residual = streams.clone();
                        let (x, post, comb) = hc_pre(
                            &streams,
                            &hc.ffn,
                            hc.hc_mult,
                            hc.hidden,
                            hc.sinkhorn_iters,
                            hc.eps,
                            hc.norm_eps,
                        );
                        let y = self.weights.moe_block_delta(
                            block.ffn.as_ref(),
                            block.moe.as_ref(),
                            &x,
                            Some(last_tid as u32),
                        );
                        streams = hc_post(&y, &residual, &post, &comb, hc.hc_mult, hc.hidden);
                    } else {
                        let mut h = if streams.len() == self.weights.hidden {
                            streams.clone()
                        } else {
                            mean_collapse_streams(&streams, hc_mult, self.weights.hidden)
                        };
                        let xn = self.weights.rms_norm_with(&h, Some(&block.attn.attn_norm));
                        let (mut kk, _) = self.weights.project_kv_layer(&block.attn, &xn);
                        kk.resize(kv_width, 0.0);
                        finalize_attn_kv_latent(&mut kk, head_dim, &block.attn.rope, abs_pos);
                        let vv = kk.clone();
                        let (seq_len, did_append) =
                            store_layer_kv_at_pos(&self.kv, &pfx, &kk, &vv, abs_pos);
                        if did_append {
                            maybe_append_compress(
                                &self.kv,
                                &self.compress_states,
                                &self.index_kv,
                                &pfx,
                                block,
                                &xn,
                                abs_pos,
                                kv_width,
                            );
                        }
                        if seq_len == 0 {
                            streams = h;
                            continue;
                        }
                        let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                        let q_pos = abs_pos.min((seq_len as usize).saturating_sub(1));
                        let mut qq = self.weights.project_q_layer(&block.attn, &xn);
                        qq.resize(q_width, 0.0);
                        apply_rope_heads(&mut qq, n_q, head_dim, &block.attn.rope, q_pos, false);
                        let (k_exp, v_exp, seq_len) = prepare_kv_for_decode(
                            self.multihead,
                            k_cache,
                            v_cache,
                            seq_len,
                            n_q,
                            head_dim,
                            self.sliding_window,
                            block.attn.rope.compress_ratio,
                            {
                                let qr = self.weights.project_q_lora(&block.attn, &xn);
                                filter_compress_pool_for_block(
                                    block,
                                    Some(&qr),
                                    Some(&xn),
                                    q_pos,
                                    materialize_compress_pool(&self.kv, &pfx),
                                    &self.index_kv,
                                    &pfx,
                                    head_dim,
                                )
                            },
                        );
                        let dec = self
                            .kernel
                            .decode(DecodeRequest {
                                q: qq,
                                k_cache: k_exp,
                                v_cache: v_exp,
                                seq_len,
                                num_heads: n_q as u32,
                                head_dim: head_dim as u32,
                                layout: KvLayout::Nhd,
                                attn_sink: sink_for_heads(&block.attn, n_q),
                            })
                            .await
                            .map_err(|e| {
                                TrajectError::Inference(format!("local decode L{li}: {e}"))
                            })?;
                        let mut attn_o = dec.o;
                        attn_o.resize(q_width, 0.0);
                        apply_rope_heads(&mut attn_o, n_q, head_dim, &block.attn.rope, q_pos, true);
                        let delta =
                            self.weights.attn_to_hidden_layer(&block.attn, &attn_o, &h);
                        if block.attn.has_o_proj() {
                            for (a, b) in h.iter_mut().zip(delta.iter()) {
                                *a += *b;
                            }
                        } else {
                            for (a, b) in h.iter_mut().zip(delta.iter()) {
                                *a = 0.5 * *a + 0.5 * *b;
                            }
                        }
                        if let Some(ref ffn) = block.ffn {
                            h = self.weights.shared_ffn_residual_block(ffn, &h);
                        }
                        if let Some(ref moe) = block.moe {
                            let norm = block.ffn.as_ref().map(|f| f.ffn_norm.as_slice());
                            h = self.weights.routed_moe_residual_block(moe, norm, &h, Some(last_tid as u32));
                        }
                        streams = h;
                    }
                }
            }

            let h = if streams.len() == self.weights.hidden {
                streams
            } else if let Some(ref hh) = self.weights.hc_head {
                hc_head_collapse(&streams, hh)
            } else {
                mean_collapse_streams(&streams, hc_mult, self.weights.hidden)
            };
            let logits = self.weights.logits(&h);
            let sampled = self
                .kernel
                .sample(SampleRequest {
                    logits,
                    temperature: req.constraints.temperature.unwrap_or(0.0),
                    top_p: req.constraints.top_p.unwrap_or(1.0),
                })
                .await?;

            let tid = sampled.token_id % self.weights.vocab;
            // Next loop iteration processes `tid` at abs_pos=prompt_len+out_ids.len()
            // and appends its K/V from the full residual (no separate post-sample path).
            let _ = (n_layers, abs_pos); // used above / in logs

            out_ids.push(tid);
            if self.tokenizer.is_some() {
                // Decode full sequence at end for correct BPE boundaries.
            } else if !self.weights.source.starts_with("safetensors") {
                out_text.push_str(&self.detokenize(&[tid]));
            }

            if eos == Some(tid) {
                break;
            }
            if self.tokenizer.is_none()
                && !self.weights.source.starts_with("safetensors")
                && tid % 97 == 0
            {
                break;
            }
        }

        if self.tokenizer.is_some() || self.weights.source.starts_with("safetensors") {
            out_text = self.detokenize(&out_ids);
        }

        let produced = out_ids.len() as u32;
        let finished = req.decoded_so_far + produced >= req.max_tokens
            || produced < budget
            || req.decoded_so_far + produced >= req.chunk_tokens
            || out_ids.last().copied() == eos;

        info!(
            trajectory = %req.trajectory_id,
            prefix = %prefix,
            produced,
            pages = self.pages_allocated(),
            source = %self.weights.source,
            vocab = self.weights.vocab,
            has_tokenizer = self.tokenizer.is_some(),
            has_layer0 = self.weights.has_layer0(),
            has_layer0_ffn = self.weights.has_layer0_ffn(),
            has_q_expand = self
                .weights
                .layers
                .first()
                .map(|l| l.attn.has_q_expand())
                .unwrap_or(false),
            has_o_proj = self
                .weights
                .layers
                .first()
                .map(|l| l.attn.has_o_proj())
                .unwrap_or(false),
            has_moe = self.weights.has_layer0_moe(),
            n_layers = self.weights.n_layers(),
            multihead = self.multihead,
            n_q_heads = self.n_q_heads,
            head_dim = self.head_dim,
            rope_dim = rope_dim,
            has_attn_sink = self
                .weights
                .layers
                .first()
                .and_then(|l| l.attn.attn_sink.as_ref())
                .is_some(),
            has_hc = self.weights.has_hc(),
            has_hc_head = self.weights.hc_head.is_some(),
            hc_mult = hc_mult,
            sliding_window = self.sliding_window,
            has_compressor = self
                .weights
                .layers
                .iter()
                .any(|l| l.attn.compressor.is_some()),
            has_indexer = self
                .weights
                .layers
                .iter()
                .any(|l| l.attn.indexer.is_some()),
            moe_cache = ?self.weights.layers.first().and_then(|l| {
                l.moe.as_ref().map(|m| m.cache_stats())
            }),
            kernel = self.kernel.name(),
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
        let mut total = 0;
        total += self.kv.lock().free_prefix(prefix_id);
        // Per-layer KV keys used by multi-layer stack (+ compress pools).
        for li in 0..self.weights.n_layers().max(1) {
            let pfx = format!("{prefix_id}:L{li}");
            total += self.kv.lock().free_prefix(&pfx);
            let cpfx = format!("{pfx}:C");
            total += self.kv.lock().free_prefix(&cpfx);
            self.compress_states.lock().remove(&cpfx);
            let ipfx = format!("{pfx}:I");
            self.compress_states.lock().remove(&ipfx);
            self.index_kv.lock().remove(&ipfx);
        }
        info!(%prefix_id, pages_zeroed = total, "local runner physical KV free");
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
        // Force CPU so CI / macOS unit tests never need CUDA/Python FlashInfer.
        let runner = LocalWeightRunner::new(LocalWeightConfig {
            prefer_flashinfer: false,
            ..LocalWeightConfig::default()
        });
        assert_eq!(runner.kernel_name(), "cpu-ref");
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

    #[test]
    fn overwrite_last_kv_replaces_tail() {
        let mut pool = PagedKvPool::new(8, 1, 2);
        let k1 = vec![1.0f32, 2.0];
        let v1 = vec![3.0f32, 4.0];
        pool.append_kv("p", &k1, &v1);
        let k2 = vec![5.0f32, 6.0];
        let v2 = vec![7.0f32, 8.0];
        pool.append_kv("p", &k2, &v2);
        assert!(pool.overwrite_last_kv("p", &[9.0, 10.0], &[11.0, 12.0]));
        let (k, v, n) = pool.materialize_kv("p");
        assert_eq!(n, 2);
        assert_eq!(k, vec![1.0, 2.0, 9.0, 10.0]);
        assert_eq!(v, vec![3.0, 4.0, 11.0, 12.0]);
    }

    #[test]
    fn store_layer_kv_at_pos_append_then_overwrite() {
        let pool = Mutex::new(PagedKvPool::new(8, 1, 2));
        let k0 = vec![1.0f32, 0.0];
        let v0 = vec![0.0f32, 1.0];
        // abs_pos=0, empty cache → append
        let (n, app) = store_layer_kv_at_pos(&pool, "p", &k0, &v0, 0);
        assert_eq!(n, 1);
        assert!(app);
        // abs_pos=0 again with len=1 → overwrite last
        let k0b = vec![2.0f32, 2.0];
        let (n, app) = store_layer_kv_at_pos(&pool, "p", &k0b, &k0b, 0);
        assert_eq!(n, 1);
        assert!(!app);
        // abs_pos=1 → append second token
        let k1 = vec![3.0f32, 3.0];
        let (n, app) = store_layer_kv_at_pos(&pool, "p", &k1, &k1, 1);
        assert_eq!(n, 2);
        assert!(app);
        let (k, _, n) = pool.lock().materialize_kv("p");
        assert_eq!(n, 2);
        assert_eq!(k, vec![2.0, 2.0, 3.0, 3.0]);
    }

    #[test]
    fn pool_heads_mean() {
        // 2 heads × 3 dims: head0=[1,2,3], head1=[3,4,5] → mean=[2,3,4]
        let q = vec![1.0, 2.0, 3.0, 3.0, 4.0, 5.0];
        let out = pool_heads_to_attn(&q, Some(2), 3);
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.0).abs() < 1e-5);
        assert!((out[1] - 3.0).abs() < 1e-5);
        assert!((out[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn expand_kv_mqa_repeats_heads() {
        // seq=2, head_dim=2, n_q=3
        let kv = vec![1.0, 2.0, 3.0, 4.0];
        let out = expand_kv_mqa(&kv, 2, 3, 2);
        assert_eq!(out.len(), 2 * 3 * 2);
        // token0 heads all [1,2]
        assert_eq!(&out[0..2], &[1.0, 2.0]);
        assert_eq!(&out[2..4], &[1.0, 2.0]);
        assert_eq!(&out[4..6], &[1.0, 2.0]);
        // token1 heads all [3,4]
        assert_eq!(&out[6..8], &[3.0, 4.0]);
    }

    #[test]
    fn rope_roundtrip_near_identity() {
        // Apply RoPE then inverse → recover original (within float noise).
        let rope = crate::weights::RopeParams::base(4, 10000.0);
        let mut x = vec![0.5f32, -0.25, 0.1, 0.75];
        let orig = x.clone();
        rope.apply_slice(&mut x, 7, false);
        assert!((x[0] - orig[0]).abs() > 1e-6 || (x[1] - orig[1]).abs() > 1e-6);
        rope.apply_slice(&mut x, 7, true);
        for (a, b) in x.iter().zip(orig.iter()) {
            assert!((a - b).abs() < 1e-5, "{a} vs {b}");
        }
    }

    #[test]
    fn rope_only_last_dims() {
        let rope = crate::weights::RopeParams::base(2, 10000.0);
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, -0.5];
        let head = x.clone();
        rope.apply_slice(&mut x, 3, false);
        // first 4 dims (nope) unchanged; last 2 rotated
        assert_eq!(&x[..4], &head[..4]);
        assert!((x[4] - head[4]).abs() > 1e-6 || (x[5] - head[5]).abs() > 1e-6);
    }

    #[test]
    fn yarn_inv_freq_differs_from_base() {
        let base = crate::weights::RopeParams::base(64, 160000.0);
        let yarn = crate::weights::RopeParams::yarn_or_base(
            64, 160000.0, 4, 65536, 16.0, 32.0, 1.0,
        );
        assert!(yarn.yarn);
        assert!(!base.yarn);
        // At least one frequency should change under YaRN.
        let changed = base
            .inv_freq
            .iter()
            .zip(yarn.inv_freq.iter())
            .any(|(a, b)| (a - b).abs() > 1e-12);
        assert!(changed);
    }

    #[test]
    fn hc_sinkhorn_shapes_and_positive() {
        let hc = 2usize;
        let mix_hc = (2 + hc) * hc; // 8
        let mixes = vec![0.1f32; mix_hc];
        let scale = vec![1.0f32, 1.0, 1.0];
        let base = vec![0.0f32; mix_hc];
        let (pre, post, comb) = hc_split_sinkhorn(&mixes, &scale, &base, hc, 4, 1e-6);
        assert_eq!(pre.len(), hc);
        assert_eq!(post.len(), hc);
        assert_eq!(comb.len(), hc * hc);
        for &p in &pre {
            assert!(p > 0.0);
        }
        // col sums after sinkhorn should be ~1
        for j in 0..hc {
            let mut s = 0.0f32;
            for i in 0..hc {
                s += comb[i * hc + j];
            }
            assert!((s - 1.0).abs() < 0.05, "col {j} sum {s}");
        }
    }

    #[test]
    fn expand_hc_repeats() {
        let h = vec![1.0f32, 2.0];
        let s = expand_hc_streams(&h, 3);
        assert_eq!(s, vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0]);
    }

    #[test]
    fn slice_kv_window_keeps_tail() {
        // seq=4, width=2 → last 2 tokens
        let k: Vec<f32> = (0..8).map(|x| x as f32).collect();
        let v: Vec<f32> = (10..18).map(|x| x as f32).collect();
        let (k2, v2, sl) = slice_kv_window(k, v, 4, 2, 2);
        assert_eq!(sl, 2);
        assert_eq!(k2, vec![4.0, 5.0, 6.0, 7.0]);
        assert_eq!(v2, vec![14.0, 15.0, 16.0, 17.0]);
    }

    #[test]
    fn slice_kv_window_noop_when_short() {
        let k = vec![1.0f32, 2.0];
        let v = vec![3.0f32, 4.0];
        let (k2, v2, sl) = slice_kv_window(k.clone(), v.clone(), 1, 2, 128);
        assert_eq!(sl, 1);
        assert_eq!(k2, k);
        assert_eq!(v2, v);
    }

    #[test]
    fn sparse_indices_window_and_stride() {
        // seq=20, window=4, ratio=4 → history ends 3,7,11,15 then window 16..19
        let idxs = build_sparse_kv_indices(20, 4, 4, 512);
        assert_eq!(idxs, vec![3, 7, 11, 15, 16, 17, 18, 19]);
    }

    #[test]
    fn sparse_indices_swa_only_when_ratio_one() {
        let idxs = build_sparse_kv_indices(10, 4, 0, 512);
        assert_eq!(idxs, vec![6, 7, 8, 9]);
        let idxs = build_sparse_kv_indices(10, 4, 1, 512);
        assert_eq!(idxs, vec![6, 7, 8, 9]);
    }

    #[test]
    fn gather_kv_picks_tokens() {
        // seq=3, width=2: tokens [0,1],[2,3],[4,5]
        let k: Vec<f32> = (0..6).map(|x| x as f32).collect();
        let v: Vec<f32> = (10..16).map(|x| x as f32).collect();
        let (k2, v2, n) = gather_kv_by_idx(&k, &v, 3, 2, &[0, 2]);
        assert_eq!(n, 2);
        assert_eq!(k2, vec![0.0, 1.0, 4.0, 5.0]);
        assert_eq!(v2, vec![10.0, 11.0, 14.0, 15.0]);
    }

    #[test]
    fn indexer_topk_picks_highest_score() {
        use crate::weights::{CompressorWeights, IndexerWeights, LinearMat, RopeParams, TensorF32};
        // 1 head, dim 2, topk 1, two compress tokens.
        // Index keys are pre-rotated/QAT'd as the rotate=true compressor would emit.
        let hidden = 2usize;
        let id = 2usize;
        let wq = TensorF32 {
            data: vec![1.0, 0.0, 0.0, 1.0], // identity 2x2
            shape: vec![2, 2],
        };
        let wp = TensorF32 {
            data: vec![1.0, 0.0],
            shape: vec![1, 2],
        };
        let dummy_comp = CompressorWeights {
            ratio: 4,
            head_dim: id,
            hidden,
            overlap: false,
            rotate: true,
            ape: vec![0.0; 4 * id],
            wkv: LinearMat::F32(wq.clone()),
            wgate: LinearMat::F32(wq.clone()),
            norm: vec![1.0; id],
        };
        let ix = IndexerWeights {
            n_heads: 1,
            head_dim: id,
            topk: 1,
            q_lora: 2,
            wq_b: LinearMat::F32(wq),
            weights_proj: wp,
            compressor: dummy_comp,
            rope: RopeParams::base(2, 10000.0),
        };
        // main compress tokens (unrotated main pool): t0=[1,0], t1=[0,1]
        let ck = vec![1.0, 0.0, 0.0, 1.0];
        let cv = ck.clone();
        // index stream: match q-side QAT of [1,0] vs its negation
        let mut t0 = vec![1.0f32, 0.0];
        crate::weights::dtype::indexer_qk_qat_inplace(&mut t0, 32);
        let mut t1 = vec![-t0[0], -t0[1]];
        let mut ik = t0;
        ik.append(&mut t1);
        let qr = vec![1.0, 0.0];
        let x = vec![1.0, 0.0];
        let (ok, _ov, n) =
            topk_compress_by_indexer(&ix, &qr, &x, 0, &ck, &cv, 2, 2, &ik, 2);
        assert_eq!(n, 1);
        assert_eq!(ok, vec![1.0, 0.0]);
    }

    #[test]
    fn compressor_rotate_applies_qat() {
        use crate::weights::{CompressorWeights, LinearMat, RopeParams, TensorF32};
        let hidden = 4usize;
        let head_dim = 4usize; // power of two for Hadamard
        let ratio = 2usize;
        let w = TensorF32 {
            data: {
                let mut m = vec![0.0f32; head_dim * hidden];
                for i in 0..head_dim {
                    m[i * hidden + i] = 1.0;
                }
                m
            },
            shape: vec![head_dim, hidden],
        };
        let comp = CompressorWeights {
            ratio,
            head_dim,
            hidden,
            overlap: false,
            rotate: true,
            ape: vec![0.0; ratio * head_dim],
            wkv: LinearMat::F32(w.clone()),
            wgate: LinearMat::F32(w),
            norm: vec![1.0; head_dim],
        };
        let rope = RopeParams::base(2, 10000.0);
        let mut st = CompressLayerState::new(&comp);
        let x = vec![1.0f32, 0.5, 0.25, 0.125];
        assert!(compressor_push(&comp, &mut st, &x, 0, &rope).is_none());
        let out = compressor_push(&comp, &mut st, &x, 1, &rope).expect("emit");
        assert_eq!(out.len(), head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
        // QAT projects onto the e2m1×scale lattice (not identity passthrough).
        let mut ref_x = out.clone();
        // already QAT'd once; second pass should be nearly idempotent
        crate::weights::dtype::fp4_act_quant_inplace(&mut ref_x, 32);
        for (a, b) in out.iter().zip(ref_x.iter()) {
            assert!((a - b).abs() < 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn compressor_nonoverlap_emits_every_ratio() {
        use crate::weights::{CompressorWeights, LinearMat, RopeParams, TensorF32};
        let hidden = 4usize;
        let head_dim = 2usize;
        let ratio = 2usize;
        // Identity-ish dense mats: out = first head_dim of padded hidden.
        let w = TensorF32 {
            data: {
                let mut m = vec![0.0f32; head_dim * hidden];
                for i in 0..head_dim {
                    m[i * hidden + i] = 1.0;
                }
                m
            },
            shape: vec![head_dim, hidden],
        };
        let comp = CompressorWeights {
            ratio,
            head_dim,
            hidden,
            overlap: false,
            rotate: false,
            ape: vec![0.0; ratio * head_dim],
            wkv: LinearMat::F32(w.clone()),
            wgate: LinearMat::F32(w),
            norm: vec![1.0; head_dim],
        };
        let rope = RopeParams::base(2, 10000.0);
        let mut st = CompressLayerState::new(&comp);
        let x = vec![1.0f32, 2.0, 0.0, 0.0];
        assert!(compressor_push(&comp, &mut st, &x, 0, &rope).is_none());
        let out = compressor_push(&comp, &mut st, &x, 1, &rope).expect("emit");
        assert_eq!(out.len(), head_dim);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
