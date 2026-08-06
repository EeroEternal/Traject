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
        }
    }

    fn from_safetensors(
        model_dir: &std::path::Path,
        attn_heads: u32,
        attn_dim_per_head: u32,
    ) -> Result<Self> {
        use crate::weights::{load_embed_head_norm, load_layer_stack, HfModelConfig};

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

    /// Hidden → Q vector for a specific layer (or toy down if no layers).
    fn project_q_layer(&self, layer: &crate::weights::Layer0AttnWeights, h: &[f32]) -> Vec<f32> {
        let hn = self.rms_norm_with(h, Some(&layer.attn_norm));
        let mut q = matvec(&layer.wq_a.data, layer.q_lora_dim(), layer.hidden, &hn);
        if let Some(ref qn) = layer.q_norm {
            q = self.rms_norm_with(&q, Some(qn));
        }
        if let Some(ref wq_b) = layer.wq_b {
            let q_full = matvec(&wq_b.data, wq_b.rows(), wq_b.cols(), &q);
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
            return out;
        }
        let mut out = q;
        out.resize(self.attn_dim, 0.0);
        out
    }

    fn project_kv_layer(
        &self,
        layer: &crate::weights::Layer0AttnWeights,
        h: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let hn = self.rms_norm_with(h, Some(&layer.attn_norm));
        let mut kv = matvec(&layer.wkv.data, layer.kv_dim(), layer.hidden, &hn);
        if let Some(ref kn) = layer.kv_norm {
            kv = self.rms_norm_with(&kv, Some(kn));
        }
        kv.resize(self.attn_dim, 0.0);
        let v = kv.iter().map(|x| x * 0.5).collect();
        (kv, v)
    }

    fn project_q(&self, h: &[f32]) -> Vec<f32> {
        if let Some(block) = self.layers.first() {
            return self.project_q_layer(&block.attn, h);
        }
        matvec(&self.w_down, self.attn_dim, self.hidden, h)
    }

    fn project_kv(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if let Some(block) = self.layers.first() {
            return self.project_kv_layer(&block.attn, h);
        }
        let a = matvec(&self.w_down, self.attn_dim, self.hidden, h);
        let v = a.iter().map(|x| x * 0.5).collect();
        (a, v)
    }

    fn attn_to_hidden_layer(
        &self,
        layer: &crate::weights::Layer0AttnWeights,
        attn_o: &[f32],
        h_resid: &[f32],
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
            if n_heads > 1 && attn_o.len() >= n_heads * head_dim {
                // Multi-head o: [H, D] → group-mean → o_lora slots.
                let hpg = (n_heads / g).max(1);
                for gi in 0..g {
                    let mut pooled = vec![0.0f32; head_dim];
                    let mut count = 0usize;
                    for hi in 0..hpg {
                        let h = gi * hpg + hi;
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
                // Single-vector o (legacy / toy): inject into each group.
                let take = attn_o.len().min(lor);
                let scale = 1.0 / (g as f32).sqrt();
                for gi in 0..g {
                    let base = gi * lor;
                    for d in 0..take {
                        mid[base + d] += attn_o[d] * scale;
                    }
                }
            }
            if let Some(ref wo_a) = layer.wo_a {
                if wo_a.rows() == inter && wo_a.cols() == layer.hidden && h_resid.len() == layer.hidden
                {
                    let add = matvec(&wo_a.data, inter, layer.hidden, h_resid);
                    for (m, a) in mid.iter_mut().zip(add.iter()) {
                        *m += *a;
                    }
                }
            }
            return matvec(&wo_b.data, layer.hidden, inter, &mid);
        }
        matvec(&self.w_up, self.hidden, self.attn_dim, attn_o)
    }

    fn attn_to_hidden(&self, attn_o: &[f32], h_resid: &[f32]) -> Vec<f32> {
        if let Some(block) = self.layers.first() {
            return self.attn_to_hidden_layer(&block.attn, attn_o, h_resid);
        }
        matvec(&self.w_up, self.hidden, self.attn_dim, attn_o)
    }

    fn shared_ffn_residual_block(
        &self,
        ffn: &crate::weights::Layer0SharedFfn,
        h: &[f32],
    ) -> Vec<f32> {
        let n = self.rms_norm_with(h, Some(&ffn.ffn_norm));
        let delta = swiglu_delta(
            &n,
            &ffn.w1.data,
            &ffn.w2.data,
            &ffn.w3.data,
            ffn.hidden,
            ffn.intermediate,
        );
        let mut out = h.to_vec();
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += *d;
        }
        out
    }

    fn routed_moe_residual_block(
        &self,
        moe: &crate::weights::Layer0RoutedMoe,
        ffn_norm: Option<&[f32]>,
        h: &[f32],
    ) -> Vec<f32> {
        let n = if let Some(g) = ffn_norm {
            self.rms_norm_with(h, Some(g))
        } else {
            h.to_vec()
        };
        let routes = moe.route(&n);
        let mut delta = vec![0.0f32; moe.hidden];
        for (eid, w) in routes {
            match moe.expert(eid).and_then(|ex| ex.swiglu(&n)) {
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
        let scale = moe.route_scale;
        let mut out = h.to_vec();
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += scale * *d;
        }
        out
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
        let h = self.rms_norm(h);
        let v = self.vocab as usize;
        let mut out = vec![0.0f32; v];
        // head is [vocab, hidden] — logits[i] = dot(head[i], h)
        for i in 0..v {
            let row = &self.head[i * self.hidden..(i + 1) * self.hidden];
            let mut s = 0.0;
            for (a, b) in h.iter().zip(row.iter()) {
                s += a * b;
            }
            out[i] = s;
        }
        out
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


/// Q heads for KernelBackend (`TRAJECT_ATTN_HEADS`, default 8, max model heads).
fn attn_heads_to_use(model_heads: usize) -> usize {
    let env = std::env::var("TRAJECT_ATTN_HEADS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let want = env.unwrap_or(8).max(1);
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

/// How many transformer layers to load for the local runner.
///
/// `TRAJECT_LOCAL_LAYERS` env (default 2, min 1, max 8 for memory safety).
fn local_layer_count(cfg: Option<&crate::weights::HfModelConfig>) -> usize {
    let env = std::env::var("TRAJECT_LOCAL_LAYERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok());
    let n = env.unwrap_or(2).max(1);
    let max_model = cfg
        .and_then(|c| c.num_hidden_layers)
        .map(|x| x as usize)
        .unwrap_or(43);
    n.min(max_model).min(8)
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
        let kernel = select_kernel(cfg.prefer_flashinfer);
        info!(
            kernel = kernel.name(),
            multihead,
            n_q_heads,
            head_dim,
            pool_heads,
            pool_dim,
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

        // Prefill prompt through the full layer stack (per-layer KV under prefix:L{i}).
        if req.decoded_so_far == 0 {
            for &tid in &prompt_ids {
                let mut h = self.weights.embed_token(tid);
                if self.weights.layers.is_empty() {
                    let (mut kk, mut vv) = self.weights.project_kv(&h);
                    kk.resize(kv_width, 0.0);
                    vv.resize(kv_width, 0.0);
                    self.kv.lock().append_kv(&prefix, &kk, &vv);
                } else {
                    for (li, block) in self.weights.layers.iter().enumerate() {
                        let pfx = format!("{prefix}:L{li}");
                        let (mut kk, mut vv) = self.weights.project_kv_layer(&block.attn, &h);
                        kk.resize(kv_width, 0.0);
                        vv.resize(kv_width, 0.0);
                        self.kv.lock().append_kv(&pfx, &kk, &vv);
                        let mut qq = self.weights.project_q_layer(&block.attn, &h);
                        qq.resize(q_width, 0.0);
                        let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                        if seq_len > 0 {
                            let k_exp = if self.multihead {
                                expand_kv_mqa(&k_cache, seq_len as usize, n_q, head_dim)
                            } else {
                                k_cache
                            };
                            let v_exp = if self.multihead {
                                expand_kv_mqa(&v_cache, seq_len as usize, n_q, head_dim)
                            } else {
                                v_cache
                            };
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
                                })
                                .await
                            {
                                let mut attn_o = dec.o;
                                attn_o.resize(q_width, 0.0);
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
                            }
                        }
                        if let Some(ref ffn) = block.ffn {
                            h = self.weights.shared_ffn_residual_block(ffn, &h);
                        }
                        if let Some(ref moe) = block.moe {
                            let norm = block.ffn.as_ref().map(|f| f.ffn_norm.as_slice());
                            h = self.weights.routed_moe_residual_block(moe, norm, &h);
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
            let mut h = self.weights.embed_token(last_tid);

            if self.weights.layers.is_empty() {
                let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
                if seq_len == 0 {
                    break;
                }
                let mut q = self.weights.project_q(&h);
                q.resize(q_width, 0.0);
                let k_exp = if self.multihead {
                    expand_kv_mqa(&k_cache, seq_len as usize, n_q, head_dim)
                } else {
                    k_cache
                };
                let v_exp = if self.multihead {
                    expand_kv_mqa(&v_cache, seq_len as usize, n_q, head_dim)
                } else {
                    v_cache
                };
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
                    })
                    .await
                    .map_err(|e| TrajectError::Inference(format!("local decode: {e}")))?;
                let mut attn_o = dec.o;
                attn_o.resize(q_width, 0.0);
                let attn_h = self.weights.attn_to_hidden(&attn_o, &h);
                for (a, b) in h.iter_mut().zip(attn_h.iter()) {
                    *a = 0.5 * *a + 0.5 * *b;
                }
            } else {
                for (li, block) in self.weights.layers.iter().enumerate() {
                    let pfx = format!("{prefix}:L{li}");
                    let mut qq = self.weights.project_q_layer(&block.attn, &h);
                    qq.resize(q_width, 0.0);
                    let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&pfx);
                    if seq_len == 0 {
                        let (mut kk, mut vv) = self.weights.project_kv_layer(&block.attn, &h);
                        kk.resize(kv_width, 0.0);
                        vv.resize(kv_width, 0.0);
                        self.kv.lock().append_kv(&pfx, &kk, &vv);
                        continue;
                    }
                    let k_exp = if self.multihead {
                        expand_kv_mqa(&k_cache, seq_len as usize, n_q, head_dim)
                    } else {
                        k_cache
                    };
                    let v_exp = if self.multihead {
                        expand_kv_mqa(&v_cache, seq_len as usize, n_q, head_dim)
                    } else {
                        v_cache
                    };
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
                        })
                        .await
                        .map_err(|e| TrajectError::Inference(format!("local decode L{li}: {e}")))?;
                    let mut attn_o = dec.o;
                    attn_o.resize(q_width, 0.0);
                    let delta = self.weights.attn_to_hidden_layer(&block.attn, &attn_o, &h);
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
                        h = self.weights.routed_moe_residual_block(moe, norm, &h);
                    }
                }
            }

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
            // Append new-token KV for every layer after sampling (for next step).
            let emb_n = self.weights.embed_token(tid);
            if self.weights.layers.is_empty() {
                let (mut k, mut v) = self.weights.project_kv(&emb_n);
                k.resize(kv_width, 0.0);
                v.resize(kv_width, 0.0);
                self.kv.lock().append_kv(&prefix, &k, &v);
            } else {
                let mut hh = emb_n;
                for (li, block) in self.weights.layers.iter().enumerate() {
                    let pfx = format!("{prefix}:L{li}");
                    let (mut k, mut v) = self.weights.project_kv_layer(&block.attn, &hh);
                    k.resize(kv_width, 0.0);
                    v.resize(kv_width, 0.0);
                    self.kv.lock().append_kv(&pfx, &k, &v);
                    // Advance residual lightly so deeper layers see something non-constant.
                    if let Some(ref ffn) = block.ffn {
                        hh = self.weights.shared_ffn_residual_block(ffn, &hh);
                    }
                }
            }
            let _ = n_layers; // used in logs

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
        // Per-layer KV keys used by multi-layer stack.
        for li in 0..self.weights.n_layers().max(1) {
            let pfx = format!("{prefix_id}:L{li}");
            total += self.kv.lock().free_prefix(&pfx);
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
}
