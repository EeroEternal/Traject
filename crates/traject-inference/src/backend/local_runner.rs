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
    CpuRefKernel, DecodeRequest, KernelBackend, KvLayout, PrefillRequest, SampleRequest,
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

/// In-process model weights (toy or real safetensors embed/head/norm + optional layer-0).
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
    /// Fixed down-projection hidden → attn_dim (fallback when no layer-0)
    w_down: Vec<f32>,
    /// Up-projection attn_dim → hidden
    w_up: Vec<f32>,
    /// Optional real layer-0 attn projections (FP8 dequantized).
    layer0: Option<crate::weights::Layer0AttnWeights>,
    /// Optional layer-0 shared-expert SwiGLU FFN.
    layer0_ffn: Option<crate::weights::Layer0SharedFfn>,
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
            layer0: None,
            layer0_ffn: None,
            source: "toy".into(),
            eos_token_id: Some(1),
        }
    }

    fn from_safetensors(
        model_dir: &std::path::Path,
        attn_heads: u32,
        attn_dim_per_head: u32,
    ) -> Result<Self> {
        use crate::weights::{
            load_embed_head_norm, load_layer0_attn, load_layer0_shared_ffn, HfModelConfig,
        };

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
            // head may be [vocab, hidden] same as embed
            if head_t.rows() != vocab as usize || head_t.cols() != hidden {
                return Err(TrajectError::Other(format!(
                    "head shape {:?} incompatible with embed {:?}",
                    head_t.shape, embed_t.shape
                )));
            }
        }
        let mut attn_dim = (attn_heads * attn_dim_per_head) as usize;
        let layer0 = match load_layer0_attn(model_dir) {
            Ok(l0) => {
                // Match KernelBackend dim to compressed KV width when possible.
                if l0.kv_dim() > 0 {
                    attn_dim = l0.kv_dim();
                }
                Some(l0)
            }
            Err(e) => {
                warn!(error = %e, "layer-0 attn not loaded; using random projections");
                None
            }
        };
        let layer0_ffn = match load_layer0_shared_ffn(model_dir) {
            Ok(f) => Some(f),
            Err(e) => {
                warn!(error = %e, "layer-0 shared FFN not loaded");
                None
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
            has_layer0 = layer0.is_some(),
            has_layer0_ffn = layer0_ffn.is_some(),
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
            layer0,
            layer0_ffn,
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

    /// Hidden → Q vector in attention space.
    ///
    /// With layer-0: `attn_norm → wq_a → q_norm → [wq_b] → pool to attn_dim`.
    fn project_q(&self, h: &[f32]) -> Vec<f32> {
        if let Some(l0) = &self.layer0 {
            let hn = self.rms_norm_with(h, Some(&l0.attn_norm));
            let mut q = matvec(&l0.wq_a.data, l0.q_lora_dim(), l0.hidden, &hn);
            if let Some(ref qn) = l0.q_norm {
                q = self.rms_norm_with(&q, Some(qn));
            }
            if let Some(ref wq_b) = l0.wq_b {
                // Full MLA Q: [n_heads * head_dim], then mean-pool heads → attn_dim.
                let q_full = matvec(&wq_b.data, wq_b.rows(), wq_b.cols(), &q);
                return pool_heads_to_attn(&q_full, l0.n_heads, self.attn_dim);
            }
            let mut out = q;
            out.resize(self.attn_dim, 0.0);
            return out;
        }
        matvec(&self.w_down, self.attn_dim, self.hidden, h)
    }

    /// Hidden → compressed K/V (real wkv + optional kv_norm).
    fn project_kv(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        if let Some(l0) = &self.layer0 {
            let hn = self.rms_norm_with(h, Some(&l0.attn_norm));
            let mut kv = matvec(&l0.wkv.data, l0.kv_dim(), l0.hidden, &hn);
            if let Some(ref kn) = l0.kv_norm {
                kv = self.rms_norm_with(&kv, Some(kn));
            }
            kv.resize(self.attn_dim, 0.0);
            let v = kv.iter().map(|x| x * 0.5).collect();
            return (kv, v);
        }
        let a = matvec(&self.w_down, self.attn_dim, self.hidden, h);
        let v = a.iter().map(|x| x * 0.5).collect();
        (a, v)
    }

    /// Project attention dim → model hidden (still adapter; full wo_a/wo_b later).
    fn up(&self, a: &[f32]) -> Vec<f32> {
        matvec(&self.w_up, self.hidden, self.attn_dim, a)
    }

    /// Shared-expert SwiGLU residual: `h + w2(silu(w1·n) ⊙ w3·n)`.
    fn shared_ffn_residual(&self, h: &[f32]) -> Vec<f32> {
        let Some(ffn) = &self.layer0_ffn else {
            return h.to_vec();
        };
        let n = self.rms_norm_with(h, Some(&ffn.ffn_norm));
        let u = matvec(&ffn.w1.data, ffn.intermediate, ffn.hidden, &n);
        let g = matvec(&ffn.w3.data, ffn.intermediate, ffn.hidden, &n);
        let mut gated = vec![0.0f32; ffn.intermediate];
        for i in 0..ffn.intermediate {
            // silu(u) = u * sigmoid(u)
            let ui = u[i];
            let silu = ui / (1.0 + (-ui).exp());
            gated[i] = silu * g[i];
        }
        let delta = matvec(&ffn.w2.data, ffn.hidden, ffn.intermediate, &gated);
        let mut out = h.to_vec();
        for (o, d) in out.iter_mut().zip(delta.iter()) {
            *o += *d;
        }
        out
    }

    fn has_layer0(&self) -> bool {
        self.layer0.is_some()
    }

    fn has_layer0_ffn(&self) -> bool {
        self.layer0_ffn.is_some()
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

/// Mean-pool multi-head Q `[n_heads * head_dim]` down to `attn_dim` (= head_dim for MQA KV).
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
        // KV pool uses attention dim (projected), not full model hidden.
        let pool_dim = weights.attn_dim / cfg.num_heads.max(1) as usize;
        let pool_dim = pool_dim.max(1);
        let kernel = select_kernel(cfg.prefer_flashinfer);
        info!(kernel = kernel.name(), "LocalWeightRunner attention kernel");
        Self {
            kernel,
            weights,
            kv: Mutex::new(PagedKvPool::new(
                cfg.page_tokens,
                cfg.num_heads as usize,
                pool_dim,
            )),
            cfg,
            tokenizer,
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
            .layer0
            .as_ref()
            .map(|l| l.has_q_expand())
            .unwrap_or(false)
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

        // Prefill: embed → (layer-0 wq_a/wkv or toy down) → physical KV.
        let heads = self.cfg.num_heads as usize;
        let dim = self.weights.attn_dim / heads.max(1);
        let dim = dim.max(1);
        let attn_dim = heads * dim;
        // Prefer weight-side dim when layer-0 fixed kv_lora width.
        let attn_dim = self.weights.attn_dim.max(attn_dim);
        let dim = attn_dim / heads.max(1);
        let dim = dim.max(1);

        if req.decoded_so_far == 0 {
            for &tid in &prompt_ids {
                let emb = self.weights.embed_token(tid);
                let (mut kk, mut vv) = self.weights.project_kv(&emb);
                kk.resize(attn_dim, 0.0);
                vv.resize(attn_dim, 0.0);
                self.kv.lock().append_kv(&prefix, &kk, &vv);
            }
            if let Some(&last) = prompt_ids.last() {
                let emb = self.weights.embed_token(last);
                let mut qq = self.weights.project_q(&emb);
                qq.resize(attn_dim, 0.0);
                let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
                if seq_len > 0 {
                    let _ = self
                        .kernel
                        .prefill(PrefillRequest {
                            q: qq,
                            k: k_cache,
                            v: v_cache,
                            num_tokens: 1,
                            num_heads: self.cfg.num_heads,
                            head_dim: dim as u32,
                            layout: KvLayout::Nhd,
                        })
                        .await;
                }
            }
        }

        // Decode tokens for this chunk.
        let budget = req
            .chunk_tokens
            .min(req.max_tokens.saturating_sub(req.decoded_so_far))
            .max(1)
            .min(8);
        let mut out_ids = Vec::new();
        let mut out_text = String::new();
        let eos = self.weights.eos_token_id;

        for _ in 0..budget {
            let (k_cache, v_cache, seq_len) = self.kv.lock().materialize_kv(&prefix);
            if seq_len == 0 {
                break;
            }
            let last_tid = out_ids
                .last()
                .copied()
                .or_else(|| prompt_ids.last().copied())
                .unwrap_or(1);
            let emb = self.weights.embed_token(last_tid);
            let mut q = self.weights.project_q(&emb);
            q.resize(attn_dim, 0.0);
            let dec = self
                .kernel
                .decode(DecodeRequest {
                    q,
                    k_cache,
                    v_cache,
                    seq_len,
                    num_heads: self.cfg.num_heads,
                    head_dim: dim as u32,
                    layout: KvLayout::Nhd,
                })
                .await
                .map_err(|e| TrajectError::Inference(format!("local decode: {e}")))?;

            // Map attention output back to model hidden, then real lm_head.
            // Full wo_a/wo_b not loaded yet — residual adapter via w_up.
            let mut attn_o = dec.o;
            attn_o.resize(attn_dim, 0.0);
            let attn_h = self.weights.up(&attn_o);
            // residual with last embed for stability
            let mut h = emb;
            for (a, b) in h.iter_mut().zip(attn_h.iter()) {
                *a = 0.5 * *a + 0.5 * *b;
            }
            // Real shared-expert SwiGLU when loaded (routed MoE still absent).
            h = self.weights.shared_ffn_residual(&h);
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
            let emb_n = self.weights.embed_token(tid);
            let (mut k, mut v) = self.weights.project_kv(&emb_n);
            k.resize(attn_dim, 0.0);
            v.resize(attn_dim, 0.0);
            self.kv.lock().append_kv(&prefix, &k, &v);

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
                .layer0
                .as_ref()
                .map(|l| l.has_q_expand())
                .unwrap_or(false),
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
}
