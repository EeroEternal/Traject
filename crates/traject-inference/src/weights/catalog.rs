//! Sharded HuggingFace safetensors catalog (index.json + per-file mmap).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use serde::Deserialize;
use tracing::{info, warn};
use traject_core::{Result, TrajectError};

use super::dtype::{
    bytes_to_f32_vec, dequant_fp4_block_scaled, dequant_fp8_block_scaled, matvec_fp4_block_scaled,
    matvec_fp8_block_scaled, row_dot_fp8_block_scaled,
};

#[derive(Debug, Deserialize)]
struct IndexFile {
    weight_map: HashMap<String, String>,
}

/// Dense f32 tensor + shape.
#[derive(Debug, Clone)]
pub struct TensorF32 {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl TensorF32 {
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }
    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(1)
    }
}

/// Packed FP8 e4m3 + e8m0 block scales (no full f32 expand).
#[derive(Debug, Clone)]
pub struct PackedFp8Mat {
    pub weight: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
    pub scale: Vec<u8>,
    pub scale_rows: usize,
    pub scale_cols: usize,
    pub block: usize,
}

impl PackedFp8Mat {
    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        matvec_fp8_block_scaled(
            &self.weight,
            self.rows,
            self.cols,
            &self.scale,
            self.scale_rows,
            self.scale_cols,
            self.block,
            x,
        )
        .map_err(|e| TrajectError::Other(e))
    }

    pub fn row_dot(&self, row: usize, x: &[f32]) -> Result<f32> {
        row_dot_fp8_block_scaled(
            &self.weight,
            self.rows,
            self.cols,
            &self.scale,
            self.scale_cols,
            self.block,
            row,
            x,
        )
        .map_err(|e| TrajectError::Other(e))
    }
}

/// Dense linear weight: f32 dequant **or** packed FP8 (preferred for V4).
#[derive(Debug, Clone)]
pub enum LinearMat {
    F32(TensorF32),
    Fp8(PackedFp8Mat),
}

impl LinearMat {
    pub fn rows(&self) -> usize {
        match self {
            LinearMat::F32(t) => t.rows(),
            LinearMat::Fp8(p) => p.rows,
        }
    }
    pub fn cols(&self) -> usize {
        match self {
            LinearMat::F32(t) => t.cols(),
            LinearMat::Fp8(p) => p.cols,
        }
    }
    pub fn is_fp8(&self) -> bool {
        matches!(self, LinearMat::Fp8(_))
    }
    /// `y = W @ x` with `W` shape `[rows, cols]`.
    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match self {
            LinearMat::F32(t) => {
                let rows = t.rows();
                let cols = t.cols().max(1);
                let mut y = vec![0.0f32; rows];
                for i in 0..rows {
                    let row = &t.data[i * cols..(i + 1) * cols];
                    let mut s = 0.0f32;
                    for (a, b) in row.iter().zip(x.iter()) {
                        s += a * b;
                    }
                    y[i] = s;
                }
                y
            }
            LinearMat::Fp8(p) => p.matvec(x).unwrap_or_else(|_| vec![0.0; p.rows]),
        }
    }
    pub fn row_dot(&self, row: usize, x: &[f32]) -> f32 {
        match self {
            LinearMat::F32(t) => {
                let cols = t.cols().max(1);
                if row >= t.rows() || x.len() < cols {
                    return 0.0;
                }
                let r = &t.data[row * cols..(row + 1) * cols];
                let mut s = 0.0f32;
                for (a, b) in r.iter().zip(x.iter()) {
                    s += a * b;
                }
                s
            }
            LinearMat::Fp8(p) => p.row_dot(row, x).unwrap_or(0.0),
        }
    }
}

/// Open a model directory with optional `model.safetensors.index.json`.
pub struct SafetensorCatalog {
    root: PathBuf,
    /// tensor name → relative shard filename
    weight_map: HashMap<String, String>,
    /// cache of mmap'd shards
    open: HashMap<String, Mmap>,
}

impl SafetensorCatalog {
    pub fn open(model_dir: &Path) -> Result<Self> {
        let root = model_dir.to_path_buf();
        if !root.is_dir() {
            return Err(TrajectError::Other(format!(
                "model dir not found: {}",
                root.display()
            )));
        }
        let index_path = root.join("model.safetensors.index.json");
        let weight_map = if index_path.exists() {
            let raw = std::fs::read_to_string(&index_path)
                .map_err(|e| TrajectError::Other(format!("read index: {e}")))?;
            let idx: IndexFile = serde_json::from_str(&raw)
                .map_err(|e| TrajectError::Other(format!("parse index: {e}")))?;
            idx.weight_map
        } else {
            // Single-file or discover first *.safetensors and list keys lazily later.
            HashMap::new()
        };
        info!(
            dir = %root.display(),
            tensors = weight_map.len(),
            "opened safetensors catalog"
        );
        Ok(Self {
            root,
            weight_map,
            open: HashMap::new(),
        })
    }

    fn shard_for(&self, name: &str) -> Result<PathBuf> {
        if let Some(rel) = self.weight_map.get(name) {
            return Ok(self.root.join(rel));
        }
        // Fallback: scan single safetensors files in root.
        let entries = std::fs::read_dir(&self.root)
            .map_err(|e| TrajectError::Other(format!("read dir: {e}")))?;
        for e in entries.flatten() {
            let p = e.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                // open and check key existence
                if let Ok(file) = File::open(&p) {
                    if let Ok(mmap) = unsafe { Mmap::map(&file) } {
                        if let Ok(st) = SafeTensors::deserialize(&mmap) {
                            if st.tensor(name).is_ok() {
                                return Ok(p);
                            }
                        }
                    }
                }
            }
        }
        Err(TrajectError::Other(format!(
            "tensor `{name}` not found in {}",
            self.root.display()
        )))
    }

    fn ensure_mmap(&mut self, shard: &Path) -> Result<&Mmap> {
        let key = shard.to_string_lossy().into_owned();
        if !self.open.contains_key(&key) {
            let file = File::open(shard)
                .map_err(|e| TrajectError::Other(format!("open {}: {e}", shard.display())))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| TrajectError::Other(format!("mmap {}: {e}", shard.display())))?;
            self.open.insert(key.clone(), mmap);
        }
        Ok(self.open.get(&key).expect("just inserted"))
    }

    /// Load a tensor as contiguous f32 (BF16/F16/F32/F8_* unscaled).
    pub fn load_f32(&mut self, name: &str) -> Result<TensorF32> {
        let shard = self.shard_for(name)?;
        let mmap = self.ensure_mmap(&shard)?;
        let st = SafeTensors::deserialize(mmap)
            .map_err(|e| TrajectError::Other(format!("deserialize {}: {e}", shard.display())))?;
        let view = st
            .tensor(name)
            .map_err(|e| TrajectError::Other(format!("tensor `{name}`: {e}")))?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = bytes_to_f32_vec(view.data(), view.dtype())
            .map_err(|e| TrajectError::Other(format!("decode `{name}`: {e}")))?;
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            return Err(TrajectError::Other(format!(
                "tensor `{name}` numel mismatch: got {} expect {expected} shape={shape:?}",
                data.len()
            )));
        }
        info!(
            tensor = name,
            shape = ?shape,
            dtype = ?view.dtype(),
            shard = %shard.file_name().unwrap_or_default().to_string_lossy(),
            "loaded safetensor as f32"
        );
        Ok(TensorF32 { data, shape })
    }

    /// Load DeepSeek-style block-scaled FP8 weight (`*.weight` + sibling `*.scale`).
    ///
    /// Weight is F8_E4M3, scale is F8_E8M0, block size 128 (V4 default).
    /// Example: `layers.0.attn.wq_a.weight` + `layers.0.attn.wq_a.scale`.
    pub fn load_fp8_block_scaled(&mut self, weight_name: &str, block: usize) -> Result<TensorF32> {
        // `layers.0.attn.wq_a.weight` → `layers.0.attn.wq_a.scale`
        let scale_name = if weight_name.ends_with(".weight") {
            format!("{}.scale", weight_name.trim_end_matches(".weight"))
        } else {
            format!("{weight_name}.scale")
        };
        if !self.has(&scale_name) {
            return Err(TrajectError::Other(format!(
                "no scale tensor for `{weight_name}` (expected `{scale_name}`)"
            )));
        }

        let (w_bytes, w_shape, w_dtype) = self.load_raw(weight_name)?;
        let (s_bytes, s_shape, s_dtype) = self.load_raw(&scale_name)?;
        if w_dtype != safetensors::Dtype::F8_E4M3 {
            return Err(TrajectError::Other(format!(
                "`{weight_name}` dtype {w_dtype:?}, expected F8_E4M3"
            )));
        }
        if s_dtype != safetensors::Dtype::F8_E8M0 {
            return Err(TrajectError::Other(format!(
                "`{scale_name}` dtype {s_dtype:?}, expected F8_E8M0"
            )));
        }
        let data = dequant_fp8_block_scaled(&w_bytes, &w_shape, &s_bytes, &s_shape, block)
            .map_err(|e| TrajectError::Other(format!("dequant `{weight_name}`: {e}")))?;
        info!(
            tensor = weight_name,
            scale = %scale_name,
            shape = ?w_shape,
            block,
            "loaded FP8 block-scaled weight as f32"
        );
        Ok(TensorF32 {
            data,
            shape: w_shape,
        })
    }

    /// Load FP8 weight **packed** (no f32 expand). Prefer for dense attn/FFN.
    pub fn load_fp8_packed(&mut self, weight_name: &str, block: usize) -> Result<PackedFp8Mat> {
        let scale_name = if weight_name.ends_with(".weight") {
            format!("{}.scale", weight_name.trim_end_matches(".weight"))
        } else {
            format!("{weight_name}.scale")
        };
        if !self.has(&scale_name) {
            return Err(TrajectError::Other(format!(
                "no scale tensor for `{weight_name}` (expected `{scale_name}`)"
            )));
        }
        let (w_bytes, w_shape, w_dtype) = self.load_raw(weight_name)?;
        let (s_bytes, s_shape, s_dtype) = self.load_raw(&scale_name)?;
        if w_dtype != safetensors::Dtype::F8_E4M3 {
            return Err(TrajectError::Other(format!(
                "`{weight_name}` dtype {w_dtype:?}, expected F8_E4M3"
            )));
        }
        if s_dtype != safetensors::Dtype::F8_E8M0 {
            return Err(TrajectError::Other(format!(
                "`{scale_name}` dtype {s_dtype:?}, expected F8_E8M0"
            )));
        }
        if w_shape.len() != 2 || s_shape.len() != 2 {
            return Err(TrajectError::Other(format!(
                "fp8 packed expects 2D weight/scale, got {w_shape:?} / {s_shape:?}"
            )));
        }
        let rows = w_shape[0];
        let cols = w_shape[1];
        info!(
            tensor = weight_name,
            scale = %scale_name,
            shape = ?w_shape,
            block,
            bytes = w_bytes.len(),
            "loaded FP8 block-scaled weight packed (no f32 expand)"
        );
        Ok(PackedFp8Mat {
            weight: w_bytes,
            rows,
            cols,
            scale: s_bytes,
            scale_rows: s_shape[0],
            scale_cols: s_shape[1],
            block: block.max(1),
        })
    }

    /// Load DeepSeek FP4 expert weight: packed `I8` + `F8_E8M0` scale, block_k=32.
    ///
    /// Returns dequantized f32 with logical shape `[rows, packed_cols * 2]`.
    pub fn load_fp4_block_scaled(
        &mut self,
        weight_name: &str,
        block_k: usize,
    ) -> Result<TensorF32> {
        let scale_name = if weight_name.ends_with(".weight") {
            format!("{}.scale", weight_name.trim_end_matches(".weight"))
        } else {
            format!("{weight_name}.scale")
        };
        if !self.has(&scale_name) {
            return Err(TrajectError::Other(format!(
                "no scale for FP4 `{weight_name}` (expected `{scale_name}`)"
            )));
        }
        let (w_bytes, w_shape, w_dtype) = self.load_raw(weight_name)?;
        let (s_bytes, s_shape, s_dtype) = self.load_raw(&scale_name)?;
        // Packed FP4 stored as I8 or U8 in safetensors.
        if !matches!(w_dtype, safetensors::Dtype::I8 | safetensors::Dtype::U8) {
            return Err(TrajectError::Other(format!(
                "`{weight_name}` dtype {w_dtype:?}, expected I8/U8 packed FP4"
            )));
        }
        if s_dtype != safetensors::Dtype::F8_E8M0 {
            return Err(TrajectError::Other(format!(
                "`{scale_name}` dtype {s_dtype:?}, expected F8_E8M0"
            )));
        }
        let data = dequant_fp4_block_scaled(&w_bytes, &w_shape, &s_bytes, &s_shape, block_k)
            .map_err(|e| TrajectError::Other(format!("fp4 dequant `{weight_name}`: {e}")))?;
        let logical = vec![w_shape[0], w_shape[1] * 2];
        debug_assert_eq!(data.len(), logical[0] * logical[1]);
        Ok(TensorF32 {
            data,
            shape: logical,
        })
    }

    /// Load packed FP4 weight + scale **without** expanding to f32.
    pub fn load_fp4_packed(&mut self, weight_name: &str, block_k: usize) -> Result<PackedFp4Mat> {
        let scale_name = if weight_name.ends_with(".weight") {
            format!("{}.scale", weight_name.trim_end_matches(".weight"))
        } else {
            format!("{weight_name}.scale")
        };
        if !self.has(&scale_name) {
            return Err(TrajectError::Other(format!(
                "no scale for FP4 `{weight_name}` (expected `{scale_name}`)"
            )));
        }
        let (w_bytes, w_shape, w_dtype) = self.load_raw(weight_name)?;
        let (s_bytes, s_shape, s_dtype) = self.load_raw(&scale_name)?;
        if !matches!(w_dtype, safetensors::Dtype::I8 | safetensors::Dtype::U8) {
            return Err(TrajectError::Other(format!(
                "`{weight_name}` dtype {w_dtype:?}, expected I8/U8 packed FP4"
            )));
        }
        if s_dtype != safetensors::Dtype::F8_E8M0 {
            return Err(TrajectError::Other(format!(
                "`{scale_name}` dtype {s_dtype:?}, expected F8_E8M0"
            )));
        }
        if w_shape.len() != 2 || s_shape.len() != 2 {
            return Err(TrajectError::Other(format!(
                "fp4 packed wants 2D shapes, got weight={w_shape:?} scale={s_shape:?}"
            )));
        }
        Ok(PackedFp4Mat {
            packed: w_bytes,
            rows: w_shape[0],
            packed_cols: w_shape[1],
            scale: s_bytes,
            scale_cols: s_shape[1],
            block_k: block_k.max(1),
        })
    }

    fn load_raw(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
        let shard = self.shard_for(name)?;
        let mmap = self.ensure_mmap(&shard)?;
        let st = SafeTensors::deserialize(mmap)
            .map_err(|e| TrajectError::Other(format!("deserialize {}: {e}", shard.display())))?;
        let view = st
            .tensor(name)
            .map_err(|e| TrajectError::Other(format!("tensor `{name}`: {e}")))?;
        Ok((view.data().to_vec(), view.shape().to_vec(), view.dtype()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.weight_map.contains_key(name) || {
            // avoid recursive error noise on missing single-file fallback
            self.shard_for(name).is_ok()
        }
    }
}

/// Layer-0 attention projections (DeepSeek-V4 MLA compressed + Q expand + o_proj).
///
/// Routed MoE experts are **not** loaded here.
#[derive(Debug, Clone)]
pub struct Layer0AttnWeights {
    /// RMSNorm γ before attention, shape [hidden].
    pub attn_norm: Vec<f32>,
    pub hidden: usize,
    /// `wq_a`: [q_lora, hidden] — Q down-projection.
    pub wq_a: LinearMat,
    /// `wkv`: [kv_lora / head_dim, hidden] — shared MQA latent (K=V).
    pub wkv: LinearMat,
    /// Optional RMSNorm on q_lora (after `wq_a`).
    pub q_norm: Option<Vec<f32>>,
    /// Optional RMSNorm on kv_lora (after `wkv`).
    pub kv_norm: Option<Vec<f32>>,
    /// Optional `wq_b`: [n_heads * head_dim, q_lora] — expand Q to full heads.
    pub wq_b: Option<LinearMat>,
    /// Head count implied by `wq_b` rows / `kv_lora` (when present).
    pub n_heads: Option<usize>,
    /// Optional `wo_a`: [o_groups * o_lora, heads_per_group * head_dim] —
    /// maps concatenated group heads → o_lora (V4 Flash; often 4096 == hidden).
    pub wo_a: Option<LinearMat>,
    /// Optional `wo_b`: [hidden, o_groups * o_lora] — maps o-intermediate → hidden.
    pub wo_b: Option<LinearMat>,
    /// `o_groups` (default 8 for V4 Flash).
    pub o_groups: usize,
    /// `o_lora_rank` per group (default 1024).
    pub o_lora_rank: usize,
    /// Per-head attention-sink logits (`attn_sink`), length `n_heads` when present.
    pub attn_sink: Option<Vec<f32>>,
    /// RoPE dims applied on the last slice of each head (`qk_rope_head_dim`, default 64).
    pub rope_head_dim: usize,
}

impl Layer0AttnWeights {
    pub fn q_lora_dim(&self) -> usize {
        self.wq_a.rows()
    }
    pub fn kv_dim(&self) -> usize {
        self.wkv.rows()
    }
    pub fn has_q_expand(&self) -> bool {
        self.wq_b.is_some()
    }
    pub fn has_o_proj(&self) -> bool {
        self.wo_b.is_some()
    }
    /// Full Q width after `wq_b` (e.g. 32768), if loaded.
    pub fn q_full_dim(&self) -> Option<usize> {
        self.wq_b.as_ref().map(|t| t.rows())
    }
    pub fn o_intermediate(&self) -> usize {
        self.o_groups.saturating_mul(self.o_lora_rank).max(1)
    }
}

/// Layer-0 **shared** expert SwiGLU FFN (not the 256 routed experts).
///
/// `y = w2( silu(w1 x) ⊙ (w3 x) )` after `ffn_norm`.
#[derive(Debug, Clone)]
pub struct Layer0SharedFfn {
    pub ffn_norm: Vec<f32>,
    pub hidden: usize,
    pub intermediate: usize,
    /// [intermediate, hidden]
    pub w1: LinearMat,
    /// [hidden, intermediate]
    pub w2: LinearMat,
    /// [intermediate, hidden]
    pub w3: LinearMat,
}

/// Prefer packed FP8; fall back to f32 dequant / native f32.
fn load_weight_linear(cat: &mut SafetensorCatalog, names: &[&str]) -> Result<LinearMat> {
    for n in names {
        if !cat.has(n) {
            continue;
        }
        let key = if n.ends_with(".weight") {
            (*n).to_string()
        } else if cat.has(&format!("{n}.weight")) {
            format!("{n}.weight")
        } else {
            (*n).to_string()
        };
        let scale = if key.ends_with(".weight") {
            format!("{}.scale", key.trim_end_matches(".weight"))
        } else {
            format!("{key}.scale")
        };
        if cat.has(&scale) {
            if let Ok(p) = cat.load_fp8_packed(&key, 128) {
                return Ok(LinearMat::Fp8(p));
            }
            // Fallback: full dequant (legacy path).
            if let Ok(t) = cat.load_fp8_block_scaled(&key, 128) {
                return Ok(LinearMat::F32(t));
            }
        }
        if let Ok(t) = cat.load_f32(&key) {
            return Ok(LinearMat::F32(t));
        }
        if let Ok(t) = cat.load_f32(n) {
            return Ok(LinearMat::F32(t));
        }
    }
    Err(TrajectError::Other(format!(
        "none of {names:?} found/loadable"
    )))
}

/// Load attention weights for `layers.{layer}` (DeepSeek-V4 MLA path).
pub fn load_layer_attn(model_dir: &Path, layer: usize) -> Result<Layer0AttnWeights> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    load_layer_attn_into(&mut cat, model_dir, layer)
}

fn load_layer_attn_into(
    mut cat: &mut SafetensorCatalog,
    model_dir: &Path,
    layer: usize,
) -> Result<Layer0AttnWeights> {
    let norm_names = [
        format!("layers.{layer}.attn_norm.weight"),
        format!("model.layers.{layer}.input_layernorm.weight"),
    ];
    let mut attn_norm = None;
    for n in &norm_names {
        if let Ok(t) = cat.load_f32(n) {
            attn_norm = Some(t);
            break;
        }
    }
    let attn_norm = attn_norm.ok_or_else(|| {
        TrajectError::Other(format!(
            "layer-{layer} attn_norm not found in {}",
            model_dir.display()
        ))
    })?;

    let wq_a_names = [
        format!("layers.{layer}.attn.wq_a.weight"),
        format!("layers.{layer}.attn.wq_a"),
        format!("model.layers.{layer}.self_attn.q_a_proj.weight"),
    ];
    let wq_a_refs: Vec<&str> = wq_a_names.iter().map(|s| s.as_str()).collect();
    let wq_a = load_weight_linear(&mut cat, &wq_a_refs)?;
    let wkv_names = [
        format!("layers.{layer}.attn.wkv.weight"),
        format!("layers.{layer}.attn.wkv"),
        format!("model.layers.{layer}.self_attn.kv_a_proj_with_mqa.weight"),
    ];
    let wkv_refs: Vec<&str> = wkv_names.iter().map(|s| s.as_str()).collect();
    let wkv = load_weight_linear(&mut cat, &wkv_refs)?;

    let hidden = attn_norm.data.len();
    if wq_a.cols() != hidden || wkv.cols() != hidden {
        return Err(TrajectError::Other(format!(
            "layer-{layer} in_features mismatch: norm_h={hidden} wq_a=[{},{}] wkv=[{},{}]",
            wq_a.rows(),
            wq_a.cols(),
            wkv.rows(),
            wkv.cols()
        )));
    }

    let q_norm = cat
        .load_f32(&format!("layers.{layer}.attn.q_norm.weight"))
        .ok()
        .map(|t| t.data);
    let kv_norm = cat
        .load_f32(&format!("layers.{layer}.attn.kv_norm.weight"))
        .ok()
        .map(|t| t.data);

    // Optional Q expand (full MLA heads). ~134MB f32 for V4 Flash.
    let wq_b_names = [
        format!("layers.{layer}.attn.wq_b.weight"),
        format!("layers.{layer}.attn.wq_b"),
        format!("model.layers.{layer}.self_attn.q_b_proj.weight"),
    ];
    let wq_b_refs: Vec<&str> = wq_b_names.iter().map(|s| s.as_str()).collect();
    let wq_b = match load_weight_linear(&mut cat, &wq_b_refs) {
        Ok(t) if t.cols() == wq_a.rows() => Some(t),
        Ok(t) => {
            warn!(
                rows = t.rows(),
                cols = t.cols(),
                q_lora = wq_a.rows(),
                "wq_b shape incompatible with wq_a; skipping Q expand"
            );
            None
        }
        Err(e) => {
            warn!(error = %e, "wq_b not loaded; Q stays at q_lora");
            None
        }
    };

    let n_heads = wq_b.as_ref().and_then(|t| {
        let kv = wkv.rows();
        if kv > 0 && t.rows() % kv == 0 {
            Some(t.rows() / kv)
        } else {
            None
        }
    });

    if let Some(ref qn) = q_norm {
        if qn.len() != wq_a.rows() {
            return Err(TrajectError::Other(format!(
                "q_norm len {} != q_lora {}",
                qn.len(),
                wq_a.rows()
            )));
        }
    }
    if let Some(ref kn) = kv_norm {
        if kn.len() != wkv.rows() {
            return Err(TrajectError::Other(format!(
                "kv_norm len {} != kv_lora {}",
                kn.len(),
                wkv.rows()
            )));
        }
    }

    // o_proj factors (V4 Flash defaults: o_groups=8, o_lora_rank=1024 → 8192).
    let (o_groups, o_lora_rank, rope_head_dim) = {
        use crate::weights::HfModelConfig;
        match HfModelConfig::load(model_dir) {
            Ok(cfg) => (
                cfg.o_groups.unwrap_or(8) as usize,
                cfg.o_lora_rank.unwrap_or(1024) as usize,
                cfg.qk_rope_head_dim.unwrap_or(64) as usize,
            ),
            Err(_) => (8, 1024, 64),
        }
    };
    let o_inter = o_groups * o_lora_rank;
    // Group input width = (n_heads / o_groups) * head_dim; equals hidden on V4 Flash (8×512).
    let head_dim = wkv.rows().max(1);
    let n_heads_guess = n_heads.unwrap_or(o_groups.max(1));
    let heads_per_group = (n_heads_guess / o_groups.max(1)).max(1);
    let group_in = heads_per_group * head_dim;

    let wo_a_names = [
        format!("layers.{layer}.attn.wo_a.weight"),
        format!("layers.{layer}.attn.wo_a"),
        format!("model.layers.{layer}.self_attn.o_a_proj.weight"),
    ];
    let wo_a_refs: Vec<&str> = wo_a_names.iter().map(|s| s.as_str()).collect();
    let wo_a = match load_weight_linear(&mut cat, &wo_a_refs) {
        Ok(t)
            if (t.rows() == o_inter && t.cols() == group_in)
                || (t.rows() == o_inter && t.cols() == hidden) =>
        {
            Some(t)
        }
        Ok(t) => {
            warn!(
                rows = t.rows(),
                cols = t.cols(),
                want_group = ?[o_inter, group_in],
                "wo_a shape mismatch; skip"
            );
            None
        }
        Err(e) => {
            warn!(error = %e, "wo_a not loaded");
            None
        }
    };

    let attn_sink = cat
        .load_f32(&format!("layers.{layer}.attn.attn_sink"))
        .ok()
        .map(|t| t.data);
    let wo_b_names = [
        format!("layers.{layer}.attn.wo_b.weight"),
        format!("layers.{layer}.attn.wo_b"),
        format!("model.layers.{layer}.self_attn.o_b_proj.weight"),
    ];
    let wo_b_refs: Vec<&str> = wo_b_names.iter().map(|s| s.as_str()).collect();
    let wo_b = match load_weight_linear(&mut cat, &wo_b_refs) {
        Ok(t) if t.rows() == hidden && t.cols() == o_inter => Some(t),
        Ok(t) => {
            warn!(
                rows = t.rows(),
                cols = t.cols(),
                want = ?[hidden, o_inter],
                "wo_b shape mismatch; skip"
            );
            None
        }
        Err(e) => {
            warn!(error = %e, "wo_b not loaded");
            None
        }
    };

    info!(
        dir = %model_dir.display(),
        hidden,
        q_lora = wq_a.rows(),
        kv_lora = wkv.rows(),
        packed_fp8 = wq_a.is_fp8() && wkv.is_fp8(),
        has_q_norm = q_norm.is_some(),
        has_kv_norm = kv_norm.is_some(),
        has_wq_b = wq_b.is_some(),
        has_wo_a = wo_a.is_some(),
        has_wo_b = wo_b.is_some(),
        has_attn_sink = attn_sink.is_some(),
        rope_head_dim,
        q_full = wq_b.as_ref().map(|t| t.rows()),
        n_heads = ?n_heads,
        o_groups,
        o_lora_rank,
        layer,
        "loaded layer attention projections (MLA Q + o_proj)"
    );

    Ok(Layer0AttnWeights {
        attn_norm: attn_norm.data,
        hidden,
        wq_a,
        wkv,
        q_norm,
        kv_norm,
        wq_b,
        n_heads,
        wo_a,
        wo_b,
        o_groups,
        o_lora_rank,
        attn_sink,
        rope_head_dim,
    })
}

/// Packed FP4 matrix (I8 nibbles + e8m0 scales) for fused matvec.
#[derive(Debug, Clone)]
pub struct PackedFp4Mat {
    pub packed: Vec<u8>,
    pub rows: usize,
    pub packed_cols: usize,
    pub scale: Vec<u8>,
    pub scale_cols: usize,
    pub block_k: usize,
}

impl PackedFp4Mat {
    pub fn logical_cols(&self) -> usize {
        self.packed_cols.saturating_mul(2)
    }

    pub fn matvec(&self, x: &[f32]) -> Result<Vec<f32>> {
        matvec_fp4_block_scaled(
            &self.packed,
            self.rows,
            self.packed_cols,
            &self.scale,
            self.scale_cols,
            self.block_k,
            x,
        )
        .map_err(|e| TrajectError::Other(e))
    }
}

/// Dequantized f32 views of an expert (built lazily on first SwiGLU).
#[derive(Debug, Clone)]
struct ExpertF32Mats {
    w1: Vec<f32>,
    w1_rows: usize,
    w1_cols: usize,
    w2: Vec<f32>,
    w2_rows: usize,
    w2_cols: usize,
    w3: Vec<f32>,
    w3_rows: usize,
    w3_cols: usize,
}

/// One routed expert: packed FP4 on load; f32 expanded once on first use.
#[derive(Debug)]
pub struct ExpertPacked {
    pub w1: PackedFp4Mat,
    pub w2: PackedFp4Mat,
    pub w3: PackedFp4Mat,
    /// Lazy full dequant for repeated SwiGLU (OnceLock is not Clone).
    f32_once: std::sync::OnceLock<ExpertF32Mats>,
}

impl Clone for ExpertPacked {
    fn clone(&self) -> Self {
        // Fresh OnceLock — clone keeps packed bytes; f32 rebuilds on next use.
        Self {
            w1: self.w1.clone(),
            w2: self.w2.clone(),
            w3: self.w3.clone(),
            f32_once: std::sync::OnceLock::new(),
        }
    }
}

impl ExpertPacked {
    fn ensure_f32(&self) -> Result<&ExpertF32Mats> {
        if let Some(m) = self.f32_once.get() {
            return Ok(m);
        }
        let w1 = dequant_fp4_block_scaled(
            &self.w1.packed,
            &[self.w1.rows, self.w1.packed_cols],
            &self.w1.scale,
            &[self.w1.rows, self.w1.scale_cols],
            self.w1.block_k,
        )
        .map_err(|e| TrajectError::Other(e))?;
        let w2 = dequant_fp4_block_scaled(
            &self.w2.packed,
            &[self.w2.rows, self.w2.packed_cols],
            &self.w2.scale,
            &[self.w2.rows, self.w2.scale_cols],
            self.w2.block_k,
        )
        .map_err(|e| TrajectError::Other(e))?;
        let w3 = dequant_fp4_block_scaled(
            &self.w3.packed,
            &[self.w3.rows, self.w3.packed_cols],
            &self.w3.scale,
            &[self.w3.rows, self.w3.scale_cols],
            self.w3.block_k,
        )
        .map_err(|e| TrajectError::Other(e))?;
        let mats = ExpertF32Mats {
            w1_rows: self.w1.rows,
            w1_cols: self.w1.logical_cols(),
            w1,
            w2_rows: self.w2.rows,
            w2_cols: self.w2.logical_cols(),
            w2,
            w3_rows: self.w3.rows,
            w3_cols: self.w3.logical_cols(),
            w3,
        };
        let _ = self.f32_once.set(mats);
        Ok(self.f32_once.get().expect("just set"))
    }

    /// SwiGLU via lazy f32 expand (first call dequants; later calls are pure GEMV).
    pub fn swiglu(&self, x: &[f32]) -> Result<Vec<f32>> {
        let m = self.ensure_f32()?;
        // Prefer f32 matvec after expand.
        let u = matvec_f32(&m.w1, m.w1_rows, m.w1_cols, x);
        let g = matvec_f32(&m.w3, m.w3_rows, m.w3_cols, x);
        let inter = u.len().min(g.len());
        let mut gated = vec![0.0f32; inter];
        for i in 0..inter {
            let ui = u[i];
            let silu = ui / (1.0 + (-ui).exp());
            gated[i] = silu * g[i];
        }
        Ok(matvec_f32(&m.w2, m.w2_rows, m.w2_cols, &gated))
    }

    /// One-shot fused path without caching f32 (used in unit tests / tiny mats).
    pub fn swiglu_fused(&self, x: &[f32]) -> Result<Vec<f32>> {
        let u = self.w1.matvec(x)?;
        let g = self.w3.matvec(x)?;
        let inter = u.len().min(g.len());
        let mut gated = vec![0.0f32; inter];
        for i in 0..inter {
            let ui = u[i];
            let silu = ui / (1.0 + (-ui).exp());
            gated[i] = silu * g[i];
        }
        self.w2.matvec(&gated)
    }
}

fn matvec_f32(w: &[f32], out_dim: usize, in_dim: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row = &w[i * in_dim..(i + 1) * in_dim];
        let mut s = 0.0f32;
        for (a, b) in row.iter().zip(x.iter()) {
            s += a * b;
        }
        y[i] = s;
    }
    y
}

/// Backward-compat alias: experts are packed, not fully dequantized.
pub type ExpertF32 = ExpertPacked;

/// True LRU cache of packed FP4 experts (`Arc` so OnceLock f32 expand is shared).
struct ExpertLru {
    map: HashMap<usize, std::sync::Arc<ExpertPacked>>,
    /// Front = oldest, back = newest.
    order: std::collections::VecDeque<usize>,
    cap: usize,
    hits: u64,
    misses: u64,
}

impl ExpertLru {
    fn new(cap: usize) -> Self {
        Self {
            map: HashMap::new(),
            order: std::collections::VecDeque::new(),
            cap: cap.max(1),
            hits: 0,
            misses: 0,
        }
    }

    fn get(&mut self, id: usize) -> Option<std::sync::Arc<ExpertPacked>> {
        if let Some(e) = self.map.get(&id) {
            self.hits += 1;
            if let Some(pos) = self.order.iter().position(|&x| x == id) {
                self.order.remove(pos);
            }
            self.order.push_back(id);
            return Some(std::sync::Arc::clone(e));
        }
        self.misses += 1;
        None
    }

    fn put(&mut self, id: usize, e: std::sync::Arc<ExpertPacked>) {
        if self.map.contains_key(&id) {
            self.map.insert(id, e);
            if let Some(pos) = self.order.iter().position(|&x| x == id) {
                self.order.remove(pos);
            }
            self.order.push_back(id);
            return;
        }
        while self.map.len() >= self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            } else {
                break;
            }
        }
        self.map.insert(id, e);
        self.order.push_back(id);
    }
}

#[cfg(test)]
mod expert_lru_tests {
    use super::*;

    fn dummy_expert(tag: u8) -> ExpertPacked {
        let mat = PackedFp4Mat {
            packed: vec![tag],
            rows: 1,
            packed_cols: 1,
            scale: vec![127],
            scale_cols: 1,
            block_k: 2,
        };
        ExpertPacked {
            w1: mat.clone(),
            w2: mat.clone(),
            w3: mat,
            f32_once: std::sync::OnceLock::new(),
        }
    }

    #[test]
    fn lru_evicts_oldest() {
        use std::sync::Arc;
        let mut c = ExpertLru::new(2);
        c.put(1, Arc::new(dummy_expert(1)));
        c.put(2, Arc::new(dummy_expert(2)));
        assert!(c.get(1).is_some());
        c.put(3, Arc::new(dummy_expert(3)));
        assert!(c.get(2).is_none());
        assert!(c.get(1).is_some());
        assert!(c.get(3).is_some());
        assert_eq!(c.hits, 3);
        assert_eq!(c.misses, 1);
    }
}

/// Routed MoE: gate + lazy **packed FP4** expert cache (kept-open catalog + LRU).
///
/// Expert miss loads packed weights only (~12MB); matvec dequants on the fly.
pub struct Layer0RoutedMoe {
    pub model_dir: PathBuf,
    pub layer: usize,
    /// [n_experts, hidden]
    pub gate: TensorF32,
    pub n_experts: usize,
    pub top_k: usize,
    pub route_scale: f32,
    pub hidden: usize,
    pub intermediate: usize,
    catalog: std::sync::Mutex<SafetensorCatalog>,
    cache: std::sync::Mutex<ExpertLru>,
}

impl std::fmt::Debug for Layer0RoutedMoe {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Layer0RoutedMoe")
            .field("model_dir", &self.model_dir)
            .field("layer", &self.layer)
            .field("n_experts", &self.n_experts)
            .field("top_k", &self.top_k)
            .field("route_scale", &self.route_scale)
            .field("hidden", &self.hidden)
            .field("intermediate", &self.intermediate)
            .finish_non_exhaustive()
    }
}

impl Clone for Layer0RoutedMoe {
    fn clone(&self) -> Self {
        // Re-open catalog + empty LRU (mmap state is not shared across clones).
        let catalog = SafetensorCatalog::open(&self.model_dir).unwrap_or_else(|_| {
            // Fallback empty catalog; expert() will error clearly.
            SafetensorCatalog {
                root: self.model_dir.clone(),
                weight_map: HashMap::new(),
                open: HashMap::new(),
            }
        });
        let cap = self.cache.lock().map(|c| c.cap).unwrap_or(32);
        Self {
            model_dir: self.model_dir.clone(),
            layer: self.layer,
            gate: self.gate.clone(),
            n_experts: self.n_experts,
            top_k: self.top_k,
            route_scale: self.route_scale,
            hidden: self.hidden,
            intermediate: self.intermediate,
            catalog: std::sync::Mutex::new(catalog),
            cache: std::sync::Mutex::new(ExpertLru::new(cap)),
        }
    }
}

impl Layer0RoutedMoe {
    /// Softmax gate → top-k (id, weight) pairs.
    pub fn route(&self, h: &[f32]) -> Vec<(usize, f32)> {
        let n = self.n_experts.min(self.gate.rows());
        let mut logits = vec![0.0f32; n];
        for i in 0..n {
            let row = &self.gate.data[i * self.hidden..(i + 1) * self.hidden];
            let mut s = 0.0f32;
            for (a, b) in row.iter().zip(h.iter()) {
                s += a * b;
            }
            logits[i] = s;
        }
        let k = self.top_k.min(n).max(1);
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            logits[b]
                .partial_cmp(&logits[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.truncate(k);
        let max_l = idx
            .iter()
            .map(|&i| logits[i])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut weights: Vec<f32> = idx.iter().map(|&i| (logits[i] - max_l).exp()).collect();
        let sum: f32 = weights.iter().sum::<f32>().max(1e-9);
        for w in &mut weights {
            *w /= sum;
        }
        idx.into_iter().zip(weights).collect()
    }

    /// Cache stats `(hits, misses)` for diagnostics.
    pub fn cache_stats(&self) -> (u64, u64) {
        self.cache
            .lock()
            .map(|c| (c.hits, c.misses))
            .unwrap_or((0, 0))
    }

    /// Load one packed expert (LRU + kept-open catalog).
    ///
    /// Returns `Arc` so lazy f32 expand (`OnceLock`) is shared across uses.
    pub fn expert(&self, id: usize) -> Result<std::sync::Arc<ExpertPacked>> {
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(e) = cache.get(id) {
                return Ok(e);
            }
        }
        let e = {
            let mut cat = self.catalog.lock().unwrap_or_else(|e| e.into_inner());
            std::sync::Arc::new(load_fp4_expert_packed(&mut cat, self.layer, id)?)
        };
        {
            let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
            cache.put(id, std::sync::Arc::clone(&e));
        }
        Ok(e)
    }
}

fn load_fp4_expert_packed(
    cat: &mut SafetensorCatalog,
    layer: usize,
    id: usize,
) -> Result<ExpertPacked> {
    let w1 = cat.load_fp4_packed(&format!("layers.{layer}.ffn.experts.{id}.w1.weight"), 32)?;
    let w2 = cat.load_fp4_packed(&format!("layers.{layer}.ffn.experts.{id}.w2.weight"), 32)?;
    let w3 = cat.load_fp4_packed(&format!("layers.{layer}.ffn.experts.{id}.w3.weight"), 32)?;
    Ok(ExpertPacked {
        w1,
        w2,
        w3,
        f32_once: std::sync::OnceLock::new(),
    })
}

/// Load routed MoE gate for `layers.{layer}` (packed FP4 experts, lazy LRU).
pub fn load_layer_routed_moe(model_dir: &Path, layer: usize) -> Result<Layer0RoutedMoe> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let gate_name = format!("layers.{layer}.ffn.gate.weight");
    if !cat.has(&gate_name) {
        return Err(TrajectError::Other(format!("{gate_name} not found")));
    }
    let gate = cat.load_f32(&gate_name)?;
    if gate.shape.len() != 2 {
        return Err(TrajectError::Other(format!(
            "gate shape {:?} want [n_experts, hidden]",
            gate.shape
        )));
    }
    let n_experts = gate.rows();
    let hidden = gate.cols();

    // Probe expert 0 (packed only) for intermediate size.
    let e0 = load_fp4_expert_packed(&mut cat, layer, 0)?;
    if e0.w1.logical_cols() != hidden || e0.w3.logical_cols() != hidden {
        return Err(TrajectError::Other(format!(
            "expert0 in_features mismatch: hidden={hidden} w1_cols={} w3_cols={}",
            e0.w1.logical_cols(),
            e0.w3.logical_cols()
        )));
    }
    let intermediate = e0.w1.rows;
    if e0.w2.rows != hidden || e0.w2.logical_cols() != intermediate {
        return Err(TrajectError::Other(format!(
            "expert0 w2 shape [{}, {}] want [{hidden}, {intermediate}]",
            e0.w2.rows,
            e0.w2.logical_cols()
        )));
    }

    let (top_k, route_scale) = {
        use crate::weights::HfModelConfig;
        match HfModelConfig::load(model_dir) {
            Ok(cfg) => (
                cfg.num_experts_per_tok.unwrap_or(6) as usize,
                cfg.routed_scaling_factor.unwrap_or(1.5),
            ),
            Err(_) => (6, 1.5),
        }
    };

    let cache_cap = std::env::var("TRAJECT_MOE_CACHE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32usize)
        .max(1);
    let mut lru = ExpertLru::new(cache_cap);
    lru.put(0, std::sync::Arc::new(e0));

    info!(
        dir = %model_dir.display(),
        n_experts,
        top_k,
        route_scale,
        hidden,
        intermediate,
        layer,
        cache_cap,
        "loaded layer routed MoE gate (packed FP4 experts, fused matvec, catalog kept open)"
    );

    Ok(Layer0RoutedMoe {
        model_dir: model_dir.to_path_buf(),
        layer,
        gate,
        n_experts,
        top_k,
        route_scale,
        hidden,
        intermediate,
        catalog: std::sync::Mutex::new(cat),
        cache: std::sync::Mutex::new(lru),
    })
}

/// Load shared-expert SwiGLU for `layers.{layer}`.
pub fn load_layer_shared_ffn(model_dir: &Path, layer: usize) -> Result<Layer0SharedFfn> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    load_layer_shared_ffn_into(&mut cat, model_dir, layer)
}

fn load_layer_shared_ffn_into(
    mut cat: &mut SafetensorCatalog,
    model_dir: &Path,
    layer: usize,
) -> Result<Layer0SharedFfn> {
    let mut ffn_norm = None;
    for n in [
        format!("layers.{layer}.ffn_norm.weight"),
        format!("model.layers.{layer}.post_attention_layernorm.weight"),
    ] {
        if let Ok(t) = cat.load_f32(&n) {
            ffn_norm = Some(t);
            break;
        }
    }
    let ffn_norm = ffn_norm.ok_or_else(|| {
        TrajectError::Other(format!(
            "layer-{layer} ffn_norm not found in {}",
            model_dir.display()
        ))
    })?;

    let w1n = [
        format!("layers.{layer}.ffn.shared_experts.w1.weight"),
        format!("layers.{layer}.ffn.shared_experts.w1"),
        format!("layers.{layer}.mlp.shared_experts.gate_proj.weight"),
    ];
    let w1r: Vec<&str> = w1n.iter().map(|s| s.as_str()).collect();
    let w1 = load_weight_linear(&mut cat, &w1r)?;
    let w2n = [
        format!("layers.{layer}.ffn.shared_experts.w2.weight"),
        format!("layers.{layer}.ffn.shared_experts.w2"),
        format!("layers.{layer}.mlp.shared_experts.down_proj.weight"),
    ];
    let w2r: Vec<&str> = w2n.iter().map(|s| s.as_str()).collect();
    let w2 = load_weight_linear(&mut cat, &w2r)?;
    let w3n = [
        format!("layers.{layer}.ffn.shared_experts.w3.weight"),
        format!("layers.{layer}.ffn.shared_experts.w3"),
        format!("layers.{layer}.mlp.shared_experts.up_proj.weight"),
    ];
    let w3r: Vec<&str> = w3n.iter().map(|s| s.as_str()).collect();
    let w3 = load_weight_linear(&mut cat, &w3r)?;

    let hidden = ffn_norm.data.len();
    let intermediate = w1.rows();
    if w1.cols() != hidden || w3.cols() != hidden || w3.rows() != intermediate {
        return Err(TrajectError::Other(format!(
            "shared ffn shape mismatch: hidden={hidden} w1=[{},{}] w3=[{},{}]",
            w1.rows(),
            w1.cols(),
            w3.rows(),
            w3.cols()
        )));
    }
    if w2.rows() != hidden || w2.cols() != intermediate {
        return Err(TrajectError::Other(format!(
            "shared ffn w2 shape [{}, {}] want [{hidden}, {intermediate}]",
            w2.rows(),
            w2.cols()
        )));
    }

    info!(
        dir = %model_dir.display(),
        hidden,
        intermediate,
        layer,
        packed_fp8 = w1.is_fp8(),
        "loaded layer shared expert FFN"
    );

    Ok(Layer0SharedFfn {
        ffn_norm: ffn_norm.data,
        hidden,
        intermediate,
        w1,
        w2,
        w3,
    })
}

/// Backward-compatible layer-0 loaders.
pub fn load_layer0_attn(model_dir: &Path) -> Result<Layer0AttnWeights> {
    load_layer_attn(model_dir, 0)
}
pub fn load_layer0_shared_ffn(model_dir: &Path) -> Result<Layer0SharedFfn> {
    load_layer_shared_ffn(model_dir, 0)
}
pub fn load_layer0_routed_moe(model_dir: &Path) -> Result<Layer0RoutedMoe> {
    load_layer_routed_moe(model_dir, 0)
}

/// Hyper-Connection weights for one residual branch (attn or ffn).
#[derive(Debug, Clone)]
pub struct HcBranchWeights {
    /// `[mix_hc, hc_mult * hidden]` — linear on flattened multi-stream state.
    pub fn_w: TensorF32,
    /// `[mix_hc]` bias inside sigmoid / comb.
    pub base: Vec<f32>,
    /// `[3]` scales for pre / post / comb.
    pub scale: Vec<f32>,
}

/// Per-layer Hyper-Connections (DeepSeek-V4 Block).
#[derive(Debug, Clone)]
pub struct LayerHcWeights {
    pub hc_mult: usize,
    pub hidden: usize,
    pub sinkhorn_iters: usize,
    pub eps: f32,
    pub norm_eps: f32,
    pub attn: HcBranchWeights,
    pub ffn: HcBranchWeights,
}

impl LayerHcWeights {
    pub fn mix_hc(&self) -> usize {
        (2 + self.hc_mult) * self.hc_mult
    }
    pub fn hc_dim(&self) -> usize {
        self.hc_mult.saturating_mul(self.hidden)
    }
}

/// Final HC collapse before lm_head (`hc_head_*`).
#[derive(Debug, Clone)]
pub struct HcHeadWeights {
    pub hc_mult: usize,
    pub hidden: usize,
    /// `[hc_mult, hc_mult * hidden]`
    pub fn_w: TensorF32,
    pub base: Vec<f32>,
    pub scale: f32,
    pub eps: f32,
}

/// Load layer HC parameters (soft-fail if missing).
pub fn load_layer_hc(model_dir: &Path, layer: usize) -> Result<LayerHcWeights> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    load_layer_hc_into(&mut cat, model_dir, layer)
}

fn load_layer_hc_into(
    cat: &mut SafetensorCatalog,
    model_dir: &Path,
    layer: usize,
) -> Result<LayerHcWeights> {
    let (hc_mult, sinkhorn_iters, eps, norm_eps) = {
        use crate::weights::HfModelConfig;
        match HfModelConfig::load(model_dir) {
            Ok(cfg) => (
                cfg.hc_mult.unwrap_or(4) as usize,
                cfg.hc_sinkhorn_iters.unwrap_or(20) as usize,
                cfg.hc_eps.unwrap_or(1e-6),
                1e-6f32,
            ),
            Err(_) => (4, 20, 1e-6, 1e-6),
        }
    };
    let mix_hc = (2 + hc_mult) * hc_mult;
    // Infer hidden from hc_attn_fn cols / hc_mult.
    let attn_fn = cat.load_f32(&format!("layers.{layer}.hc_attn_fn"))?;
    if attn_fn.shape.len() != 2 || attn_fn.rows() != mix_hc {
        return Err(TrajectError::Other(format!(
            "layer-{layer} hc_attn_fn shape {:?} want [{mix_hc}, hc*hidden]",
            attn_fn.shape
        )));
    }
    let hc_dim = attn_fn.cols();
    if hc_dim % hc_mult != 0 {
        return Err(TrajectError::Other(format!(
            "layer-{layer} hc_dim {hc_dim} not divisible by hc_mult {hc_mult}"
        )));
    }
    let hidden = hc_dim / hc_mult;
    let attn_base = cat.load_f32(&format!("layers.{layer}.hc_attn_base"))?.data;
    let attn_scale = cat.load_f32(&format!("layers.{layer}.hc_attn_scale"))?.data;
    let ffn_fn = cat.load_f32(&format!("layers.{layer}.hc_ffn_fn"))?;
    let ffn_base = cat.load_f32(&format!("layers.{layer}.hc_ffn_base"))?.data;
    let ffn_scale = cat.load_f32(&format!("layers.{layer}.hc_ffn_scale"))?.data;
    if attn_base.len() != mix_hc || attn_scale.len() != 3 {
        return Err(TrajectError::Other(format!(
            "layer-{layer} hc_attn base/scale len base={} scale={}",
            attn_base.len(),
            attn_scale.len()
        )));
    }
    if ffn_fn.shape != attn_fn.shape || ffn_base.len() != mix_hc || ffn_scale.len() != 3 {
        return Err(TrajectError::Other(format!(
            "layer-{layer} hc_ffn shape mismatch fn={:?} base={} scale={}",
            ffn_fn.shape,
            ffn_base.len(),
            ffn_scale.len()
        )));
    }
    info!(
        layer,
        hc_mult, hidden, mix_hc, sinkhorn_iters, "loaded layer Hyper-Connection weights"
    );
    Ok(LayerHcWeights {
        hc_mult,
        hidden,
        sinkhorn_iters,
        eps,
        norm_eps,
        attn: HcBranchWeights {
            fn_w: attn_fn,
            base: attn_base,
            scale: attn_scale,
        },
        ffn: HcBranchWeights {
            fn_w: ffn_fn,
            base: ffn_base,
            scale: ffn_scale,
        },
    })
}

/// Load final HC head collapse weights.
pub fn load_hc_head(model_dir: &Path) -> Result<HcHeadWeights> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let (hc_mult, eps) = {
        use crate::weights::HfModelConfig;
        match HfModelConfig::load(model_dir) {
            Ok(cfg) => (
                cfg.hc_mult.unwrap_or(4) as usize,
                cfg.hc_eps.unwrap_or(1e-6),
            ),
            Err(_) => (4, 1e-6),
        }
    };
    let fn_w = cat.load_f32("hc_head_fn")?;
    let base = cat.load_f32("hc_head_base")?.data;
    let scale_t = cat.load_f32("hc_head_scale")?;
    if fn_w.shape.len() != 2 || fn_w.rows() != hc_mult {
        return Err(TrajectError::Other(format!(
            "hc_head_fn shape {:?} want [{hc_mult}, hc*hidden]",
            fn_w.shape
        )));
    }
    let hc_dim = fn_w.cols();
    if hc_dim % hc_mult != 0 {
        return Err(TrajectError::Other(format!(
            "hc_head_fn cols {hc_dim} not divisible by hc_mult {hc_mult}"
        )));
    }
    let hidden = hc_dim / hc_mult;
    if base.len() != hc_mult {
        return Err(TrajectError::Other(format!(
            "hc_head_base len {} != {hc_mult}",
            base.len()
        )));
    }
    let scale = scale_t.data.first().copied().unwrap_or(1.0);
    info!(hc_mult, hidden, scale, "loaded HC head collapse weights");
    Ok(HcHeadWeights {
        hc_mult,
        hidden,
        fn_w,
        base,
        scale,
        eps,
    })
}

/// Load the first `n_layers` transformer blocks (attn + shared FFN + routed MoE + HC).
///
/// Reuses one [`SafetensorCatalog`] for dense attn/FFN/HC tensors. Each routed MoE
/// layer still opens its own catalog (kept live for lazy expert loads).
pub fn load_layer_stack(model_dir: &Path, n_layers: usize) -> Result<Vec<LayerBlock>> {
    let n = n_layers.max(1);
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let mut out = Vec::with_capacity(n);
    for layer in 0..n {
        let attn = load_layer_attn_into(&mut cat, model_dir, layer)?;
        let ffn = load_layer_shared_ffn_into(&mut cat, model_dir, layer).ok();
        // MoE keeps its own catalog for expert LRU; separate open is intentional.
        let moe = load_layer_routed_moe(model_dir, layer).ok();
        let hc = match load_layer_hc_into(&mut cat, model_dir, layer) {
            Ok(h) => Some(h),
            Err(e) => {
                warn!(layer, error = %e, "layer HC weights missing; simple residual");
                None
            }
        };
        if ffn.is_none() {
            warn!(layer, "shared FFN missing for layer");
        }
        if moe.is_none() {
            warn!(layer, "routed MoE missing for layer");
        }
        out.push(LayerBlock {
            layer,
            attn,
            ffn,
            moe,
            hc,
        });
    }
    info!(
        n_layers = n,
        dir = %model_dir.display(),
        shared_catalog = true,
        "loaded layer stack"
    );
    Ok(out)
}

/// One transformer block (attn + optional FFN/MoE + HC).
#[derive(Debug, Clone)]
pub struct LayerBlock {
    pub layer: usize,
    pub attn: Layer0AttnWeights,
    pub ffn: Option<Layer0SharedFfn>,
    pub moe: Option<Layer0RoutedMoe>,
    pub hc: Option<LayerHcWeights>,
}

/// Load the tensors needed for LocalWeightRunner from a HF model dir.
///
/// Supports DeepSeek-V4 naming (`embed.weight`, `head.weight`, `norm.weight`)
/// and common HF names (`model.embed_tokens.weight`, `lm_head.weight`).
pub fn load_embed_head_norm(
    model_dir: &Path,
) -> Result<(TensorF32, TensorF32, Option<TensorF32>, String)> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let embed_names = [
        "embed.weight",
        "model.embed_tokens.weight",
        "transformer.wte.weight",
        "tok_embeddings.weight",
    ];
    let head_names = [
        "head.weight",
        "lm_head.weight",
        "output.weight",
        "model.embed_tokens.weight", // tied
    ];
    let norm_names = [
        "norm.weight",
        "model.norm.weight",
        "transformer.ln_f.weight",
    ];

    let mut embed = None;
    let mut embed_key = String::new();
    for n in embed_names {
        if let Ok(t) = cat.load_f32(n) {
            embed_key = n.into();
            embed = Some(t);
            break;
        }
    }
    let embed = embed.ok_or_else(|| {
        TrajectError::Other(format!(
            "no embedding tensor found in {} (tried {embed_names:?})",
            model_dir.display()
        ))
    })?;

    let mut head = None;
    for n in head_names {
        if n == embed_key {
            // tied embeddings
            head = Some(embed.clone());
            break;
        }
        if let Ok(t) = cat.load_f32(n) {
            head = Some(t);
            break;
        }
    }
    let head = head.ok_or_else(|| {
        TrajectError::Other(format!(
            "no lm head tensor found in {} (tried {head_names:?})",
            model_dir.display()
        ))
    })?;

    let mut norm = None;
    for n in norm_names {
        if let Ok(t) = cat.load_f32(n) {
            norm = Some(t);
            break;
        }
    }

    Ok((embed, head, norm, embed_key))
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{serialize, TensorView};
    use safetensors::Dtype;

    #[test]
    fn load_single_file_roundtrip() {
        let dir = tempfile_dir();
        // tiny embed 4x3 f32
        let data: Vec<u8> = (0..12u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
        let tensor = TensorView::new(Dtype::F32, vec![4, 3], &data).unwrap();
        let mut tensors = std::collections::BTreeMap::new();
        tensors.insert("embed.weight".to_string(), tensor);
        let tensor2 = TensorView::new(Dtype::F32, vec![4, 3], &data).unwrap();
        tensors.insert("head.weight".to_string(), tensor2);
        let bytes = serialize(&tensors, None).unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();

        let (embed, head, _norm, _) = load_embed_head_norm(&dir).unwrap();
        assert_eq!(embed.shape, vec![4, 3]);
        assert_eq!(head.shape, vec![4, 3]);
        assert!((embed.data[0] - 0.0).abs() < 1e-5);
        assert!((embed.data[1] - 1.0).abs() < 1e-5);
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "traject-st-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
