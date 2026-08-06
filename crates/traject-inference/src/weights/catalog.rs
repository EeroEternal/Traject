//! Sharded HuggingFace safetensors catalog (index.json + per-file mmap).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use serde::Deserialize;
use tracing::{info, warn};
use traject_core::{Result, TrajectError};

use super::dtype::{bytes_to_f32_vec, dequant_fp8_block_scaled};

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
            let raw = std::fs::read_to_string(&index_path).map_err(|e| {
                TrajectError::Other(format!("read index: {e}"))
            })?;
            let idx: IndexFile = serde_json::from_str(&raw).map_err(|e| {
                TrajectError::Other(format!("parse index: {e}"))
            })?;
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
        let entries = std::fs::read_dir(&self.root).map_err(|e| {
            TrajectError::Other(format!("read dir: {e}"))
        })?;
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
            let file = File::open(shard).map_err(|e| {
                TrajectError::Other(format!("open {}: {e}", shard.display()))
            })?;
            let mmap = unsafe { Mmap::map(&file) }.map_err(|e| {
                TrajectError::Other(format!("mmap {}: {e}", shard.display()))
            })?;
            self.open.insert(key.clone(), mmap);
        }
        Ok(self.open.get(&key).expect("just inserted"))
    }

    /// Load a tensor as contiguous f32 (BF16/F16/F32/F8_* unscaled).
    pub fn load_f32(&mut self, name: &str) -> Result<TensorF32> {
        let shard = self.shard_for(name)?;
        let mmap = self.ensure_mmap(&shard)?;
        let st = SafeTensors::deserialize(mmap).map_err(|e| {
            TrajectError::Other(format!("deserialize {}: {e}", shard.display()))
        })?;
        let view = st.tensor(name).map_err(|e| {
            TrajectError::Other(format!("tensor `{name}`: {e}"))
        })?;
        let shape: Vec<usize> = view.shape().to_vec();
        let data = bytes_to_f32_vec(view.data(), view.dtype()).map_err(|e| {
            TrajectError::Other(format!("decode `{name}`: {e}"))
        })?;
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

    fn load_raw(&mut self, name: &str) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
        let shard = self.shard_for(name)?;
        let mmap = self.ensure_mmap(&shard)?;
        let st = SafeTensors::deserialize(mmap).map_err(|e| {
            TrajectError::Other(format!("deserialize {}: {e}", shard.display()))
        })?;
        let view = st.tensor(name).map_err(|e| {
            TrajectError::Other(format!("tensor `{name}`: {e}"))
        })?;
        Ok((view.data().to_vec(), view.shape().to_vec(), view.dtype()))
    }

    pub fn has(&self, name: &str) -> bool {
        self.weight_map.contains_key(name)
            || {
                // avoid recursive error noise on missing single-file fallback
                self.shard_for(name).is_ok()
            }
    }
}

/// Layer-0 attention projections (DeepSeek-V4 MLA compressed + optional Q expand).
///
/// Routed MoE experts / full `wo_*` output are **not** loaded here.
#[derive(Debug, Clone)]
pub struct Layer0AttnWeights {
    /// RMSNorm γ before attention, shape [hidden].
    pub attn_norm: Vec<f32>,
    pub hidden: usize,
    /// `wq_a`: [q_lora, hidden] — Q down-projection.
    pub wq_a: TensorF32,
    /// `wkv`: [kv_lora, hidden] — compressed KV projection.
    pub wkv: TensorF32,
    /// Optional RMSNorm on q_lora (after `wq_a`).
    pub q_norm: Option<Vec<f32>>,
    /// Optional RMSNorm on kv_lora (after `wkv`).
    pub kv_norm: Option<Vec<f32>>,
    /// Optional `wq_b`: [n_heads * head_dim, q_lora] — expand Q to full heads.
    pub wq_b: Option<TensorF32>,
    /// Head count implied by `wq_b` rows / `kv_lora` (when present).
    pub n_heads: Option<usize>,
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
    /// Full Q width after `wq_b` (e.g. 32768), if loaded.
    pub fn q_full_dim(&self) -> Option<usize> {
        self.wq_b.as_ref().map(|t| t.rows())
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
    pub w1: TensorF32,
    /// [hidden, intermediate]
    pub w2: TensorF32,
    /// [intermediate, hidden]
    pub w3: TensorF32,
}

fn load_weight_fp8_or_f32(cat: &mut SafetensorCatalog, names: &[&str]) -> Result<TensorF32> {
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
        // Prefer FP8 block-scaled when a sibling `.scale` exists.
        let scale = if key.ends_with(".weight") {
            format!("{}.scale", key.trim_end_matches(".weight"))
        } else {
            format!("{key}.scale")
        };
        if cat.has(&scale) {
            if let Ok(t) = cat.load_fp8_block_scaled(&key, 128) {
                return Ok(t);
            }
        }
        if let Ok(t) = cat.load_f32(&key) {
            return Ok(t);
        }
        if let Ok(t) = cat.load_f32(n) {
            return Ok(t);
        }
    }
    Err(TrajectError::Other(format!(
        "none of {names:?} found/loadable"
    )))
}

/// Load layer-0 attention norms + FP8 `wq_a` / `wkv` (block-scaled).
pub fn load_layer0_attn(model_dir: &Path) -> Result<Layer0AttnWeights> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let norm_names = [
        "layers.0.attn_norm.weight",
        "model.layers.0.input_layernorm.weight",
    ];
    let mut attn_norm = None;
    for n in norm_names {
        if let Ok(t) = cat.load_f32(n) {
            attn_norm = Some(t);
            break;
        }
    }
    let attn_norm = attn_norm.ok_or_else(|| {
        TrajectError::Other(format!(
            "layer-0 attn_norm not found in {}",
            model_dir.display()
        ))
    })?;

    let wq_a = load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.attn.wq_a.weight",
            "layers.0.attn.wq_a",
            "model.layers.0.self_attn.q_a_proj.weight",
        ],
    )?;
    let wkv = load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.attn.wkv.weight",
            "layers.0.attn.wkv",
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        ],
    )?;

    if wq_a.shape.len() != 2 || wkv.shape.len() != 2 {
        return Err(TrajectError::Other(format!(
            "layer-0 projections must be 2D, wq_a={:?} wkv={:?}",
            wq_a.shape, wkv.shape
        )));
    }
    let hidden = attn_norm.data.len();
    if wq_a.cols() != hidden || wkv.cols() != hidden {
        return Err(TrajectError::Other(format!(
            "layer-0 in_features mismatch: norm_h={hidden} wq_a={:?} wkv={:?}",
            wq_a.shape, wkv.shape
        )));
    }

    let q_norm = cat
        .load_f32("layers.0.attn.q_norm.weight")
        .ok()
        .map(|t| t.data);
    let kv_norm = cat
        .load_f32("layers.0.attn.kv_norm.weight")
        .ok()
        .map(|t| t.data);

    // Optional Q expand (full MLA heads). ~134MB f32 for V4 Flash.
    let wq_b = match load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.attn.wq_b.weight",
            "layers.0.attn.wq_b",
            "model.layers.0.self_attn.q_b_proj.weight",
        ],
    ) {
        Ok(t) => {
            if t.shape.len() == 2 && t.cols() == wq_a.rows() {
                Some(t)
            } else {
                warn!(
                    shape = ?t.shape,
                    q_lora = wq_a.rows(),
                    "wq_b shape incompatible with wq_a; skipping Q expand"
                );
                None
            }
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

    info!(
        dir = %model_dir.display(),
        hidden,
        q_lora = wq_a.rows(),
        kv_lora = wkv.rows(),
        has_q_norm = q_norm.is_some(),
        has_kv_norm = kv_norm.is_some(),
        has_wq_b = wq_b.is_some(),
        q_full = wq_b.as_ref().map(|t| t.rows()),
        n_heads = ?n_heads,
        "loaded layer-0 attention projections (MLA Q path)"
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
    })
}

/// Load layer-0 shared-expert SwiGLU (`ffn_norm` + `w1`/`w2`/`w3`).
///
/// Does **not** load the 256 routed experts.
pub fn load_layer0_shared_ffn(model_dir: &Path) -> Result<Layer0SharedFfn> {
    let mut cat = SafetensorCatalog::open(model_dir)?;
    let mut ffn_norm = None;
    for n in [
        "layers.0.ffn_norm.weight",
        "model.layers.0.post_attention_layernorm.weight",
    ] {
        if let Ok(t) = cat.load_f32(n) {
            ffn_norm = Some(t);
            break;
        }
    }
    let ffn_norm = ffn_norm.ok_or_else(|| {
        TrajectError::Other(format!(
            "layer-0 ffn_norm not found in {}",
            model_dir.display()
        ))
    })?;

    let w1 = load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.ffn.shared_experts.w1.weight",
            "layers.0.ffn.shared_experts.w1",
            "layers.0.mlp.shared_experts.gate_proj.weight",
        ],
    )?;
    let w2 = load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.ffn.shared_experts.w2.weight",
            "layers.0.ffn.shared_experts.w2",
            "layers.0.mlp.shared_experts.down_proj.weight",
        ],
    )?;
    let w3 = load_weight_fp8_or_f32(
        &mut cat,
        &[
            "layers.0.ffn.shared_experts.w3.weight",
            "layers.0.ffn.shared_experts.w3",
            "layers.0.mlp.shared_experts.up_proj.weight",
        ],
    )?;

    if w1.shape.len() != 2 || w2.shape.len() != 2 || w3.shape.len() != 2 {
        return Err(TrajectError::Other(format!(
            "shared ffn weights must be 2D: w1={:?} w2={:?} w3={:?}",
            w1.shape, w2.shape, w3.shape
        )));
    }
    let hidden = ffn_norm.data.len();
    let intermediate = w1.rows();
    if w1.cols() != hidden || w3.cols() != hidden || w3.rows() != intermediate {
        return Err(TrajectError::Other(format!(
            "shared ffn shape mismatch: hidden={hidden} w1={:?} w3={:?}",
            w1.shape, w3.shape
        )));
    }
    if w2.rows() != hidden || w2.cols() != intermediate {
        return Err(TrajectError::Other(format!(
            "shared ffn w2 shape {:?} want [{hidden}, {intermediate}]",
            w2.shape
        )));
    }

    info!(
        dir = %model_dir.display(),
        hidden,
        intermediate,
        "loaded layer-0 shared expert FFN (no routed MoE)"
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
    let norm_names = ["norm.weight", "model.norm.weight", "transformer.ln_f.weight"];

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
        let data: Vec<u8> = (0..12u32)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect();
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
