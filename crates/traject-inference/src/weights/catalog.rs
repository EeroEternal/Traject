//! Sharded HuggingFace safetensors catalog (index.json + per-file mmap).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use safetensors::tensor::SafeTensors;
use serde::Deserialize;
use tracing::info;
use traject_core::{Result, TrajectError};

use super::dtype::bytes_to_f32_vec;

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

    /// Load a tensor as contiguous f32 (BF16/F16/F32).
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

    pub fn has(&self, name: &str) -> bool {
        self.weight_map.contains_key(name) || self.shard_for(name).is_ok()
    }
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
        let bytes = serialize(&tensors, &None).unwrap();
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
