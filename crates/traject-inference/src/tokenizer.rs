//! HuggingFace `tokenizer.json` loader for text ↔ token ids.
//!
//! DeepSeek-V4-Flash (and most HF checkpoints) ship a standard
//! `tokenizer.json` that the `tokenizers` crate loads directly.
//! Note: `encoding_dsv4.py` in the model repo is a **chat template**
//! helper, not the BPE vocabulary.

use std::path::{Path, PathBuf};

use tokenizers::Tokenizer;
use tracing::info;
use traject_core::{Result, TrajectError};

/// Thin wrapper around HuggingFace `tokenizers::Tokenizer`.
#[derive(Clone)]
pub struct HfTokenizer {
    inner: Tokenizer,
    path: PathBuf,
    /// Vocab size reported by the tokenizer model (may differ slightly from
    /// embed rows when padding tokens exist).
    vocab_size: u32,
    eos_token_id: Option<u32>,
    bos_token_id: Option<u32>,
}

impl std::fmt::Debug for HfTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HfTokenizer")
            .field("path", &self.path)
            .field("vocab_size", &self.vocab_size)
            .field("eos_token_id", &self.eos_token_id)
            .field("bos_token_id", &self.bos_token_id)
            .finish()
    }
}

impl HfTokenizer {
    /// Load `tokenizer.json` from a HuggingFace model directory.
    ///
    /// Looks for `dir/tokenizer.json`. Optionally reads
    /// `tokenizer_config.json` / `generation_config.json` for special ids.
    pub fn from_model_dir(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let path = dir.join("tokenizer.json");
        if !path.is_file() {
            return Err(TrajectError::Other(format!(
                "tokenizer.json not found at {}",
                path.display()
            )));
        }
        Self::from_file(&path).map(|mut t| {
            // Prefer special ids from companion JSON configs when present.
            if let Some(eos) = read_special_id(dir, "eos_token_id") {
                t.eos_token_id = Some(eos);
            }
            if let Some(bos) = read_special_id(dir, "bos_token_id") {
                t.bos_token_id = Some(bos);
            }
            // Also try tokenizer_config string → id resolution.
            if t.eos_token_id.is_none() {
                if let Some(s) = read_special_token_str(dir, "eos_token") {
                    t.eos_token_id = t.inner.token_to_id(&s);
                }
            }
            if t.bos_token_id.is_none() {
                if let Some(s) = read_special_token_str(dir, "bos_token") {
                    t.bos_token_id = t.inner.token_to_id(&s);
                }
            }
            info!(
                path = %t.path.display(),
                vocab = t.vocab_size,
                eos = ?t.eos_token_id,
                bos = ?t.bos_token_id,
                "loaded HF tokenizer.json"
            );
            t
        })
    }

    /// Load directly from a `tokenizer.json` path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let inner = Tokenizer::from_file(&path).map_err(|e| {
            TrajectError::Other(format!("load tokenizer {}: {e}", path.display()))
        })?;
        let vocab_size = inner.get_vocab_size(true) as u32;
        Ok(Self {
            inner,
            path,
            vocab_size,
            eos_token_id: None,
            bos_token_id: None,
        })
    }

    pub fn vocab_size(&self) -> u32 {
        self.vocab_size
    }

    pub fn eos_token_id(&self) -> Option<u32> {
        self.eos_token_id
    }

    pub fn bos_token_id(&self) -> Option<u32> {
        self.bos_token_id
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Encode text → token ids. `add_special_tokens` mirrors HF default for
    /// single-string encode (usually true for model inputs).
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, add_special_tokens)
            .map_err(|e| TrajectError::Other(format!("tokenizer encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token ids → text. Skips special tokens by default.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| TrajectError::Other(format!("tokenizer decode: {e}")))
    }

    /// Convenience: encode without special tokens (raw piece split).
    pub fn encode_ordinary(&self, text: &str) -> Result<Vec<u32>> {
        self.encode(text, false)
    }
}

fn read_special_id(dir: &Path, key: &str) -> Option<u32> {
    for name in ["generation_config.json", "config.json", "tokenizer_config.json"] {
        let p = dir.join(name);
        let Ok(raw) = std::fs::read_to_string(&p) else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<serde_json::Value>(&raw) else {
            continue;
        };
        if let Some(n) = v.get(key).and_then(|x| x.as_u64()) {
            return Some(n as u32);
        }
        // Some configs use a list for eos_token_id.
        if let Some(arr) = v.get(key).and_then(|x| x.as_array()) {
            if let Some(n) = arr.first().and_then(|x| x.as_u64()) {
                return Some(n as u32);
            }
        }
    }
    None
}

fn read_special_token_str(dir: &Path, key: &str) -> Option<String> {
    let p = dir.join("tokenizer_config.json");
    let raw = std::fs::read_to_string(p).ok()?;
    let v: serde_json::Value = serde_json::from_str(&raw).ok()?;
    match v.get(key)? {
        serde_json::Value::String(s) => Some(s.clone()),
        serde_json::Value::Object(o) => o
            .get("content")
            .and_then(|c| c.as_str())
            .map(|s| s.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal WordLevel tokenizer.json (no external model download).
    const TINY_TOKENIZER_JSON: &str = r#"{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 0, "content": "[UNK]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 1, "content": "[EOS]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {
      "[UNK]": 0,
      "[EOS]": 1,
      "hello": 2,
      "world": 3,
      "你好": 4
    },
    "unk_token": "[UNK]"
  }
}"#;

    fn write_tiny(dir: &Path) {
        std::fs::write(dir.join("tokenizer.json"), TINY_TOKENIZER_JSON).unwrap();
        // Companion config for special-id resolution.
        std::fs::write(
            dir.join("tokenizer_config.json"),
            r#"{"eos_token": "[EOS]", "bos_token": null}"#,
        )
        .unwrap();
    }

    #[test]
    fn load_encode_decode_roundtrip() {
        let dir = tempfile_dir();
        write_tiny(&dir);
        let tok = HfTokenizer::from_model_dir(&dir).unwrap();
        assert!(tok.vocab_size() >= 5);
        assert_eq!(tok.eos_token_id(), Some(1));
        let ids = tok.encode("hello world", false).unwrap();
        assert_eq!(ids, vec![2, 3]);
        let text = tok.decode(&ids, true).unwrap();
        assert!(text.contains("hello"), "got {text:?}");
        assert!(text.contains("world"), "got {text:?}");
    }

    #[test]
    fn missing_file_errors() {
        let dir = tempfile_dir();
        let err = HfTokenizer::from_model_dir(&dir).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("tokenizer.json"), "{msg}");
    }

    /// Set `TRAJECT_TEST_TOKENIZER_DIR` to a HF model dir with `tokenizer.json`
    /// (e.g. DeepSeek-V4-Flash) to exercise the real BPE path.
    #[test]
    fn real_deepseek_tokenizer_if_present() {
        let Some(dir) = std::env::var_os("TRAJECT_TEST_TOKENIZER_DIR") else {
            eprintln!("skip: TRAJECT_TEST_TOKENIZER_DIR not set");
            return;
        };
        let dir = PathBuf::from(dir);
        if !dir.join("tokenizer.json").is_file() {
            eprintln!("skip: no tokenizer.json under {}", dir.display());
            return;
        }
        let tok = HfTokenizer::from_model_dir(&dir).expect("load real tokenizer");
        assert!(tok.vocab_size() >= 100_000, "vocab={}", tok.vocab_size());
        let ids = tok
            .encode("hello 你好", false)
            .expect("encode");
        // Matches HF AutoTokenizer on DeepSeek-V4-Flash: [33310, 223, 30594]
        assert_eq!(ids, vec![33310, 223, 30594], "ids={ids:?}");
        let text = tok.decode(&ids, true).expect("decode");
        assert!(
            text.contains("hello") && text.contains("你好"),
            "decoded={text:?}"
        );
    }

    fn tempfile_dir() -> PathBuf {
        let mut d = std::env::temp_dir();
        d.push(format!(
            "traject-tok-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }
}
