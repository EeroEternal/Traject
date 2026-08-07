//! In-process FlashInfer via embedded CPython (same OS process as Traject).
//!
//! Requires the `flashinfer` cargo feature and a venv where `flashinfer` + `torch`
//! are importable. Site-packages discovery order:
//! 1. `FlashInferKernelConfig.site_packages` if set
//! 2. `TRAJECT_FLASHINFER_SITE` / `SGLANG_VENV` env
//! 3. Well-known pro6000 / local paths

use std::path::{Path, PathBuf};
use std::sync::Once;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use tracing::{info, warn};
use traject_core::{Result, TrajectError};

use super::{
    DecodeRequest, DecodeResult, KernelBackend, PrefillRequest, PrefillResult, SampleRequest,
    SampleResult,
};

static PYTHON_INIT: Once = Once::new();

/// Resolve venv `bin/` from a site-packages path.
fn venv_bin_from_site(site_packages: &str) -> Option<PathBuf> {
    // .../lib/pythonX.Y/site-packages → .../bin
    let sp = Path::new(site_packages);
    let bin = sp
        .parent() // pythonX.Y
        .and_then(|p| p.parent()) // lib
        .and_then(|p| p.parent()) // venv root
        .map(|root| root.join("bin"))?;
    bin.is_dir().then_some(bin)
}

/// Prepend venv `bin/` to process PATH so FlashInfer JIT can find `ninja`.
fn prepend_venv_bin_rust(site_packages: &str) -> Option<String> {
    let bin = venv_bin_from_site(site_packages)?;
    let bin_s = bin.display().to_string();
    let path = std::env::var_os("PATH").unwrap_or_default();
    let mut paths = std::env::split_paths(&path).collect::<Vec<_>>();
    if !paths.iter().any(|p| p == &bin) {
        paths.insert(0, bin);
        if let Ok(joined) = std::env::join_paths(paths) {
            // SAFETY: single-process init; PATH only affects this process.
            unsafe { std::env::set_var("PATH", joined) };
        }
    }
    Some(bin_s)
}

fn ensure_python(site_packages: Option<&str>) -> Result<()> {
    PYTHON_INIT.call_once(|| {
        pyo3::prepare_freethreaded_python();
    });
    let bin_s = site_packages.and_then(prepend_venv_bin_rust);
    Python::with_gil(|py| {
        if let Some(sp) = site_packages {
            // Also update Python's os.environ — subprocesses from JIT use that.
            let bin_py = bin_s.as_deref().unwrap_or("");
            let code = format!(
                r#"
import sys, os
p = {sp:?}
if p not in sys.path:
    sys.path.insert(0, p)
bin_dir = {bin_py:?}
if bin_dir:
    path = os.environ.get("PATH", "")
    if bin_dir not in path.split(os.pathsep):
        os.environ["PATH"] = bin_dir + os.pathsep + path
"#
            );
            py.run_bound(&code, None, None).map_err(py_err)?;
            if !bin_py.is_empty() {
                info!(bin = %bin_py, "prepended venv bin to PATH for FlashInfer JIT");
            }
        }
        py.import_bound("flashinfer").map_err(|e| {
            TrajectError::Inference(format!(
                "flashinfer import failed ({e}); set TRAJECT_FLASHINFER_SITE to venv site-packages"
            ))
        })?;
        py.import_bound("torch").map_err(py_err)?;
        Ok(())
    })
}

fn py_err(e: PyErr) -> TrajectError {
    TrajectError::Inference(format!("python: {e}"))
}

/// Candidate site-packages directories (first existing wins).
pub fn discover_site_packages() -> Option<String> {
    if let Ok(p) = std::env::var("TRAJECT_FLASHINFER_SITE") {
        if Path::new(&p).is_dir() {
            return Some(p);
        }
    }
    // SGLANG_VENV=/path/to/venv → site-packages
    if let Ok(venv) = std::env::var("SGLANG_VENV") {
        for rel in [
            "lib/python3.12/site-packages",
            "lib/python3.11/site-packages",
            "lib/python3.10/site-packages",
            "lib/python3.9/site-packages",
        ] {
            let p = PathBuf::from(&venv).join(rel);
            if p.is_dir() {
                return Some(p.display().to_string());
            }
        }
    }
    const CANDIDATES: &[&str] = &[
        "/home/bodesi/venvs/sglang-lite/lib/python3.10/site-packages",
        "/home/bodesi/sglang-dflash-venv/lib/python3.10/site-packages",
        "/home/bodesi/venvs/sglang-lite/lib/python3.12/site-packages",
    ];
    for c in CANDIDATES {
        if Path::new(c).is_dir() {
            return Some((*c).to_string());
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct FlashInferKernelConfig {
    /// Absolute path to venv `site-packages`.
    pub site_packages: Option<String>,
    pub device: String,
}

impl Default for FlashInferKernelConfig {
    fn default() -> Self {
        Self {
            site_packages: discover_site_packages(),
            device: std::env::var("TRAJECT_CUDA_DEVICE").unwrap_or_else(|_| "cuda:0".into()),
        }
    }
}

pub struct FlashInferKernel {
    config: FlashInferKernelConfig,
}

impl FlashInferKernel {
    pub fn new(config: FlashInferKernelConfig) -> Result<Self> {
        let mut config = config;
        if config.site_packages.is_none() {
            config.site_packages = discover_site_packages();
        }
        ensure_python(config.site_packages.as_deref())?;
        info!(
            site = ?config.site_packages,
            device = %config.device,
            "FlashInferKernel ready"
        );
        Ok(Self { config })
    }

    /// Best-effort: load FlashInfer or return `None` (caller falls back to CPU).
    pub fn try_new(config: FlashInferKernelConfig) -> Option<Self> {
        match Self::new(config) {
            Ok(k) => Some(k),
            Err(e) => {
                warn!(error = %e, "FlashInfer unavailable; falling back to CPU kernel");
                None
            }
        }
    }

    pub fn device(&self) -> &str {
        &self.config.device
    }

    fn run_decode_py(&self, req: &DecodeRequest) -> Result<Vec<f32>> {
        Python::with_gil(|py| {
            let code = r#"
import torch
import flashinfer

def decode(q, k, v, seq_len, num_heads, head_dim, device, layout):
    q_t = torch.tensor(q, dtype=torch.float16, device=device).view(num_heads, head_dim)
    k_t = torch.tensor(k, dtype=torch.float16, device=device).view(seq_len, num_heads, head_dim)
    v_t = torch.tensor(v, dtype=torch.float16, device=device).view(seq_len, num_heads, head_dim)
    # flashinfer single_decode expects q: [H, D], k/v: [N, H, D] for NHD
    o = flashinfer.single_decode_with_kv_cache(q_t, k_t, v_t, kv_layout=layout)
    return o.detach().float().cpu().view(-1).tolist()
"#;
            let module = PyModule::from_code_bound(py, code, "traject_flashinfer.py", "traject_flashinfer")
                .map_err(py_err)?;
            let decode = module.getattr("decode").map_err(py_err)?;
            let out: Vec<f32> = decode
                .call1((
                    req.q.clone(),
                    req.k_cache.clone(),
                    req.v_cache.clone(),
                    req.seq_len,
                    req.num_heads,
                    req.head_dim,
                    self.config.device.as_str(),
                    req.layout.as_flashinfer(),
                ))
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            Ok(out)
        })
    }
}

#[async_trait]
impl KernelBackend for FlashInferKernel {
    fn name(&self) -> &str {
        "flashinfer-py"
    }

    async fn prefill(&self, req: PrefillRequest) -> Result<PrefillResult> {
        // Phase-1: fall back to sequential decode for correctness smoke.
        let h = req.num_heads;
        let d = req.head_dim;
        let t = req.num_tokens as usize;
        let mut o = Vec::new();
        for i in 0..t {
            let start = i * (h * d) as usize;
            let end = (i + 1) * (h * d) as usize;
            let dec = DecodeRequest {
                q: req.q[start..end].to_vec(),
                k_cache: req.k[..end].to_vec(),
                v_cache: req.v[..end].to_vec(),
                seq_len: (i + 1) as u32,
                num_heads: h,
                head_dim: d,
                layout: req.layout,
                // FlashInfer single_decode path does not yet consume attn_sink.
                attn_sink: None,
            };
            let r = self.decode(dec).await?;
            o.extend(r.o);
        }
        Ok(PrefillResult { o })
    }

    async fn decode(&self, req: DecodeRequest) -> Result<DecodeResult> {
        let o = self.run_decode_py(&req)?;
        Ok(DecodeResult { o })
    }

    async fn sample(&self, req: SampleRequest) -> Result<SampleResult> {
        // Sampling stays on CPU for now; FlashInfer sampling hooks later.
        crate::kernel::CpuRefKernel.sample(req).await
    }
}
