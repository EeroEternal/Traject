//! In-process FlashInfer via embedded CPython (same OS process as Traject).
//!
//! Requires the `flashinfer` cargo feature and a venv where `flashinfer` + `torch`
//! are importable (set PYTHONPATH / use sglang-dflash-venv).

use std::sync::Once;

use async_trait::async_trait;
use pyo3::prelude::*;
use pyo3::types::PyModule;
use traject_core::{Result, TrajectError};

use super::{
    DecodeRequest, DecodeResult, KernelBackend, PrefillRequest, PrefillResult, SampleRequest,
    SampleResult,
};

static PYTHON_INIT: Once = Once::new();

fn ensure_python(site_packages: Option<&str>) -> Result<()> {
    PYTHON_INIT.call_once(|| {
        pyo3::prepare_freethreaded_python();
    });
    Python::with_gil(|py| {
        if let Some(sp) = site_packages {
            let code = format!(
                "import sys\np = {sp:?}\nif p not in sys.path:\n    sys.path.insert(0, p)\n"
            );
            py.run_bound(&code, None, None).map_err(py_err)?;
        }
        py.import_bound("flashinfer").map_err(|e| {
            TrajectError::Inference(format!(
                "flashinfer import failed ({e}); use sglang-dflash-venv site-packages"
            ))
        })?;
        py.import_bound("torch").map_err(py_err)?;
        Ok(())
    })
}

fn py_err(e: PyErr) -> TrajectError {
    TrajectError::Inference(format!("python: {e}"))
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
            site_packages: Some(
                "/home/bodesi/sglang-dflash-venv/lib/python3.10/site-packages".into(),
            ),
            device: "cuda:0".into(),
        }
    }
}

pub struct FlashInferKernel {
    config: FlashInferKernelConfig,
}

impl FlashInferKernel {
    pub fn new(config: FlashInferKernelConfig) -> Result<Self> {
        ensure_python(config.site_packages.as_deref())?;
        Ok(Self { config })
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
