//! Spawn and supervise a co-located sglang-lite GPU engine process.
//!
//! Traject owns the engine lifecycle: start torchrun → wait `/readyz` → serve
//! Generate steps over the native engine API. This is the Phase-1 form of
//! "in-process GPU": runtime and inference share a host and are managed as one
//! system; FlashInfer/custom kernels can later replace the Python engine
//! without changing Trajectory scheduling.

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use traject_core::{Result, TrajectError};

use crate::backend::SglangLiteEngineBackend;

#[derive(Debug, Clone)]
pub struct LocalEngineConfig {
    pub model: PathBuf,
    pub converted: Option<PathBuf>,
    pub python: PathBuf,
    pub host: String,
    pub port: u16,
    pub tp: u32,
    pub cuda_home: Option<PathBuf>,
    pub working_dir: Option<PathBuf>,
}

impl Default for LocalEngineConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::from("/home/bodesi/models/ds-v4-flash"),
            converted: Some(PathBuf::from("/tmp/ds-v4-mp8")),
            python: PathBuf::from("/home/bodesi/sglang-dflash-venv/bin/python"),
            host: "127.0.0.1".into(),
            port: 9001,
            tp: 8,
            cuda_home: Some(PathBuf::from("/usr/local/cuda")),
            working_dir: Some(PathBuf::from("/home/bodesi/project/sglang-lite")),
        }
    }
}

impl LocalEngineConfig {
    pub fn engine_url(&self) -> String {
        format!("http://{}:{}", self.host, self.port)
    }
}

pub struct LocalEngineHandle {
    child: Option<Child>,
    pub config: LocalEngineConfig,
    pub backend: SglangLiteEngineBackend,
    log_path: PathBuf,
}

impl LocalEngineHandle {
    /// Attach to an already-running engine (no spawn).
    pub fn attach(config: LocalEngineConfig) -> Self {
        let backend =
            SglangLiteEngineBackend::new(config.engine_url(), config.model.display().to_string());
        Self {
            child: None,
            log_path: PathBuf::from(format!("/tmp/traject-logs/engine-{}.log", config.port)),
            config,
            backend,
        }
    }

    /// Spawn `torchrun -m sglang_lite.process` and return a handle.
    pub fn spawn(config: LocalEngineConfig) -> Result<Self> {
        std::fs::create_dir_all("/tmp/traject-logs")
            .map_err(|e| TrajectError::Other(format!("mkdir logs: {e}")))?;
        let log_path = PathBuf::from(format!("/tmp/traject-logs/engine-{}.log", config.port));
        let log_file = std::fs::File::create(&log_path)
            .map_err(|e| TrajectError::Other(format!("open log: {e}")))?;

        let torchrun = config
            .python
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("torchrun");

        let mut cmd = Command::new(&torchrun);
        cmd.arg(format!("--nproc-per-node={}", config.tp))
            .arg("-m")
            .arg("sglang_lite.process")
            .arg("--model")
            .arg(&config.model)
            .arg("--device")
            .arg("cuda")
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--host")
            .arg(&config.host)
            .stdout(Stdio::from(log_file.try_clone().map_err(|e| {
                TrajectError::Other(format!("clone log: {e}"))
            })?))
            .stderr(Stdio::from(log_file))
            .env(
                "SGLANG_LITE_DSV4_HF",
                config.model.display().to_string(),
            );

        if let Some(converted) = &config.converted {
            cmd.env(
                "SGLANG_LITE_DSV4_CONVERTED",
                converted.display().to_string(),
            );
        }
        if let Some(cuda) = &config.cuda_home {
            let bin = cuda.join("bin");
            let include = cuda.join("include");
            let path = std::env::var("PATH").unwrap_or_default();
            cmd.env("PATH", format!("{}:{path}", bin.display()));
            cmd.env("CPATH", include.display().to_string());
        }
        if let Some(cwd) = &config.working_dir {
            cmd.current_dir(cwd);
        }

        let child = cmd
            .spawn()
            .map_err(|e| TrajectError::Inference(format!("failed to spawn engine: {e}")))?;

        let backend =
            SglangLiteEngineBackend::new(config.engine_url(), config.model.display().to_string());
        Ok(Self {
            child: Some(child),
            config,
            backend,
            log_path,
        })
    }

    pub async fn wait_ready(&self, timeout: Duration) -> Result<()> {
        self.backend.wait_ready(timeout).await
    }

    pub fn log_path(&self) -> &Path {
        &self.log_path
    }

    pub fn into_backend(mut self) -> SglangLiteEngineBackend {
        // Detach child so Drop does not kill the engine.
        if let Some(child) = self.child.take() {
            std::mem::forget(child);
        }
        self.backend.clone()
    }
}

impl Drop for LocalEngineHandle {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl LocalEngineHandle {
    pub fn shutdown(&mut self) -> Result<()> {
        if let Some(mut child) = self.child.take() {
            let _ = child.kill();
            let _ = child.wait();
        }
        Ok(())
    }
}
