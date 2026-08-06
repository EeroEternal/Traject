//! sglang-lite-serving — official standalone inference entry.
//!
//! ```text
//! sglang-lite-serving serve --model <moe> --device cuda --port 8000
//! ```

use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use sglang_lite_control::{EngineClient, StubEngineClient};
use tracing::info;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;

#[derive(Parser, Debug)]
#[command(name = "sglang-lite-serving")]
#[command(about = "Official standalone MoE inference server for sglang-lite")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Start the OpenAI-compatible control plane + Python engine process
    Serve {
        /// MoE model id (or fixture:/path). Required unless --stub.
        #[arg(long)]
        model: Option<String>,

        #[arg(long, default_value = "cuda")]
        device: String,

        #[arg(long, default_value_t = 8000)]
        port: u16,

        /// Internal Python engine HTTP port
        #[arg(long, default_value_t = 9001)]
        engine_port: u16,

        /// Use in-process stub engine (no Python / no real model)
        #[arg(long)]
        stub: bool,

        /// Do not spawn Python; connect to an already-running engine at this URL
        #[arg(long)]
        engine_url: Option<String>,

        /// Spawn engine via ``torchrun --nproc-per-node=<tp>`` (DeepSeek-V4 Hybrid).
        /// When set, ``python -m sglang_lite.process`` is not used directly.
        #[arg(long)]
        tp: Option<u16>,
    },
}

struct EngineChild {
    child: Child,
}

impl Drop for EngineChild {
    fn drop(&mut self) {
        // Prefer graceful SIGTERM so the engine can stop accepting work.
        #[cfg(unix)]
        {
            let _ = Command::new("kill")
                .args(["-TERM", &self.child.id().to_string()])
                .status();
            for _ in 0..50 {
                if let Ok(Some(_)) = self.child.try_wait() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
        }
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn spawn_python_engine(
    model: &str,
    device: &str,
    engine_port: u16,
    tp: Option<u16>,
) -> Result<EngineChild> {
    let mut cmd = if let Some(n) = tp {
        if n < 2 {
            return Err(anyhow!("--tp must be >= 2 (got {})", n));
        }
        let mut c = Command::new("torchrun");
        c.arg(format!("--nproc-per-node={}", n))
            .arg("-m")
            .arg("sglang_lite.process");
        c
    } else {
        let mut c = Command::new("python");
        c.arg("-m").arg("sglang_lite.process");
        c
    };

    cmd.arg("--model")
        .arg(model)
        .arg("--device")
        .arg(device)
        .arg("--port")
        .arg(engine_port.to_string())
        .arg("--host")
        .arg("127.0.0.1")
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());

    if model == "stub" {
        cmd.arg("--allow-stub");
    }

    let child = cmd.spawn().with_context(|| {
        if tp.is_some() {
            "failed to spawn torchrun -m sglang_lite.process"
        } else {
            "failed to spawn python -m sglang_lite.process (is sglang-lite installed?)"
        }
    })?;
    Ok(EngineChild { child })
}

async fn wait_engine_ready(url: &str, timeout: Duration) -> Result<()> {
    let client = reqwest::Client::builder()
        .no_proxy()
        .build()
        .context("build reqwest client")?;
    let readyz = format!("{}/readyz", url.trim_end_matches('/'));
    let start = std::time::Instant::now();
    loop {
        if start.elapsed() > timeout {
            return Err(anyhow!("engine not ready within {:?}", timeout));
        }
        if let Ok(resp) = client.get(&readyz).send().await {
            if resp.status().is_success() {
                return Ok(());
            }
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "sglang_lite=info,tower_http=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    match cli.command {
        Commands::Serve {
            model,
            device,
            port,
            engine_port,
            stub,
            engine_url,
            tp,
        } => run_serve(model, device, port, engine_port, stub, engine_url, tp).await,
    }
}

async fn run_serve(
    model: Option<String>,
    device: String,
    port: u16,
    engine_port: u16,
    stub: bool,
    engine_url: Option<String>,
    tp: Option<u16>,
) -> Result<()> {
    let ready = Arc::new(AtomicBool::new(false));
    let draining = Arc::new(AtomicBool::new(false));

    let (engine, core_url, model_id, _child_guard): (
        EngineClient,
        Option<String>,
        String,
        Option<EngineChild>,
    ) = if stub {
        info!("starting in --stub mode (no real MoE model)");
        (
            EngineClient::Stub(StubEngineClient::new()),
            None,
            model.unwrap_or_else(|| "stub".into()),
            None,
        )
    } else if let Some(url) = engine_url {
        let model_id = model.ok_or_else(|| anyhow!("--model is required with --engine-url"))?;
        info!("using external engine at {}", url);
        wait_engine_ready(&url, Duration::from_secs(120)).await?;
        (EngineClient::http(url.clone()), Some(url), model_id, None)
    } else {
        let model_id = model.ok_or_else(|| anyhow!("--model is required (or pass --stub)"))?;
        let url = format!("http://127.0.0.1:{}", engine_port);
        info!(
            "spawning Python engine model={} device={} port={} tp={:?}",
            model_id, device, engine_port, tp
        );
        let child = spawn_python_engine(&model_id, &device, engine_port, tp)?;
        let ready_timeout = if tp.is_some() {
            Duration::from_secs(1800)
        } else {
            Duration::from_secs(600)
        };
        wait_engine_ready(&url, ready_timeout).await?;
        (
            EngineClient::http(url.clone()),
            Some(url),
            model_id,
            Some(child),
        )
    };

    // Only the actually loaded model is advertised — no alias remapping.
    let supported_models = vec![model_id.clone()];

    ready.store(true, Ordering::Relaxed);

    let ready_bg = ready.clone();
    let draining_bg = draining.clone();
    tokio::spawn(async move {
        let _ = tokio::signal::ctrl_c().await;
        info!("SIGINT: draining…");
        draining_bg.store(true, Ordering::Relaxed);
        ready_bg.store(false, Ordering::Relaxed);
    });

    info!("sglang-lite-serving ready on port {}", port);
    info!("OpenAI: POST http://127.0.0.1:{}/v1/chat/completions", port);
    info!("readyz: GET http://127.0.0.1:{}/readyz", port);

    sglang_lite_control::serve(
        Arc::new(engine),
        Arc::new(supported_models),
        core_url,
        port,
        ready,
        draining,
    )
    .await
}
