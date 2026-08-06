use std::path::PathBuf;

use traject_core::TrajectoryConfig;
use traject_inference::StubMode;
use traject_policy::ReActPolicy;
use traject_runtime::{Driver, DriverConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut args = std::env::args().skip(1).collect::<Vec<_>>();

    if args.first().map(|s| s.as_str()) == Some("serve") {
        args.remove(0);
        let upstream = take_flag_value(&mut args, "--upstream");
        let addr: std::net::SocketAddr = args
            .first()
            .and_then(|s| s.parse().ok())
            .unwrap_or_else(|| "0.0.0.0:8080".parse().unwrap());
        if let Some(upstream) = upstream {
            // Tool-bridge: OpenAI tools → text protocol → sglang-lite (no native tools).
            let upstream = upstream.trim_end_matches('/').to_string();
            tracing::info!(%addr, %upstream, "serving traject tool-bridge");
            traject_api::serve_tool_bridge(addr, upstream).await?;
        } else {
            traject_api::serve(addr).await?;
        }
        return Ok(());
    }

    // traject agent — full Zene coding agent + local GPU inference
    if args.first().map(|s| s.as_str()) == Some("agent") {
        args.remove(0);
        return run_zene_agent(args).await;
    }

    let use_tools = args.iter().any(|a| a == "--tools");
    let kernel = args.iter().any(|a| a == "--kernel-smoke");
    let flashinfer = args.iter().any(|a| a == "--flashinfer");
    let engine_url = take_flag_value(&mut args, "--engine-url");
    let backend_url = take_flag_value(&mut args, "--backend-url");
    let model = take_flag_value(&mut args, "--model")
        .unwrap_or_else(|| "/home/bodesi/models/ds-v4-flash".into());
    let max_tokens = take_flag_value(&mut args, "--max-tokens")
        .and_then(|s| s.parse().ok())
        .unwrap_or(64u32);
    args.retain(|a| a != "--tools" && a != "--kernel-smoke" && a != "--flashinfer");
    let prompt = args
        .first()
        .cloned()
        .unwrap_or_else(|| "Say hello and stop.".into());

    tracing::info!(
        %prompt,
        use_tools,
        kernel,
        flashinfer,
        ?engine_url,
        ?backend_url,
        %model,
        max_tokens,
        "starting trajectory"
    );

    let mut policy = ReActPolicy::new(prompt);
    policy.max_steps = 4;

    let mut driver = Driver::new(DriverConfig {
        scheduler: {
            let mut s = traject_scheduler::SchedulerConfig::default();
            s.chunk_tokens = max_tokens;
            s
        },
        ..DriverConfig::default()
    })
    .with_policy(std::sync::Arc::new(policy));

    driver = if flashinfer || kernel {
        #[cfg(feature = "flashinfer")]
        {
            if flashinfer {
                use traject_inference::{
                    FlashInferKernel, FlashInferKernelConfig, KernelBackend, KernelSmokeBackend,
                };
                let k = FlashInferKernel::new(FlashInferKernelConfig::default())?;
                tracing::info!(kernel = k.name(), "using in-process FlashInfer kernel");
                driver.with_kernel_smoke(KernelSmokeBackend::with_kernel(std::sync::Arc::new(k)))
            } else {
                tracing::info!("using in-process CPU kernel smoke");
                driver.with_kernel_smoke(traject_inference::KernelSmokeBackend::cpu())
            }
        }
        #[cfg(not(feature = "flashinfer"))]
        {
            if flashinfer {
                return Err(
                    "rebuild with --features flashinfer to use in-process FlashInfer".into(),
                );
            }
            tracing::info!("using in-process CPU kernel smoke");
            driver.with_kernel_smoke(traject_inference::KernelSmokeBackend::cpu())
        }
    } else if let Some(url) = engine_url {
        driver.with_engine_backend(&url, &model)
    } else if let Some(url) = backend_url {
        driver.with_http_backend(&url, &model)
    } else if use_tools {
        driver.with_stub_mode(StubMode::ToolThenStop {
            remaining_tools: 1,
        })
    } else {
        driver.with_stub_mode(StubMode::AlwaysStop)
    };

    let id = driver.create_trajectory(TrajectoryConfig::default());
    driver.run_until_finished(id).await?;

    let traj = driver.manager.get(id)?;
    let answer = traj.last_outcome().and_then(|o| match o {
        traject_core::StepOutcome::Generated { text, .. } => Some(text.as_str()),
        _ => None,
    });
    tracing::info!(
        trajectory = %id,
        state = ?traj.state,
        steps = traj.history.len(),
        memory_nodes = driver.memory.stats().nodes,
        ?answer,
        "trajectory finished"
    );
    if let Some(a) = answer {
        println!("{a}");
    }
    Ok(())
}

async fn run_zene_agent(mut args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let backend_url = take_flag_value(&mut args, "--backend-url")
        .unwrap_or_else(|| "http://127.0.0.1:8000/v1".into());
    let model = take_flag_value(&mut args, "--model")
        .unwrap_or_else(|| "/home/bodesi/models/ds-v4-flash".into());
    let workdir = take_flag_value(&mut args, "--workdir")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let max_turns = take_flag_value(&mut args, "--max-turns")
        .and_then(|s| s.parse().ok())
        .unwrap_or(8u32);
    let no_yolo = args.iter().any(|a| a == "--no-yolo");
    // sglang-lite rejects `tools`; by default spawn an in-process tool-bridge.
    let direct = args.iter().any(|a| a == "--direct");
    args.retain(|a| a != "--no-yolo" && a != "--direct");
    let prompt = args.join(" ");
    if prompt.trim().is_empty() {
        return Err(
            "usage: traject agent [--backend-url URL] [--model ID] [--workdir DIR] [--direct] <prompt>"
                .into(),
        );
    }

    let upstream = backend_url
        .trim_end_matches('/')
        .trim_end_matches("/v1")
        .to_string();

    let (agent_base, _bridge_handle) = if direct {
        tracing::info!(%backend_url, "direct mode: no tool-bridge");
        (backend_url.clone(), None)
    } else {
        let (addr, handle) = traject_api::spawn_tool_bridge(upstream.clone()).await?;
        let bridged = format!("http://{addr}/v1");
        tracing::info!(%bridged, upstream = %upstream, "tool-bridge ready for zene");
        (bridged, Some(handle))
    };

    tracing::info!(
        %prompt,
        backend_url = %agent_base,
        %model,
        workdir = %workdir.display(),
        max_turns,
        "starting zene agent on traject"
    );

    let mut runner = traject_zene::ZeneRunner::new(traject_zene::ZeneRunConfig {
        workdir,
        base_url: agent_base,
        model,
        api_key: std::env::var("OPENAI_API_KEY")
            .or_else(|_| std::env::var("ZENE_API_KEY"))
            .unwrap_or_else(|_| "sk-traject-local".into()),
        max_turns,
        profile: traject_zene::AgentProfile::parse(
            &std::env::var("ZENE_AGENT_PROFILE").unwrap_or_else(|_| "coder".into()),
        )
        .unwrap_or(traject_zene::AgentProfile::Coder),
        yolo: !no_yolo,
        system_prompt: None,
    });

    let result = runner.prompt(&prompt).await?;
    tracing::info!(
        trajectory = %result.trajectory_id,
        workdir = %result.workdir.display(),
        "zene agent finished"
    );
    println!("{}", result.answer);
    Ok(())
}

fn take_flag_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    if let Some(pos) = args.iter().position(|a| a == flag) {
        args.remove(pos);
        if pos < args.len() {
            return Some(args.remove(pos));
        }
    }
    if let Some(pos) = args.iter().position(|a| a.starts_with(&format!("{flag}="))) {
        let raw = args.remove(pos);
        return raw.split_once('=').map(|(_, v)| v.to_string());
    }
    None
}
