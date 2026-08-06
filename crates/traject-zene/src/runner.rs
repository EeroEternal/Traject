use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::sync::Mutex;
use tracing::info;
use traject_core::{Trajectory, TrajectoryId};
use traject_inference::InferenceBackend;
use traject_runtime::{Driver, DriverConfig};
use traject_scheduler::{Scheduler, SchedulerConfig};
use zene_config::{AgentProfile, ZeneConfig};
use zene_core::{Agent, PermissionMode, PromptOptions};
use zene_llm::ChatClient;
use zene_sandbox::LocalSandbox;
use zene_session::SessionRecord;

use crate::provider::{finish_session_snapshot, TrajectLlmProvider, TrajectSession};

/// How to host Zene on Traject-owned inference + Driver.
#[derive(Debug, Clone)]
pub struct ZeneRunConfig {
    pub workdir: PathBuf,
    /// Fallback OpenAI base URL (only used for legacy HTTP path).
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub max_turns: u32,
    pub profile: AgentProfile,
    pub yolo: bool,
    pub system_prompt: Option<String>,
    pub max_tokens: u32,
}

impl Default for ZeneRunConfig {
    fn default() -> Self {
        Self {
            workdir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            base_url: "http://127.0.0.1:9001".into(),
            model: "/home/bodesi/models/ds-v4-flash".into(),
            api_key: "sk-traject-local".into(),
            max_turns: 12,
            profile: AgentProfile::Coder,
            yolo: true,
            system_prompt: None,
            max_tokens: 1024,
        }
    }
}

impl ZeneRunConfig {
    pub fn for_local_engine(workdir: impl Into<PathBuf>) -> Self {
        Self {
            workdir: workdir.into(),
            ..Self::default()
        }
    }

    fn to_zene_config(&self) -> ZeneConfig {
        let mut cfg = ZeneConfig::default();
        cfg.provider = "openai".into();
        cfg.base_url = self.base_url.trim_end_matches('/').to_string();
        if !cfg.base_url.contains("/v1") && !cfg.base_url.contains(":9001") {
            cfg.base_url = format!("{}/v1", cfg.base_url.trim_end_matches('/'));
        }
        cfg.model = self.model.clone();
        cfg.api_key = Some(self.api_key.clone());
        cfg.max_turns = self.max_turns;
        cfg.agent_profile = self.profile;
        cfg.permission_mode = if self.yolo {
            "yolo".into()
        } else {
            "manual".into()
        };
        if let Some(sp) = &self.system_prompt {
            cfg.system_prompt = sp.clone();
        } else {
            cfg.system_prompt = DEFAULT_TRAJECT_ZENE_PROMPT.to_string();
        }
        cfg.sandbox.profile = Some("off".into());
        cfg
    }
}

const DEFAULT_TRAJECT_ZENE_PROMPT: &str = r#"You are Zene running inside Traject, an agent-native runtime that colocates agent steps with local GPU inference.
Be concise. Prefer Read/Grep/Glob before Write/Edit/Bash. Complete the user task and stop when done.
"#;

#[derive(Debug, Clone)]
pub struct ZeneRunResult {
    pub trajectory_id: TrajectoryId,
    pub answer: String,
    pub workdir: PathBuf,
    pub generate_steps: u32,
    pub tool_steps: u32,
    pub total_cache_hit_tokens: u32,
    pub history_len: usize,
}

/// Hosts a Zene `Agent` on a Traject Driver (Scheduler + MemoryManager + Inference).
pub struct ZeneRunner {
    config: ZeneRunConfig,
    backend: Option<Arc<dyn InferenceBackend>>,
    trajectories: Vec<Trajectory>,
}

impl ZeneRunner {
    pub fn new(config: ZeneRunConfig) -> Self {
        Self {
            config,
            backend: None,
            trajectories: Vec::new(),
        }
    }

    /// Route LLM steps through a Traject inference backend (sglang-lite engine).
    pub fn with_backend(mut self, backend: Arc<dyn InferenceBackend>) -> Self {
        self.backend = Some(backend);
        self
    }

    pub fn config(&self) -> &ZeneRunConfig {
        &self.config
    }

    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Run one user prompt through the full Zene agent loop on a Trajectory.
    pub async fn prompt(&mut self, user_input: &str) -> Result<ZeneRunResult> {
        let workdir = canonicalize_workdir(&self.config.workdir)?;
        std::fs::create_dir_all(&workdir)
            .with_context(|| format!("create workdir {}", workdir.display()))?;

        let zene_cfg = self.config.to_zene_config();
        let sandbox = LocalSandbox::new(&workdir);
        let session = SessionRecord::new(&workdir);
        let permission = if self.config.yolo {
            PermissionMode::BypassPermissions
        } else {
            PermissionMode::Default
        };

        let (trajectory_id, answer, stats) = if let Some(backend) = &self.backend {
            let chunk = self.config.max_tokens.max(64);
            let mut sched_cfg = SchedulerConfig::default();
            sched_cfg.chunk_tokens = chunk;
            let mut driver = Driver::new(DriverConfig {
                scheduler: sched_cfg.clone(),
                max_ticks: 4096,
                block_capacity: 8192,
                ..DriverConfig::default()
            })
            .with_backend_arc(Arc::clone(backend));
            driver.scheduler = Scheduler::new(sched_cfg);
            if let Ok(url) = std::env::var("TRAJECT_ENGINE_URL") {
                driver.set_engine_prefix_client(&url);
            } else if let Ok(url) = std::env::var("ENGINE_URL") {
                driver.set_engine_prefix_client(&url);
            }

            let mut session_host = TrajectSession::new(driver, self.config.model.clone());
            session_host.max_tokens = self.config.max_tokens;
            let host = Arc::new(Mutex::new(session_host));
            let trajectory_id = host.lock().await.trajectory_id();
            info!(
                %trajectory_id,
                workdir = %workdir.display(),
                model = %self.config.model,
                "zene agent turn starting (driver/scheduler path)"
            );

            let provider = TrajectLlmProvider::new(Arc::clone(&host));
            let client = ChatClient::from_provider(Box::new(provider));
            let event_handler = TrajectSession::event_handler(Arc::clone(&host));

            let mut agent =
                Agent::new_with_client(zene_cfg, sandbox, session, permission, client)
                    .await
                    .context("create zene agent")?;

            let answer = agent
                .prompt(
                    user_input,
                    PromptOptions {
                        stream: false,
                        quiet: true,
                        cancel: None,
                        event_handler: Some(event_handler),
                    },
                )
                .await
                .context("zene agent prompt")?;

            let generate_steps = host.lock().await.generate_steps;
            let tool_steps = host.lock().await.tool_steps;
            let total_cache_hit_tokens = host.lock().await.total_cache_hit_tokens();
            let traj = finish_session_snapshot(&host).await?;
            let history_len = traj.history.len();
            self.trajectories.push(traj);

            (
                trajectory_id,
                answer,
                (generate_steps, tool_steps, total_cache_hit_tokens, history_len),
            )
        } else {
            info!(
                workdir = %workdir.display(),
                base_url = %self.config.base_url,
                model = %self.config.model,
                "zene agent turn starting (legacy openai http — demoted)"
            );
            let mut agent = Agent::new(zene_cfg, sandbox, session, permission)
                .await
                .context("create zene agent")?;
            let answer = agent
                .prompt(
                    user_input,
                    PromptOptions {
                        stream: false,
                        quiet: true,
                        cancel: None,
                        event_handler: None,
                    },
                )
                .await
                .context("zene agent prompt")?;
            (TrajectoryId::new(), answer, (0, 0, 0, 0))
        };

        info!(
            %trajectory_id,
            answer_chars = answer.len(),
            generate_steps = stats.0,
            tool_steps = stats.1,
            cache_hit_tokens = stats.2,
            history_len = stats.3,
            "zene agent turn finished"
        );

        Ok(ZeneRunResult {
            trajectory_id,
            answer,
            workdir,
            generate_steps: stats.0,
            tool_steps: stats.1,
            total_cache_hit_tokens: stats.2,
            history_len: stats.3,
        })
    }
}

fn canonicalize_workdir(path: &Path) -> Result<PathBuf> {
    if path.exists() {
        Ok(path
            .canonicalize()
            .with_context(|| format!("canonicalize {}", path.display()))?)
    } else {
        Ok(path.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_defaults_engine_port() {
        let c = ZeneRunConfig::default();
        assert!(c.base_url.contains("9001"));
    }
}
