use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::info;
use traject_core::{Trajectory, TrajectoryConfig, TrajectoryId};
use zene_config::{AgentProfile, ZeneConfig};
use zene_core::{Agent, PermissionMode, PromptOptions};
use zene_sandbox::LocalSandbox;
use zene_session::SessionRecord;

/// How to reach the inference server Zene's ChatClient will call.
#[derive(Debug, Clone)]
pub struct ZeneRunConfig {
    /// Workspace the agent may read/write.
    pub workdir: PathBuf,
    /// OpenAI-compatible base URL, e.g. `http://127.0.0.1:8000/v1` or Traject serve.
    pub base_url: String,
    /// Model id as known by the server.
    pub model: String,
    /// Dummy/local API key (UniGateway requires a non-empty key).
    pub api_key: String,
    pub max_turns: u32,
    pub profile: AgentProfile,
    /// Auto-approve tool calls (equivalent to zene `--yolo`).
    pub yolo: bool,
    pub system_prompt: Option<String>,
}

impl Default for ZeneRunConfig {
    fn default() -> Self {
        Self {
            workdir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            base_url: "http://127.0.0.1:8000/v1".into(),
            model: "/home/bodesi/models/ds-v4-flash".into(),
            api_key: "sk-traject-local".into(),
            max_turns: 12,
            profile: AgentProfile::Coder,
            yolo: true,
            system_prompt: None,
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
        // Zene's openai_base_url returns base_url as-is; unigateway appends /chat/completions.
        // Ensure we end with /v1 if caller passed host only.
        if !cfg.base_url.contains("/v1") {
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
        // Prefer lightweight sandbox for integrated runs; Keel still available via profile.
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
}

/// Hosts a Zene `Agent` and records each prompt as a Traject Trajectory.
pub struct ZeneRunner {
    config: ZeneRunConfig,
    /// Lightweight Trajectory ledger (Phase-1: one Trajectory per prompt).
    trajectories: Vec<Trajectory>,
}

impl ZeneRunner {
    pub fn new(config: ZeneRunConfig) -> Self {
        Self {
            config,
            trajectories: Vec::new(),
        }
    }

    pub fn config(&self) -> &ZeneRunConfig {
        &self.config
    }

    pub fn trajectories(&self) -> &[Trajectory] {
        &self.trajectories
    }

    /// Run one user prompt through the full Zene agent loop.
    pub async fn prompt(&mut self, user_input: &str) -> Result<ZeneRunResult> {
        let workdir = canonicalize_workdir(&self.config.workdir)?;
        std::fs::create_dir_all(&workdir)
            .with_context(|| format!("create workdir {}", workdir.display()))?;

        let mut traj = Trajectory::create(TrajectoryConfig {
            tenant: traject_core::TenantId::new("zene"),
            ..TrajectoryConfig::default()
        });
        traj.start()?;
        let trajectory_id = traj.id;

        info!(
            %trajectory_id,
            workdir = %workdir.display(),
            base_url = %self.config.base_url,
            model = %self.config.model,
            "zene agent turn starting"
        );

        let zene_cfg = self.config.to_zene_config();
        let sandbox = LocalSandbox::new(&workdir);
        let session = SessionRecord::new(&workdir);
        let permission = if self.config.yolo {
            PermissionMode::BypassPermissions
        } else {
            PermissionMode::Default
        };

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

        traj.memory.set_slot("answer", &answer);
        traj.memory.scratchpad.push(format!("user: {user_input}"));
        traj.memory.scratchpad.push(format!("assistant: {answer}"));
        traj.finish()?;
        self.trajectories.push(traj);

        info!(%trajectory_id, answer_chars = answer.len(), "zene agent turn finished");

        Ok(ZeneRunResult {
            trajectory_id,
            answer,
            workdir,
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
    fn config_normalizes_v1() {
        let mut c = ZeneRunConfig::default();
        c.base_url = "http://127.0.0.1:8000".into();
        let z = c.to_zene_config();
        assert!(z.base_url.ends_with("/v1"));
    }
}
