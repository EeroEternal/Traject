use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;
use traject_core::{Result, StepId, ToolCall, ToolResult, TrajectError, TrajectoryId};

use crate::registry::ToolRegistry;
use crate::sandbox::Sandbox;

#[derive(Debug, Clone)]
pub struct ToolRuntimeConfig {
    pub default_timeout: Duration,
}

impl Default for ToolRuntimeConfig {
    fn default() -> Self {
        Self {
            default_timeout: Duration::from_secs(30),
        }
    }
}

#[derive(Debug)]
pub struct ToolFinishedEvent {
    pub trajectory_id: TrajectoryId,
    pub step_id: StepId,
    pub result: Result<ToolResult>,
}

pub struct ToolRuntime {
    registry: Arc<ToolRegistry>,
    sandbox: Sandbox,
    config: ToolRuntimeConfig,
    tx: mpsc::UnboundedSender<ToolFinishedEvent>,
}

impl ToolRuntime {
    pub fn new(
        registry: Arc<ToolRegistry>,
        config: ToolRuntimeConfig,
    ) -> (Self, mpsc::UnboundedReceiver<ToolFinishedEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (
            Self {
                registry,
                sandbox: Sandbox::default(),
                config,
                tx,
            },
            rx,
        )
    }

    /// Fire-and-forget async tool execution; completion arrives on the event channel.
    pub fn spawn(
        &self,
        trajectory_id: TrajectoryId,
        step_id: StepId,
        call: ToolCall,
        timeout: Option<Duration>,
    ) {
        let registry = Arc::clone(&self.registry);
        let tx = self.tx.clone();
        let timeout = timeout.unwrap_or(self.config.default_timeout);
        let allowed = self.sandbox.check_allowed(&call.name);

        tokio::spawn(async move {
            let result = if !allowed {
                Err(TrajectError::ToolFailed {
                    tool: call.name.clone(),
                    reason: "sandbox denied".into(),
                })
            } else {
                match tokio::time::timeout(timeout, registry.invoke(&call)).await {
                    Ok(r) => r,
                    Err(_) => Err(TrajectError::ToolTimeout {
                        tool: call.name.clone(),
                    }),
                }
            };
            let _ = tx.send(ToolFinishedEvent {
                trajectory_id,
                step_id,
                result,
            });
        });
    }
}
