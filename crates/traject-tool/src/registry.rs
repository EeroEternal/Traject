use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use traject_core::{Result, ToolCall, ToolResult, TrajectError};

#[async_trait]
pub trait ToolHandler: Send + Sync {
    fn name(&self) -> &str;
    async fn invoke(&self, call: &ToolCall) -> Result<ToolResult>;
}

#[derive(Default)]
pub struct ToolRegistry {
    handlers: HashMap<String, Arc<dyn ToolHandler>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, handler: Arc<dyn ToolHandler>) {
        self.handlers.insert(handler.name().to_string(), handler);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn ToolHandler>> {
        self.handlers.get(name).cloned()
    }

    pub async fn invoke(&self, call: &ToolCall) -> Result<ToolResult> {
        let handler = self.get(&call.name).ok_or_else(|| TrajectError::ToolFailed {
            tool: call.name.clone(),
            reason: "tool not registered".into(),
        })?;
        handler.invoke(call).await
    }
}

/// Echo tool for smoke tests.
pub struct EchoTool;

#[async_trait]
impl ToolHandler for EchoTool {
    fn name(&self) -> &str {
        "echo"
    }

    async fn invoke(&self, call: &ToolCall) -> Result<ToolResult> {
        Ok(ToolResult {
            call_id: call.call_id.clone(),
            name: call.name.clone(),
            output: call.arguments.clone(),
            is_error: false,
        })
    }
}
