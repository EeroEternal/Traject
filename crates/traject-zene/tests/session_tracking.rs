use std::sync::Arc;

use traject_inference::{InferenceBackend, StubBackend};
use traject_zene::{TrajectLlmProvider, TrajectSession, ZeneRunConfig, ZeneRunner};
use zene_core::AgentEvent;
use zene_llm::{ChatRequest, Message, Provider};

#[tokio::test]
async fn traject_provider_records_generate_steps() {
    let backend: Arc<dyn InferenceBackend> = Arc::new(StubBackend::always_stop());
    let host = Arc::new(parking_lot::Mutex::new(TrajectSession::new(
        Arc::clone(&backend),
        "stub-model",
    )));
    let provider = TrajectLlmProvider::new(Arc::clone(&host));

    let resp = provider
        .chat(ChatRequest {
            model: "stub".into(),
            messages: vec![Message::user("hello")],
            tools: vec![],
            stream: false,
        })
        .await
        .expect("chat");
    assert!(resp.message.content.is_some());

    let resp2 = provider
        .chat(ChatRequest {
            model: "stub".into(),
            messages: vec![
                Message::user("hello"),
                Message::assistant("ok"),
                Message::user("again"),
            ],
            tools: vec![],
            stream: false,
        })
        .await
        .expect("chat2");
    assert!(resp2.message.content.is_some());

    let s = host.lock();
    assert_eq!(s.generate_steps, 2);
    assert_eq!(s.trajectory.history.len(), 2);
    assert!(s.memory.binding(s.trajectory.id).is_some());
}

#[tokio::test]
async fn tool_events_append_tool_steps() {
    let backend: Arc<dyn InferenceBackend> = Arc::new(StubBackend::always_stop());
    let host = Arc::new(parking_lot::Mutex::new(TrajectSession::new(
        Arc::clone(&backend),
        "stub-model",
    )));
    let handler = TrajectSession::event_handler(Arc::clone(&host));
    handler(AgentEvent::ToolCall {
        id: "call_1".into(),
        name: "Glob".into(),
        arguments: r#"{"pattern":"*.txt"}"#.into(),
    });
    handler(AgentEvent::ToolResult {
        id: "call_1".into(),
        name: "Glob".into(),
        content: "hello.txt".into(),
        is_error: false,
        duration_ms: Some(1),
    });
    let s = host.lock();
    assert_eq!(s.tool_steps, 1);
    assert_eq!(s.trajectory.history.len(), 1);
}

#[tokio::test]
async fn zene_runner_stub_backend_prompt() {
    let cfg = ZeneRunConfig {
        workdir: std::env::temp_dir().join("traject-zene-stub"),
        max_turns: 1,
        ..ZeneRunConfig::default()
    };
    let _ = std::fs::create_dir_all(&cfg.workdir);
    let runner = ZeneRunner::new(cfg).with_backend(Arc::new(StubBackend::always_stop()));
    assert!(runner.config().max_turns == 1);
}
