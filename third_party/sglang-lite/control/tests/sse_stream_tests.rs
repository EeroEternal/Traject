//! Stub-engine SSE shape: role chunk → content deltas → [DONE].

use std::sync::{atomic::AtomicBool, Arc};

use axum::body::Body;
use http_body_util::BodyExt;
use sglang_lite_control::{build_router, EngineClient, StubEngineClient};
use tower::ServiceExt;

#[tokio::test]
async fn stub_chat_completions_sse_emits_done() {
    let engine = Arc::new(EngineClient::Stub(StubEngineClient::new()));
    let models = Arc::new(vec!["stub".to_string()]);
    let ready = Arc::new(AtomicBool::new(true));
    let draining = Arc::new(AtomicBool::new(false));
    let app = build_router(engine, models, None, ready, draining);

    let req = axum::http::Request::builder()
        .method("POST")
        .uri("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(
            r#"{"model":"stub","messages":[{"role":"user","content":"hi"}],"max_tokens":8,"stream":true}"#,
        ))
        .unwrap();

    let resp = app.oneshot(req).await.expect("response");
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    assert!(
        ctype.contains("text/event-stream"),
        "content-type={ctype}"
    );

    let bytes = resp.into_body().collect().await.unwrap().to_bytes();
    let body = String::from_utf8_lossy(&bytes);
    assert!(body.contains("chat.completion.chunk"), "body={body}");
    assert!(body.contains("[DONE]"), "missing [DONE]: {body}");
    assert!(body.contains("\"finish_reason\""), "body={body}");
}
