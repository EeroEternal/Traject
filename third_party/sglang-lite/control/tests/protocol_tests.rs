//! Protocol / NDJSON TokenDelta parsing tests.

use sglang_lite_control::protocol::{TokenDelta, Usage};

#[test]
fn token_delta_roundtrip_json() {
    let d = TokenDelta {
        text: "你好".to_string(),
        finish_reason: Some("stop".to_string()),
        usage: Some(Usage {
            prompt_tokens: 3,
            completion_tokens: 1,
            total_tokens: 4,
            cache_hit_tokens: Some(2),
        }),
        error: None,
        token: Some(42),
    };
    let s = serde_json::to_string(&d).unwrap();
    let back: TokenDelta = serde_json::from_str(&s).unwrap();
    assert_eq!(back.text, "你好");
    assert_eq!(back.finish_reason.as_deref(), Some("stop"));
    assert_eq!(back.usage.unwrap().cache_hit_tokens, Some(2));
    assert_eq!(back.token, Some(42));
}

#[test]
fn token_delta_error_shape() {
    let line = r#"{"text":"","finish_reason":"error","error":"oom"}"#;
    let d: TokenDelta = serde_json::from_str(line).unwrap();
    assert_eq!(d.error.as_deref(), Some("oom"));
    assert_eq!(d.finish_reason.as_deref(), Some("error"));
}
