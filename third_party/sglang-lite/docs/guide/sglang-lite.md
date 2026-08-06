# sglang-lite Integration Guide for UniGateway

sglang-lite is a lightweight, high-cohesion MoE inference engine library. UniGateway can drive it as a local backend.

## Modes

### 1. HTTP Mode (Recommended for now, P0)

Run sglang-lite as a standalone OpenAI-compatible server.

```bash
python -m sglang_lite.server \
  --port 8000 \
  --model "deepseek-ai/DeepSeek-V2-Lite-Chat" \
  --device cuda \
  --max-batch-size 8
```

Environment variables (via Config.from_env):

- SGLANG_LITE_MODEL
- SGLANG_LITE_DEVICE
- SGLANG_LITE_PORT
- SGLANG_LITE_MAX_BATCH_SIZE
- SGLANG_LITE_MAX_CONCURRENT
- SGLANG_LITE_REQUEST_TIMEOUT
- SGLANG_LITE_LOG_LEVEL

UniGateway config example (TOML):

```toml
[[providers]]
name = "my-moe"
provider_type = "sglang-lite"
base_url = "http://localhost:8000/v1"
# api_key can be empty for local
api_key = ""

[providers.model_policy]
default_model = "local-moe"
```

Metadata keys (in Endpoint.metadata):

- unigateway.sglang_lite.backend_mode = "http"
- unigateway.sglang_lite.model_path
- unigateway.sglang_lite.device
- unigateway.sglang_lite.max_batch_size

### 2. Subprocess Mode

UniGateway can launch the server as a child process and wait for /health.

See UniGateway docs for `unigateway.sglang_lite.subprocess.*` keys.

Startup command example: `python -m sglang_lite.server --port 8000 ...`

Health: poll HTTP /health until 200.

### 3. gRPC Mode (Future, P5+)

See `docs/sglang-lite-grpc-spec.md` and `proto/sglang_lite.proto`.

## Cache Hit Metrics Passthrough

sglang-lite returns `cache_hit_tokens` in the `usage` object of responses.

Example response usage:

```json
"usage": {
  "prompt_tokens": 20,
  "completion_tokens": 5,
  "total_tokens": 25,
  "cache_hit_tokens": 12
}
```

UniGateway parses this into `TokenUsage.cache_hit_tokens` and exposes via `RequestReport`.

## Current Status

- HTTP + subprocess: Supported in UniGateway v2.6.0
- gRPC: NotImplemented (both sides). See spec for details.
- Tool calling: Passthrough via OpenAI compatible path.
