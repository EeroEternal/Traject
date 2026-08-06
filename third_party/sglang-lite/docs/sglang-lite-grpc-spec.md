# sglang-lite gRPC Specification (for UniGateway)

**Status**: Future work. Both sides currently have gRPC as NotImplemented. Use HTTP mode for now.

## 1. Service Definition

See `proto/sglang_lite.proto`.

- Service: `SglangLiteService`
  - `ChatCompletions` (unary for non-stream)
  - `ChatCompletionsStream` (server streaming)
  - `Embeddings`
  - `ListModels`

Messages are OpenAI-like for compatibility. `Usage` includes `cache_hit_tokens`.

## 2. Connection Information

- Default gRPC port: 50051
- HTTP port (for compatibility): 8000 (configurable)
- TLS: not enabled by default
- Auth: none for local (empty api_key supported)
- Use standard gRPC client, no special authority needed for local.

## 3. Error & Retry Semantics

Standard gRPC status codes:

- `UNAVAILABLE`: retryable (e.g. temporary resource issue)
- `RESOURCE_EXHAUSTED`: corresponds to rate limit / 429
- `INVALID_ARGUMENT`: bad request
- `UNIMPLEMENTED`: method not available

Details can be in status message.

## 4. Health Check

Use standard `grpc.health.v1.Health`.

- `SERVING` after model is loaded.
- In subprocess mode, UniGateway polls health.

## 5. Metrics & Metadata Passthrough

- `cache_hit_tokens` is returned in `Usage` of responses (unary and last stream chunk).
- Propagate request metadata (e.g. x-request-id) via gRPC metadata.
- UniGateway can read from response and put into RequestReport.

## 6. Lifecycle / Subprocess

When UniGateway starts sglang-lite as gRPC subprocess:

- Command: `sglang-lite serve --port 50051 --model <model>` (or python equivalent for the core)
- gRPC and HTTP ports can be separate or same.
- After start, poll gRPC health or HTTP /health.
- Auto restart on crash.
- Graceful shutdown on SIGTERM.
- Forward logs.

sglang-lite server should implement health and clean exit.
