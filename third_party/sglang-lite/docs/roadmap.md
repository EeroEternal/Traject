# sglang-lite Roadmap

## Phase 0 — Core Verification

Goal: A runnable system that proves the architecture and can serve real (small) models with basic continuous batching + Radix skeleton.

**Deliverables**
- [x] Project skeleton + docs (README, architecture, scope, roadmap)
- [ ] Rust axum OpenAI server
  - POST /v1/chat/completions (non-stream + stream via SSE)
  - GET /v1/models
  - GET /healthz
  - Strict minimal request model + early reject
  - Clean internal GenerationRequest sent to backend
- [ ] Internal protocol (GenerationRequest / deltas) defined in both Rust and Python (shared or duplicated with tests)
- [ ] Python execution package with clear interfaces:
  - `KVCacheManager` (Radix skeleton + allocate/evict)
  - `Scheduler` (waiting + running, step())
  - `ModelRunner` stub + simple real path (HF generate for tiny model on CPU/CUDA when available)
- [ ] Integration: Rust can talk to Python (HTTP first)
- [ ] Basic metrics emitted (even if stubbed): queue depth, batch size, fake t/s
- [ ] Can run a full chat roundtrip for a small approved MoE model/test fixture and observe logs
- [ ] CI skeleton (cargo check + fmt + clippy, python lint + basic test)

**Success criteria**
- `curl .../v1/chat/completions` returns valid OpenAI shaped response (streaming works)
- Code for core pieces is readable and < a few thousand LOC total for engine
- Easy to point at a real model dir and get tokens

**Non goals in Phase 0**
- 70B perf
- Real Radix tree with GPU pages working end to end
- Production metrics
- Any multimodal or structured output

## Phase 1 — Independent Streaming Service + Real Core (Current Focus)

- [ ] One real MoE golden path with reference correctness
- [ ] Full per-layer Radix/Paged KV implementation with prefix compute skip
- [ ] Central token-budget continuous batching engine loop
- [ ] Real Rust ↔ Python token/delta streaming with cancel and backpressure
- [ ] Official standalone CLI and minimal OpenAI chat/stream/models surface
- [ ] Readiness, Prometheus metrics, structured logging, graceful drain
- [ ] Request timeout, local queue/max-concurrency limits, stable error normalization
- [ ] CUDA graph decode
- [ ] Verified quantization and MoE model support matrix
- [ ] Prefix-heavy concurrency, cancellation, OOM and soak benchmarks

**Stretch**
- Optional Paged strategy selectable at startup
- Simple model registry extension point

## Phase 2 — Polish + Integration + Hardening

- Optional integration tests with UniGateway (semantic routing + KV affinity hints)
- Rust control plane enhancements (more early filters, intent metadata passthrough)
- Optional future: consider Rust components only if needed for extreme performance (no PyO3 embedding planned for unigateway integration)
- Performance parity target vs SGLang on prefix-heavy chat workloads (within 10-15% on same hardware)
- Packaging: docker images, helm chart skeleton
- Documentation: operator guide, "when to choose sglang-lite vs vLLM vs full SGLang"
- Formal support policy for model families

## Later / Optional

- gRPC control plane (alternative to HTTP)
- Disaggregated mode (advanced)
- Speculative as plugin
- Rust-only hot path experiment (full control)

## Non-Roadmap (永远不做 in core)

See scope.md "不做" list. If a feature is desired by many users, the correct answer is "put it in the gateway/harness layer or as a separate sidecar".

## How to Track Progress

- Issues / milestones will mirror the phases.
- Every merged PR should update this file or link the changed scope.

Current status: **Phase 1 in progress — MoE-only scope** (reassessed 2026-06-28)

**Phase 0 complete** — v0.1.0

Reassessed boundary: sglang-lite targets **MoE models only**. Dense models are explicitly out
of scope. The engine core is a pure library, and the repository ships a thin standalone
service so users do not need SGLang, vLLM, or UniGateway. Advanced gateway concerns remain
outside.

Existing Phase 1 shells (metrics endpoints, config, request IDs, timeout/concurrency fields,
benchmark scripts, and shutdown hooks) are **not completion evidence** while execution,
KV paging, batching, and streaming still use stub or placeholder paths.

Detailed milestones and exit criteria:
[`standalone-inference-service-roadmap.md`](./standalone-inference-service-roadmap.md).
