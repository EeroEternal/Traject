# Roadmap

## Phase 0 – Skeleton ✅

- Trajectory / Step / state machine
- Prefix tree + MemoryManager (logical)
- Token-budget scheduler + stub Generate/Tool
- ReAct single + multi trajectory
- CLI + examples + e2e tests

## Phase 1 – Core value (in progress)

- [x] Async ToolStep + pin TTL on tool wait
- [x] Decode-first + interruptible multi-chunk
- [x] Priority / fairness credit
- [x] tracing spans
- [x] OpenAI-compatible HTTP shim (stub)
- [x] HTTP OpenAI backend bridge (`--backend-url`)
- [x] Native sglang-lite engine backend (`--engine-url` → `:9001`)
- [x] Local engine supervisor (`LocalEngineHandle` / `scripts/start_engine.sh`)
- [x] E2E real inference on 8×5090 (DeepSeek-V4-Flash)
- [x] In-process `KernelBackend` + FlashInfer decode smoke (`--features flashinfer`)
- [x] Vendored Zene agent crates (`crates/zene-*`) + TrajectLlmProvider session path
- [x] Vendored sglang-lite (`third_party/sglang-lite`) with trajectory/session/prefix fields
- [ ] Full in-process model runner (weights + paged KV owned by MemoryManager)
- [ ] Tool latency-aware pin TTL from histograms
- [ ] Prefetch on predicted next Generate

## Phase 2 – Production

- CPU offload tier
- Multi-tenant isolation enforcement
- Non-stub OpenAI compat + auth
- Correctness / load test suite

## Phase 3 – Advanced

- Distributed prefix + affinity
- Speculative / branched trajectories
- MLA/KDA integration
- Python bindings
