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
- [x] E2E real inference on multi-GPU (DeepSeek-V4-Flash @ pro6000 8×PRO 6000)
- [x] In-process `KernelBackend` + FlashInfer decode smoke (`--features flashinfer`)
- [x] Vendored Zene agent crates (`crates/zene-*`) + TrajectLlmProvider session path
- [x] Vendored sglang-lite (`third_party/sglang-lite`) with trajectory/session/prefix fields
- [x] MemoryManager ↔ engine prefix handles (pin / reuse / cache-hit / eviction scoring)
- [x] Zene steps fully via Driver/Scheduler (`run_generate_step` / `run_external_tool_step`)
- [x] `--legacy-http` / tool-bridge demoted to non-default path
- 执行计划：[merge-zene-sglite.md](merge-zene-sglite.md)
- [x] In-process `LocalWeightRunner` (toy weights + physical `PagedKvPool`; CLI `--local-runner`)
- [x] Load real **embed + head (+ norm)** safetensors (HF sharded index; V4 `embed.weight`/`head.weight`)
- [x] Official HF `tokenizer.json` wired into `LocalWeightRunner` (`HfTokenizer` / `tokenizers` crate)
- [x] FlashInfer default attention for `LocalWeightRunner` (`--features flashinfer`, soft-fail CPU)
- [x] Layer-0 real attn projections (FP8 block dequant `wq_a`/`wkv` + `attn_norm`)
- [ ] Full MoE/MLA layer stack in-process (prod MoE remains sglang-lite)
- [x] Tool latency-aware pin TTL from histograms (`ToolLatencyTracker` p95)
- [x] Prefetch pin after tool return (`PinReason::Prefetch`)
- [x] Engine `/v1/prefix/pin|unpin|free` + Driver client (soft-fail if offline)
- [x] V4 prefix save fix (save by prompt ids after prefill/finish)
- [x] Engine-side V4 snapshot drop + radix **GPU page zero/free** on eviction
- [x] Session `prompt_lcp` floors `cache_hit_tokens` when V4 snapshot misses

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
