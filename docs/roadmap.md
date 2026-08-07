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
- [x] Layer-0 shared-expert SwiGLU FFN (`ffn_norm` + `w1/w2/w3`; no routed MoE)
- [x] Layer-0 MLA Q expand (`q_norm`/`kv_norm`/`wq_b` + head mean-pool)
- [x] Layer-0 o_proj (`wo_a`/`wo_b`; group-inject pooled attn → hidden)
- [x] Layer-0 routed MoE (gate top-k + lazy FP4 e2m1 expert dequant)
- [x] Multi-layer local stack (`TRAJECT_LOCAL_LAYERS`, default 2; per-layer KV)
- [x] MoE kept-open safetensors catalog + LRU expert dequant cache
- [x] Packed FP4 experts + fused matvec (skip full f32 expand)
- [x] Multi-head Q + MQA KV expand (`TRAJECT_ATTN_HEADS`, no Q mean-pool)
- [x] MLA RoPE + attn_sink + K=V + o_proj group-concat (official V4 path)
- [x] Hyper-Connections residual (`hc_mult` streams + Sinkhorn + `hc_head`)
- [x] Shared dense catalog + layer cap (`TRAJECT_LOCAL_LAYERS_MAX`) + sliding-window attn
- [x] Packed FP8 dense weights + fused matvec (attn + shared FFN; ~¼ RAM of f32)
- [x] Per-layer YaRN RoPE for compressed layers (`compress_ratios` + `rope_scaling`)
- [x] Sparse window + strided history KV gather for compress layers
- [x] Learned KV compressor (gated pooling → compress pool)
- [x] Learned indexer top-k over compress pool (ratio-4 layers)
- [x] Shared MoE catalog across layers + `TRAJECT_LOCAL_LAYERS` up to full model depth
- [x] Indexer Hadamard + FP4 QAT sim (official lightning-indexer QK path)
- [ ] Quality/perf parity with sglang-lite for in-process 43-layer runs
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
