# In-process kernels & local weight runner

Traject schedules **Trajectory Steps**. Generate ultimately needs device ops:

- Prefill / Decode attention
- Paged KV read/write
- Sampling

## Two in-process paths

| Path | CLI | What it owns |
|------|-----|----------------|
| Kernel smoke | `--kernel-smoke` / `--flashinfer` | Synthetic QKV + attention kernel only |
| **Local weight runner** | `--local-runner` | Toy embed/unembed weights + **physical paged KV** + KernelBackend |
| sglang-lite subprocess | `--engine-url` | Full MoE (DeepSeek-V4) on GPU |

## Layout

- `KernelBackend` — `prefill` / `decode` / `sample`
- `CpuRefKernel` — pure Rust reference (tests / no GPU)
- `FlashInferKernel` (`--features flashinfer`) — embed CPython, call FlashInfer CUDA in-process
- `KernelSmokeBackend` — InferenceBackend over KernelBackend (synthetic QKV)
- **`LocalWeightRunner`** — in-process “weights + KV” runner (`PagedKvPool`); with `--features flashinfer` auto-picks `FlashInferKernel` for prefill/decode (soft-fail → `CpuRefKernel`)
- **`InferenceBackend::free_prefix`** — physical free for local pages **and** sglang `/v1/prefix/free`

Env for FlashInfer site-packages discovery: `TRAJECT_FLASHINFER_SITE`, `SGLANG_VENV`, or well-known pro6000 paths.

## MemoryManager handoff

1. Scheduler emits `RunGenerateChunk` with `prefix` + `engine_handle` / `prefix_hint`
2. Local runner stores K/V in `PagedKvPool` pages keyed by that handle
3. Eviction / trajectory cleanup calls `free_prefix` → **zero + drop** pages (local) or radix/V4 GPU free (sglang)

## Physical GPU free (sglang-lite)

`POST /v1/prefix/free`:

1. Drop pin + session token map  
2. Drop V4 CPU snapshots (`drop_exact`)  
3. `clear_v4_kv_slot` — zero live GPU batch-slot tensors  
4. `RadixCache.free_prefix_tokens` — unlink private leaves, **zero K/V pages**, return blocks to free list  

## Run

```bash
# In-process local runner (physical KV + toy weights; CPU attention)
cargo run -p traject-cli -- --local-runner --max-tokens 16 "hello"

# CPU kernel smoke (no weights)
cargo run -p traject-cli -- --kernel-smoke "hi"

# FlashInfer kernel smoke (synthetic QKV only)
cargo run -p traject-cli --release --features flashinfer -- --flashinfer "hi"

# LocalWeightRunner + FlashInfer attention (default when feature is on)
# Needs CUDA + flashinfer/torch in site-packages (see TRAJECT_FLASHINFER_SITE / SGLANG_VENV)
export SGLANG_VENV=/home/bodesi/venvs/sglang-lite
cargo run -p traject-cli --release --features flashinfer -- \
  --local-runner --model /home/bodesi/models/ds-v4-flash \
  --max-tokens 8 "hello"

# Full MoE (subprocess)
bash scripts/start_engine.sh
cargo run -p traject-cli --release -- --engine-url http://127.0.0.1:9001 ...
```

## Real safetensors (embed / head)

```bash
# Loads embed.weight + head.weight (+ norm.weight) from HF shard index.
# DeepSeek-V4: ~1GB embed + ~1GB head as f32 in RAM; middle MoE layers still proxy.
cargo run -p traject-cli --release -- \
  --local-runner \
  --model /home/bodesi/models/ds-v4-flash \
  --max-tokens 16 \
  "hello"
```

Loaded tensors (V4 naming): `embed.weight`, `head.weight`, `norm.weight` via
`model.safetensors.index.json`. Full 43-layer MoE/FP8/MLA forward remains in
sglang-lite; local runner uses real embed→(proxy attention)→real lm_head so
logits live in the true 129280-way vocab.

Also loads `tokenizer.json` via the HuggingFace `tokenizers` crate
(`HfTokenizer`) for real BPE text↔ids. (Model-repo `encoding_dsv4` is a chat
template helper, not the BPE vocab.) Without `tokenizer.json`, falls back to
toy char-hash encode and id-list decode.

### Layer stack (in-process)

Loads the first **N** DeepSeek-V4 blocks (`TRAJECT_LOCAL_LAYERS`, default **2**,
cap `TRAJECT_LOCAL_LAYERS_MAX` default **32**). Dense layers share one
safetensors catalog at load and keep **packed FP8** weights (fused matvec, no
full f32 expand). Each layer:

1. **Attention (MQA multi-head):** `attn_norm` + packed FP8 `wq_a`/`wkv`/`wq_b`  
   Q keeps **H heads × D** (`TRAJECT_ATTN_HEADS`, default 8; model has 64×512).  
   Per-head RMSNorm after `wq_b`. KV is a **single latent (K=V)**, stored compressed
   (`1 × D`) and expanded to H heads at kernel time.
2. **RoPE + FP8 QAT + attn_sink + SWA / sparse history:** last
   `qk_rope_head_dim` (64) dims get RoPE; **no-RoPE** dims get FP8
   `act_quant` (block 64, ue8m0) to match QAT. Inverse RoPE on attention
   output. Per-head `attn_sink` absorbs softmax mass.  
   - `compress_ratios[i]==0`: base `rope_theta`; attend last **`sliding_window`**
     tokens (default 128; `TRAJECT_SLIDING_WINDOW=0` = full context)  
   - `compress_ratios[i]>0`: **YaRN** + **learned compressor** → `{prefix}:L{i}:C`  
     (main compress: FP8 no-RoPE QAT; ratio-4 indexer: Hadamard+FP4 on Q/index-KV)  
     - `ratio==4`: **learned indexer** top-k (`index_topk`, default 512)  
     - other ratios: attend full compress pool (or strided fallback)
3. **o_proj:** group-concat heads → `wo_a` → `wo_b` (official layout; residual is
   `x += o`, not residual-through-`wo_a`).
4. **Hyper-Connections (HC):** embed expands to `hc_mult` (4) streams; each block
   does `hc_pre → (attn|ffn) → hc_post` with Sinkhorn `comb`; final `hc_head`
   collapses streams before lm_head.
5. **Shared + routed MoE:** under the FFN HC branch (pure delta, residual via HC).
   Gate matches official V4: `scoring_func` (default **sqrtsoftplus**), optional
   `gate.bias` for top-k only, first `num_hash_layers` use `gate.tid2eid[token]`,
   weights renormed then × `routed_scaling_factor`; experts apply `swiglu_limit`.

Per-layer KV is keyed `{prefix}:L{i}`. Free drops base + all layer keys.

**Remaining gaps vs full production:** true GPU FP8/FP4 GEMM kernels;
quality/perf parity with sglang-lite (full 43-layer eval).

```bash
export TRAJECT_LOCAL_LAYERS=2      # 1..num_hidden_layers (V4 Flash: 43)
export TRAJECT_LOCAL_LAYERS_MAX=43 # default = model depth
export TRAJECT_ATTN_HEADS=8        # Q heads (MQA); max 64 for V4 Flash
export TRAJECT_SLIDING_WINDOW=128  # 0 = disable SWA
export TRAJECT_COMPRESS_TOPK=512   # max strided history tokens (compress layers)
export TRAJECT_MOE_CACHE=32        # LRU packed experts per MoE layer
cargo run -p traject-cli --release -- \
  --local-runner --model /path/to/ds-v4-flash --max-tokens 4 "hello"
```

Dense attn/FFN use **packed FP8** fused matvec with **FP8 act_quant** on
inputs (block 128, ue8m0 — official `linear()`); all MoE layers share **one**
safetensors catalog; routed experts use **packed FP4** (same act_quant on x).
Chunk logs report `multihead`, `sliding_window`, `n_layers`, `moe_cache`.

## Status

- [x] Physical free path for sglang radix pages + V4 GPU slot clear  
- [x] In-process `LocalWeightRunner` with paged KV free  
- [x] Load real **embed + lm head (+ norm)** safetensors (sharded HF)  
- [x] Official HF `tokenizer.json` via `tokenizers` crate (`HfTokenizer`; text↔ids)  
- [x] FlashInfer as default attention for LocalWeightRunner when `--features flashinfer`  
- [x] Layer-0 attention projections (`attn_norm` + FP8 `wq_a`/`wkv` block dequant)  
- [x] Layer-0 shared-expert SwiGLU FFN (not routed MoE)  
- [x] Layer-0 MLA Q expand (`q_norm`/`kv_norm`/`wq_b`, head mean-pool)  
- [x] Layer-0 o_proj (`wo_a`/`wo_b` group inject + residual)  
- [x] Layer-0 routed MoE (gate top-k + lazy FP4 experts)  
- [x] Multi-layer stack (`TRAJECT_LOCAL_LAYERS`, per-layer KV)  
- [x] MoE kept-open catalog + true LRU expert cache  
- [x] Packed FP4 experts + fused matvec (no full f32 dequant)  
- [x] Multi-head Q + MQA KV expand (no Q mean-pool; `TRAJECT_ATTN_HEADS`)  
- [x] MLA RoPE (last 64 dims) + inverse RoPE on o + `attn_sink` + K=V  
- [x] o_proj group-concat (official `wo_a` layout)  
- [x] Hyper-Connections residual (`hc_*` + Sinkhorn + `hc_head`)  
- [x] Shared dense safetensors catalog + raised layer cap (`TRAJECT_LOCAL_LAYERS_MAX`)  
- [x] Sliding-window attention (`sliding_window` / `TRAJECT_SLIDING_WINDOW`)  
- [x] Packed FP8 dense weights + fused matvec (attn + shared FFN)  
- [x] Per-layer YaRN RoPE for compressed layers (`compress_ratios` / `compress_rope_theta`)  
- [x] Sparse window + strided history KV gather for compress layers  
- [x] Learned KV compressor (`compressor.*` → `{prefix}:L{i}:C` pool)  
- [x] Learned indexer top-k (`indexer.*` → score `:I`, select `:C`)  
- [x] Shared MoE safetensors catalog across layers + full-depth layer cap (43)  
- [x] Indexer Hadamard rotate + FP4 QAT sim (`rotate_activation` / `fp4_act_quant`)  
- [x] Main KV + compressor FP8 `act_quant` on no-RoPE dims (block 64, ue8m0)  
- [x] Official MoE gate: sqrtsoftplus + bias + hash tid2eid + swiglu_limit  
- [x] Linear act_quant on FP8/FP4 matvec inputs (block 128, ue8m0)  
- [ ] Quality/perf parity with sglang-lite (GPU kernels, full eval)  
