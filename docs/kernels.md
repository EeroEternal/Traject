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
- **`LocalWeightRunner`** — in-process “weights + KV” runner (`PagedKvPool`)
- **`InferenceBackend::free_prefix`** — physical free for local pages **and** sglang `/v1/prefix/free`

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
# In-process local runner (physical KV + toy weights)
cargo run -p traject-cli -- --local-runner --max-tokens 16 "hello"

# CPU kernel smoke
cargo run -p traject-cli -- --kernel-smoke "hi"

# FlashInfer in-process
cargo run -p traject-cli --release --features flashinfer -- --flashinfer "hi"

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

## Status

- [x] Physical free path for sglang radix pages + V4 GPU slot clear  
- [x] In-process `LocalWeightRunner` with paged KV free  
- [x] Load real **embed + lm head (+ norm)** safetensors (sharded HF)  
- [x] Official HF `tokenizer.json` via `tokenizers` crate (`HfTokenizer`; text↔ids)  
- [ ] Full MoE / MLA layer stack in-process (still sglang for production MoE)  
- [ ] FlashInfer as default attention for LocalWeightRunner when feature on  
