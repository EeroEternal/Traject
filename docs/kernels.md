# In-process kernels

Traject schedules **Trajectory Steps**. Generate ultimately needs device ops:

- Prefill / Decode attention
- Paged KV read/write
- Sampling

Today full MoE weights run in an **owned engine subprocess**
(`third_party/sglang-lite` on `:9001`). The kernel layer moves ops **into the Traject process**.

## Layout

- `KernelBackend` — `prefill` / `decode` / `sample`
- `CpuRefKernel` — pure Rust reference (tests / no GPU)
- `FlashInferKernel` (`--features flashinfer`) — embed CPython, call FlashInfer CUDA in-process
- `KernelSmokeBackend` — InferenceBackend that runs Generate via KernelBackend (synthetic QKV until full weights land)

## MemoryManager handoff (target)

1. Scheduler emits `RunGenerateChunk` with `prefix` + `engine_handle`
2. Kernel reads/writes paged KV blocks owned by `MemoryManager`
3. Pin/evict decisions free **physical** blocks, not only logical nodes

## Run

```bash
# CPU path (always)
cargo run -p traject-cli -- --kernel-smoke "hi"

# FlashInfer in-process (remote GPU + venv)
cargo run -p traject-cli --release --features flashinfer -- --flashinfer "hi"
```

## Status

FlashInfer attention decode is wired in-process. Full MoE weight load + continuous
batching still use sglang-lite; next is paging KV owned by `MemoryManager` into these kernels.
