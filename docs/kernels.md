# In-process kernels

Traject schedules **Trajectory Steps**. Generate ultimately needs device ops:

- Prefill / Decode attention
- Paged KV read/write
- Sampling

Today those can live in a **sibling process** (sglang-lite on `:9001`). The kernel layer moves them **into the Traject process**.

## Layout

- `KernelBackend` — `prefill` / `decode` / `sample`
- `CpuRefKernel` — pure Rust reference (tests / no GPU)
- `FlashInferKernel` (`--features flashinfer`) — embed CPython, call FlashInfer CUDA in-process
- `KernelSmokeBackend` — InferenceBackend that runs Generate via KernelBackend (synthetic QKV until full weights land)

## Run

```bash
# CPU path (always)
cargo run -p traject-cli -- --kernel-smoke "hi"

# FlashInfer in-process (remote 5090 + venv)
cargo run -p traject-cli --release --features flashinfer -- --flashinfer "hi"
```

## Status

FlashInfer attention decode is wired in-process. Full MoE weight load + continuous batching still use sglang-lite; the next step is paging KV owned by `MemoryManager` into these kernels.
