#!/usr/bin/env python3
"""LiteEngine Hybrid V4 generate + timing (torchrun TP=8).

Wires Scheduler continuous batching through official Transformer.forward
(input_ids, start_pos). KV stays in the model; Radix prefix reuse is disabled.

Launch (on 8×GPU host)::

  export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
  export SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8
  PATH=/usr/local/cuda/bin:$PATH CPATH=/usr/local/cuda/include \\
    torchrun --nproc-per-node=8 scripts/v4_lite_engine_gen.py \\
      --case 1x128 --case 4x96 --case 1x256

Cases match scripts/v4_timed_generate.py baselines (warm tok/s).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# TileLang sparse_attn requires process-local device_id==0. Remap before torch import.
if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

# Prefer isolated FlashInfer≥0.6.16 for SM120 sparse MLA (keep shared venv on 0.6.12).
_fi_prefix = os.environ.get("SGLANG_LITE_FI_PREFIX", "/tmp/fi1616")
if _fi_prefix and Path(_fi_prefix).is_dir():
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    if _fi_prefix not in sys.path:
        sys.path.insert(0, _fi_prefix)


CASES = {
    "1x128": (1, 128),
    "4x96": (4, 96),
    "1x256": (1, 256),
}


def _rprint(rank: int, *a, **k):
    if rank == 0:
        print(*a, **k)


def _encode_prompt(hf: Path, text: str, runner) -> list[int]:
    """Prefer official chat encoding; fall back to runner.tokenize."""
    encoding_dir = hf / "encoding"
    infer_dir = hf / "inference"
    if encoding_dir.is_dir():
        for p in (str(encoding_dir), str(infer_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from encoding_dsv4 import encode_messages

            rendered = encode_messages(
                [{"role": "user", "content": text}], thinking_mode="chat"
            )
            return runner.tokenizer.encode(rendered)
        except Exception as e:
            _rprint(0, f"[v4-lite-engine] encoding_dsv4 fallback: {e}")
    return runner.tokenize(text)


def _run_case(engine, prompt_ids: list[int], batch: int, max_new: int, rank: int) -> dict:
    import torch
    import torch.distributed as dist

    reqs = [
        {
            "request_id": f"c{i}",
            "input_ids": list(prompt_ids),
            "max_tokens": max_new,
            "temperature": 0.0,
            "ignore_eos": True,
        }
        for i in range(batch)
    ]

    # Cold
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t0 = time.perf_counter()
    cold = engine.generate_batch(reqs, timeout_s=3600.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t_cold = time.perf_counter() - t0
    n_cold = sum(int(r["usage"]["completion_tokens"]) for r in cold)

    # Warm (new request ids; start_pos=0 overwrites official KV)
    reqs_w = [{**r, "request_id": f"w{i}"} for i, r in enumerate(reqs)]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t1 = time.perf_counter()
    warm = engine.generate_batch(reqs_w, timeout_s=3600.0)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()
    t_warm = time.perf_counter() - t1
    n_warm = sum(int(r["usage"]["completion_tokens"]) for r in warm)

    texts = [r["text"] for r in warm]
    return {
        "batch": batch,
        "max_new_tokens": max_new,
        "completion_tokens_cold": n_cold,
        "completion_tokens_warm": n_warm,
        "generate_cold_s": round(t_cold, 3),
        "generate_warm_s": round(t_warm, 3),
        "tok_s_cold": round(n_cold / t_cold, 2) if t_cold > 0 else None,
        "tok_s_warm": round(n_warm / t_warm, 2) if t_warm > 0 else None,
        "sample_text": texts[0][:200] if texts else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--hf-ckpt",
        default=os.environ.get("SGLANG_LITE_DSV4_HF", "")
        or os.path.expanduser("~/models/ds-v4-flash"),
    )
    ap.add_argument(
        "--converted",
        default=os.environ.get("SGLANG_LITE_DSV4_CONVERTED", "/tmp/ds-v4-mp8"),
    )
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument(
        "--case",
        action="append",
        choices=sorted(CASES.keys()),
        help="Benchmark case (repeatable). Default: all three.",
    )
    ap.add_argument("--max-batch", type=int, default=8)
    args = ap.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print("requires torchrun with WORLD_SIZE>=2 (typically 8)", file=sys.stderr)
        return 2

    hf = Path(args.hf_ckpt).expanduser()
    converted = Path(args.converted).expanduser()
    os.environ["SGLANG_LITE_DSV4_HF"] = str(hf)
    os.environ["SGLANG_LITE_DSV4_CONVERTED"] = str(converted)

    # Prefer system CUDA toolchain for TileLang JIT (no pip nvidia/cu13 headers).
    cuda_bin = "/usr/local/cuda/bin"
    cuda_inc = "/usr/local/cuda/include"
    if Path(cuda_bin).is_dir():
        os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
        os.environ.setdefault("CUDA_PATH", "/usr/local/cuda")
    if Path(cuda_inc).is_dir():
        os.environ["CPATH"] = cuda_inc
        os.environ["CPLUS_INCLUDE_PATH"] = cuda_inc

    repo = Path(__file__).resolve().parents[1]
    eng = repo / "engine"
    if str(eng) not in sys.path:
        sys.path.insert(0, str(eng))
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    import torch

    if torch.cuda.is_available():
        try:
            torch.cuda.memory._set_allocator_settings("expandable_segments:True")
        except Exception:
            pass

    from sglang_lite import LiteEngine
    from sglang_lite.capability import probe_kernel_capabilities

    caps = probe_kernel_capabilities("cuda")
    _rprint(
        rank,
        "[v4-lite-engine] routing "
        f"arch={caps.arch_family.value} "
        f"sparse_mla={caps.sparse_mla_backend.value} "
        f"moe_gemm={caps.moe_gemm_backend.value} "
        f"fi={caps.flashinfer_version}",
    )

    t_load0 = time.perf_counter()
    # start_loop=False → sync pump_until_idle (all ranks participate in every forward).
    engine = LiteEngine(
        model_name=str(hf),
        device="cuda",
        max_batch_size=args.max_batch,
        start_loop=False,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier()
    t_load = time.perf_counter() - t_load0
    _rprint(rank, f"TIMING load_s={t_load:.3f} device={engine.runner.device}")
    assert getattr(engine.runner, "_v4_hybrid", False), "expected V4 Hybrid runner"

    prompt_ids = _encode_prompt(hf, args.prompt, engine.runner)
    _rprint(rank, f"[v4-lite-engine] prompt_tokens={len(prompt_ids)}")

    cases = args.case or list(CASES.keys())
    results = {"load_s": round(t_load, 3), "prompt": args.prompt, "cases": {}}
    for name in cases:
        batch, max_new = CASES[name]
        _rprint(rank, f"[v4-lite-engine] case {name} batch={batch} max_new={max_new}")
        results["cases"][name] = _run_case(engine, prompt_ids, batch, max_new, rank)

    routed = getattr(engine.runner.kernel_backend, "_v4_sparse_attn_routed", None)
    if routed is not None and hasattr(routed, "_sglang_lite_stats"):
        results["sparse_mla_hook_stats"] = dict(routed._sglang_lite_stats)
    results["sparse_mla_backend"] = getattr(
        engine.runner.kernel_backend.sparse_mla_backend, "value", "?"
    )
    _rprint(rank, json.dumps(results, indent=2))
    engine.shutdown()
    if dist.is_initialized():
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
