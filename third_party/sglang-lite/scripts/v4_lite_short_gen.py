#!/usr/bin/env python3
"""P4 short-gen entry for DeepSeek-V4-Flash.

Official gold baseline via ``generate.py``. For LiteEngine Hybrid CB + timing,
use ``scripts/v4_lite_engine_gen.py`` (torchrun TP=8).

This script:

1. Prints KernelBackend arch_family / sparse_mla / moe_gemm routing.
2. Runs official ``generate.py`` via torchrun against converted MP shards
   (gold baseline from P0) so short-prompt tokens are produced on 8×GPU.

Env:
  SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
  SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8
  MP=8
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _print_routing() -> None:
    try:
        from sglang_lite.capability import probe_kernel_capabilities

        caps = probe_kernel_capabilities("cuda")
        print(
            "[v4-lite] routing "
            f"arch={caps.arch_family.value} "
            f"sparse_mla={caps.sparse_mla_backend.value} "
            f"moe_gemm={caps.moe_gemm_backend.value} "
            f"fi={caps.flashinfer_version}"
        )
    except Exception as e:
        print(f"[v4-lite] routing probe skipped: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ckpt", default=os.environ.get("SGLANG_LITE_DSV4_HF", ""))
    ap.add_argument(
        "--converted",
        default=os.environ.get("SGLANG_LITE_DSV4_CONVERTED", "/tmp/ds-v4-mp8"),
    )
    ap.add_argument("--mp", type=int, default=int(os.environ.get("MP", "8")))
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--skip-generate", action="store_true")
    args = ap.parse_args()

    hf = Path(args.hf_ckpt or os.path.expanduser("~/models/ds-v4-flash")).expanduser()
    converted = Path(args.converted).expanduser()
    infer = hf / "inference"
    if not infer.is_dir():
        print(f"missing inference dir: {infer}", file=sys.stderr)
        return 2

    _print_routing()

    # Registry smoke (no weight load)
    from sglang_lite.models import assert_moe_supported

    fam = assert_moe_supported(str(hf), "deepseek_v4")
    print(f"[v4-lite] family={fam.name}")

    if args.skip_generate:
        return 0

    shard0 = converted / f"model0-mp{args.mp}.safetensors"
    if not shard0.is_file():
        print(
            f"converted shard missing: {shard0}\n"
            "Run: bash scripts/v4_official_smoke.sh",
            file=sys.stderr,
        )
        return 3

    prompt_file = converted / "lite_short_prompt.txt"
    prompt_file.write_text(args.prompt + "\n", encoding="utf-8")
    log_path = converted / "lite_short_gen.log"
    infer_cfg = infer / "config.json"
    if not infer_cfg.is_file():
        print(f"missing ModelArgs config: {infer_cfg}", file=sys.stderr)
        return 2
    # System CUDA only — do not mix pip nvidia/cu13 headers with toolkit nvcc.
    env = os.environ.copy()
    cuda_bin = "/usr/local/cuda/bin"
    cuda_inc = "/usr/local/cuda/include"
    if Path(cuda_bin).is_dir():
        env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        env["CUDA_HOME"] = "/usr/local/cuda"
        env["CUDA_PATH"] = "/usr/local/cuda"
    env["CPATH"] = cuda_inc
    env["CPLUS_INCLUDE_PATH"] = cuda_inc

    cmd = [
        "torchrun",
        f"--nproc-per-node={args.mp}",
        str(infer / "generate.py"),
        "--ckpt-path",
        str(converted),
        "--config",
        str(infer_cfg),
        "--input-file",
        str(prompt_file),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0.0",
    ]
    print("[v4-lite] exec:", " ".join(cmd))
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=str(infer), stdout=log, stderr=subprocess.STDOUT, env=env
        )
    print(f"[v4-lite] generate exit={proc.returncode} log={log_path}")
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        print(text[-4000:])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
