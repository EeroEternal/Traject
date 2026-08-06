#!/usr/bin/env python3
"""Wall-clock benchmark for DeepSeek-V4-Flash via official torchrun generate.py.

Measures end-to-end (load + generate) and a warm second pass when possible.
Not LiteEngine continuous-batching throughput — that path is not fully wired yet.

Example:
  SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8 \\
  python scripts/v4_official_bench.py --mp 8 --max-new-tokens 64 --num-prompts 4
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


PROMPTS = [
    "Hello",
    "What is 2+2? Answer briefly.",
    "Explain prefix caching in one sentence.",
    "Write a haiku about GPUs.",
    "Name three MoE models.",
    "Summarize continuous batching in 20 words.",
    "What is SM120 vs SM100?",
    "Say OK.",
]


def _env() -> dict:
    env = os.environ.copy()
    cuda_bin = "/usr/local/cuda/bin"
    cuda_inc = "/usr/local/cuda/include"
    if Path(cuda_bin).is_dir():
        env["PATH"] = f"{cuda_bin}:{env.get('PATH', '')}"
        env["CUDA_HOME"] = "/usr/local/cuda"
        env["CUDA_PATH"] = "/usr/local/cuda"
    env["CPATH"] = cuda_inc
    env["CPLUS_INCLUDE_PATH"] = cuda_inc
    return env


def _run_generate(
    infer: Path,
    converted: Path,
    config: Path,
    prompt_file: Path,
    mp: int,
    max_new: int,
    log_path: Path,
) -> tuple[float, int, str]:
    cmd = [
        "torchrun",
        f"--nproc-per-node={mp}",
        str(infer / "generate.py"),
        "--ckpt-path",
        str(converted),
        "--config",
        str(config),
        "--input-file",
        str(prompt_file),
        "--max-new-tokens",
        str(max_new),
        "--temperature",
        "0.0",
    ]
    t0 = time.perf_counter()
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.run(
            cmd, cwd=str(infer), stdout=log, stderr=subprocess.STDOUT, env=_env()
        )
    elapsed = time.perf_counter() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(f"generate failed rc={proc.returncode}\n{text[-2000:]}")
    # Count completion lines roughly: lines after "Completion:"
    n_comp = text.count("Completion:")
    return elapsed, n_comp, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ckpt", default=os.environ.get("SGLANG_LITE_DSV4_HF", ""))
    ap.add_argument(
        "--converted",
        default=os.environ.get("SGLANG_LITE_DSV4_CONVERTED", "/tmp/ds-v4-mp8"),
    )
    ap.add_argument("--mp", type=int, default=int(os.environ.get("MP", "8")))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=4)
    ap.add_argument("--warmup", type=int, default=0, help="extra cold runs before timed")
    ap.add_argument("--repeats", type=int, default=1)
    args = ap.parse_args()

    hf = Path(args.hf_ckpt or os.path.expanduser("~/models/ds-v4-flash")).expanduser()
    converted = Path(args.converted).expanduser()
    infer = hf / "inference"
    config = infer / "config.json"
    if not config.is_file():
        print(f"missing {config}", file=sys.stderr)
        return 2
    shard0 = converted / f"model0-mp{args.mp}.safetensors"
    if not shard0.is_file():
        print(f"missing {shard0}; run convert first", file=sys.stderr)
        return 3

    prompts = PROMPTS[: max(1, args.num_prompts)]
    prompt_file = converted / "bench_prompts.txt"
    prompt_file.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    # Routing probe
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))
        from sglang_lite.capability import probe_kernel_capabilities

        caps = probe_kernel_capabilities("cuda")
        routing = {
            "arch_family": caps.arch_family.value,
            "sparse_mla": caps.sparse_mla_backend.value,
            "moe_gemm": caps.moe_gemm_backend.value,
            "flashinfer": caps.flashinfer_version,
        }
    except Exception as e:
        routing = {"error": str(e)}

    print("=== DeepSeek-V4 official generate benchmark ===")
    print(json.dumps({"mp": args.mp, "max_new_tokens": args.max_new_tokens,
                      "num_prompts": len(prompts), "routing": routing}, indent=2))

    for i in range(args.warmup):
        log = converted / f"bench_warmup_{i}.log"
        _run_generate(infer, converted, config, prompt_file, args.mp, args.max_new_tokens, log)
        print(f"warmup[{i}] done")

    times: list[float] = []
    for r in range(args.repeats):
        log = converted / f"bench_run_{r}.log"
        elapsed, n_comp, text = _run_generate(
            infer, converted, config, prompt_file, args.mp, args.max_new_tokens, log
        )
        times.append(elapsed)
        # Rough token estimate: max_new * prompts (upper bound; EOS may stop early)
        est_tokens = args.max_new_tokens * len(prompts)
        tps = est_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"run[{r}] wall={elapsed:.2f}s completions={n_comp} "
            f"est_tokens<={est_tokens} est_tps~={tps:.1f}"
        )
        # Show last completion snippet
        if "Completion:" in text:
            snippet = text.split("Completion:")[-1].strip().splitlines()[0][:120]
            print(f"  last_completion: {snippet}")

    summary = {
        "wall_s_mean": statistics.mean(times),
        "wall_s_min": min(times),
        "wall_s_max": max(times),
        "est_tokens_per_run": args.max_new_tokens * len(prompts),
        "est_tps_mean": (args.max_new_tokens * len(prompts)) / statistics.mean(times),
        "note": (
            "wall includes model load + distributed init each torchrun; "
            "est_tps uses max_new*num_prompts (EOS may finish early)"
        ),
    }
    print("\n=== Summary ===")
    print(json.dumps(summary, indent=2))
    out = converted / "bench_summary.json"
    out.write_text(json.dumps({"routing": routing, "summary": summary, "times": times}, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
