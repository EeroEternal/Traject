#!/usr/bin/env python3
"""Timed wrapper around DeepSeek official generate (load vs decode split).

Launch with torchrun, same args as inference/generate.py plus timing prints on rank0.
"""

from __future__ import annotations

import json
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--infer-dir", type=str, default="")
    cli = parser.parse_args()

    infer_dir = cli.infer_dir or str(
        Path(os.environ.get("SGLANG_LITE_DSV4_HF", os.path.expanduser("~/models/ds-v4-flash")))
        / "inference"
    )
    sys.path.insert(0, infer_dir)
    encoding_dir = str(Path(infer_dir).parent / "encoding")
    if encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)

    import torch
    import torch.distributed as dist
    from safetensors.torch import load_model
    from transformers import AutoTokenizer

    from encoding_dsv4 import encode_messages
    from generate import generate
    from model import ModelArgs, Transformer

    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")

    def rprint(*a, **k):
        if rank == 0:
            print(*a, **k)

    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    t_all0 = time.perf_counter()
    with open(cli.config) as f:
        args = ModelArgs(**json.load(f))
    rprint("ModelArgs ready")

    t0 = time.perf_counter()
    with torch.device("cuda"):
        model = Transformer(args)
    tokenizer = AutoTokenizer.from_pretrained(cli.ckpt_path)
    shard = os.path.join(cli.ckpt_path, f"model{rank}-mp{world_size}.safetensors")
    load_model(model, shard, strict=False)
    torch.set_default_device("cuda")
    if world_size > 1:
        dist.barrier()
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    rprint(f"TIMING load_s={t_load:.3f}")

    raw = Path(cli.input_file).read_text(encoding="utf-8")
    # Official generate splits on blank lines; also accept single newlines.
    if "\n\n" in raw.strip():
        prompts = [p for p in raw.split("\n\n") if p.strip()]
    else:
        prompts = [p for p in raw.splitlines() if p.strip()]

    prompt_tokens = [
        tokenizer.encode(
            encode_messages([{"role": "user", "content": p}], thinking_mode="chat")
        )
        for p in prompts
    ]
    n_prompt_toks = sum(len(t) for t in prompt_tokens)

    # Cold generate (may include TileLang JIT)
    t1 = time.perf_counter()
    completion_tokens = generate(
        model, prompt_tokens, cli.max_new_tokens, tokenizer.eos_token_id, cli.temperature
    )
    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    t_gen_cold = time.perf_counter() - t1
    n_new_cold = sum(len(t) for t in completion_tokens)

    # Warm generate (same prompts)
    t2 = time.perf_counter()
    completion_tokens_w = generate(
        model, prompt_tokens, cli.max_new_tokens, tokenizer.eos_token_id, cli.temperature
    )
    torch.cuda.synchronize()
    if world_size > 1:
        dist.barrier()
    t_gen_warm = time.perf_counter() - t2
    n_new_warm = sum(len(t) for t in completion_tokens_w)
    completion_tokens = completion_tokens_w
    n_new = n_new_warm

    t_all = time.perf_counter() - t_all0
    rprint(
        json.dumps(
            {
                "num_prompts": len(prompts),
                "prompt_tokens": n_prompt_toks,
                "completion_tokens_cold": n_new_cold,
                "completion_tokens_warm": n_new_warm,
                "max_new_tokens": cli.max_new_tokens,
                "load_s": round(t_load, 3),
                "generate_cold_s": round(t_gen_cold, 3),
                "generate_warm_s": round(t_gen_warm, 3),
                "total_s": round(t_all, 3),
                "tok_s_cold": round(n_new_cold / t_gen_cold, 2) if t_gen_cold > 0 else None,
                "tok_s_warm": round(n_new_warm / t_gen_warm, 2) if t_gen_warm > 0 else None,
            },
            indent=2,
        )
    )
    for prompt, completion in zip(prompts, completions):
        rprint("Prompt:", prompt)
        rprint("Completion:", completion)
        rprint()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
