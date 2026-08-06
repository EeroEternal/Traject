#!/usr/bin/env python3
"""Dump greedy first token from Lite Hybrid prefill (TP=8).

  SGLANG_LITE_V4_DISABLE_FI_SPARSE=1 SGLANG_LITE_V4_NO_CVD_REMAP=1 \\
    torchrun --nproc-per-node=8 scripts/v4_debug_first_token.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Optional: match official generate.py device model (all GPUs visible).
if os.environ.get("SGLANG_LITE_V4_NO_CVD_REMAP", "").lower() not in (
    "1",
    "true",
    "yes",
):
    if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

import torch
import torch.distributed as dist

torch.manual_seed(33377335)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(33377335)


def main() -> int:
    hf = Path(os.environ.get("SGLANG_LITE_DSV4_HF", "")).expanduser()
    if not hf.is_dir():
        hf = Path.home() / "models/ds-v4-flash"

    from sglang_lite.core import LiteEngine

    eng = LiteEngine(str(hf), device="cuda", max_batch_size=4, start_loop=False)
    for p in (str(hf / "encoding"), str(hf / "inference")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from encoding_dsv4 import encode_messages

    ids = list(
        eng.runner.tokenizer.encode(
            encode_messages([{"role": "user", "content": "Hello"}], thinking_mode="chat")
        )
    )
    inp = torch.tensor([ids], dtype=torch.long, device="cuda")
    # Cold clear like serving path
    from sglang_lite.v4_prefix_cache import clear_v4_kv_slot

    clear_v4_kv_slot(eng.runner.model, batch_slot=0)
    logits = eng.runner._model_forward_v4(inp, start_pos=0)
    tok = int(logits[0].argmax().item())
    topv, topi = torch.topk(logits[0], 5)
    if int(os.environ.get("RANK", "0")) == 0:
        dec = eng.runner.tokenizer.decode([tok])
        print("prompt_ids", ids)
        print("first_token", tok, repr(dec))
        print(
            "top5",
            [
                (int(i), float(v), eng.runner.tokenizer.decode([int(i)]))
                for v, i in zip(topv.tolist(), topi.tolist())
            ],
        )
    if dist.is_initialized():
        dist.barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
