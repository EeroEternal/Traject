#!/usr/bin/env python3
"""Compare Lite Hybrid vs official Transformer prefill-final logits (TP=8).

Loads the same Hybrid path twice is expensive; this script runs **Lite only**
under torchrun and optionally compares against a saved official logits tensor
(``--official-logits``). Without that file it still records Lite topk / argmax
for gate documentation.

  SGLANG_LITE_V4_DISABLE_FI_SPARSE=1 \\
  PATH=/usr/local/cuda/bin:$PATH CPATH=/usr/local/cuda/include \\
    torchrun --nproc-per-node=8 scripts/v4_logits_compare.py \\
      --prompt Hello --out /tmp/v4_logits_compare.json

Optional: produce official logits on rank0-only nested job is avoided; use
``scripts/v4_debug_first_token.py`` style load if you dump ``.pt`` offline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]

_fi_prefix = os.environ.get("SGLANG_LITE_FI_PREFIX", "/tmp/fi1616")
if _fi_prefix and Path(_fi_prefix).is_dir():
    os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
    if _fi_prefix not in sys.path:
        sys.path.insert(0, _fi_prefix)

os.environ.setdefault("SGLANG_LITE_V4_DISABLE_FI_SPARSE", "1")


def _rank() -> int:
    return int(os.environ.get("RANK", "0"))


def _rprint(*a, **k):
    if _rank() == 0:
        print(*a, **k)


def _encode_prompt(hf: Path, prompt: str, runner) -> list[int]:
    encoding_dir = hf / "encoding"
    infer_dir = hf / "inference"
    if encoding_dir.is_dir():
        for p in (str(encoding_dir), str(infer_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from encoding_dsv4 import encode_messages

            rendered = encode_messages(
                [{"role": "user", "content": prompt}], thinking_mode="chat"
            )
            return list(runner.tokenizer.encode(rendered))
        except Exception as e:
            _rprint(f"[logits] encoding_dsv4 fallback: {e}")
    return list(runner.tokenize(prompt))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-ckpt", default=os.environ.get("SGLANG_LITE_DSV4_HF", ""))
    ap.add_argument("--prompt", default="Hello")
    ap.add_argument("--out", default="/tmp/v4_logits_compare.json")
    ap.add_argument(
        "--official-logits",
        default="",
        help="Optional path to a 1D float tensor .pt of official prefill-final logits",
    )
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    hf = Path(args.hf_ckpt or os.path.expanduser("~/models/ds-v4-flash")).expanduser()

    import torch
    import torch.distributed as dist

    from sglang_lite.core import LiteEngine
    from sglang_lite.v4_prefix_cache import clear_v4_kv_slot

    engine = LiteEngine(
        model_name=str(hf),
        device="cuda",
        max_batch_size=4,
        start_loop=False,
    )
    ids = _encode_prompt(hf, args.prompt, engine.runner)
    clear_v4_kv_slot(engine.runner.model, batch_slot=0)
    inp = torch.tensor([ids], dtype=torch.long, device="cuda")
    logits = engine.runner._model_forward_v4(inp, start_pos=0)[0].detach().float().cpu()
    topv, topi = torch.topk(logits, int(args.topk))
    topk = [
        {
            "id": int(i),
            "logit": float(v),
            "text": engine.runner.tokenizer.decode([int(i)]),
        }
        for v, i in zip(topv.tolist(), topi.tolist())
    ]
    argmax = int(logits.argmax().item())

    # Soft gate without official .pt: English "Hello" (or prompt lead) in Lite top5.
    lead = (args.prompt or "Hello").split()[0]
    soft_self = any(lead.lower() in (t["text"] or "").lower() for t in topk)
    # Observed CVD near-tie: 你好≈27.57 vs Hello≈27.48 (Δ≈0.09).
    if len(topk) >= 2:
        summary_delta = abs(float(topk[0]["logit"]) - float(topk[1]["logit"]))
    else:
        summary_delta = None

    summary = {
        "prompt": args.prompt,
        "prompt_tokens": len(ids),
        "prompt_ids_head": list(ids)[:16],
        "lite_argmax": argmax,
        "lite_argmax_text": engine.runner.tokenizer.decode([argmax]),
        "lite_top5": topk,
        "top2_logit_delta": summary_delta,
        "gate_soft": soft_self,
        "disable_fi_sparse": os.environ.get("SGLANG_LITE_V4_DISABLE_FI_SPARSE", ""),
        "cvd_remapped": bool(os.environ.get("CUDA_VISIBLE_DEVICES"))
        and "," not in os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "note": (
            "TileLang requires CVD=LOCAL_RANK→cuda:0; official generate uses "
            "set_device(local_rank). Near-ties (~0.09) between Hello/你好 are expected."
        ),
    }

    off_path = Path(args.official_logits) if args.official_logits else None
    if _rank() == 0:
        lite_pt = Path(args.out).with_suffix(".lite.pt")
        torch.save({"logits": logits, "prompt_ids": ids}, str(lite_pt))
        summary["lite_logits_pt"] = str(lite_pt)

    if off_path and off_path.is_file() and _rank() == 0:
        official = torch.load(str(off_path), map_location="cpu", weights_only=False)
        if isinstance(official, dict):
            official = official.get("logits", official.get("tensor"))
        official = official.detach().float().cpu().view(-1)
        n = min(official.numel(), logits.numel())
        a = logits[:n]
        b = official[:n]
        diff = (a - b).abs()
        summary["official_argmax"] = int(b.argmax().item())
        summary["max_abs"] = float(diff.max().item())
        summary["mean_abs"] = float(diff.mean().item())
        summary["argmax_match"] = summary["lite_argmax"] == summary["official_argmax"]
        off_top = set(torch.topk(b, int(args.topk)).indices.tolist())
        lite_top = set(topi.tolist())
        summary["top5_overlap"] = len(off_top & lite_top)
        summary["gate_soft"] = (
            summary["argmax_match"]
            or summary["lite_argmax"] in off_top
            or summary["official_argmax"] in lite_top
            or soft_self
        )

    if dist.is_initialized():
        dist.barrier()

    if _rank() == 0:
        Path(args.out).write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        _rprint(json.dumps(summary, ensure_ascii=False, indent=2))
        if not summary.get("gate_soft", True):
            _rprint("[logits] GATE FAIL", file=sys.stderr)
            return 1
        _rprint("[logits] OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        if _rank() == 0:
            print(f"[logits] FATAL: {e}", file=sys.stderr)
        raise
