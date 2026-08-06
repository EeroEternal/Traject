#!/usr/bin/env python3
"""Try FlashInfer 0.6.16 sparse MLA SM120 from an isolated install prefix."""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback


def main() -> int:
    print("fi", __import__("flashinfer").__version__)
    print("sys.path0", sys.path[0])
    for name in (
        "flashinfer.mla._sparse_mla_sm120",
        "flashinfer.mla.sparse_mla_sm120",
    ):
        try:
            m = importlib.import_module(name)
            attrs = [
                a
                for a in dir(m)
                if any(k in a.lower() for k in ("sm120", "dsv4", "paged", "sparse"))
            ][:40]
            print("OK", name, attrs)
        except Exception as e:
            print("NO", name, type(e).__name__, e)

    import flashinfer.mla as mla
    import torch

    src = inspect.getsource(mla.trtllm_batch_decode_sparse_mla_dsv4)
    print(
        "dsv4_sm120_mentions",
        any(k in src.lower() for k in ("sm120", "_sparse_mla_sm120", "packed sparse")),
    )
    print("sig", inspect.signature(mla.trtllm_batch_decode_sparse_mla_dsv4))

    # SM120 DSV4 path (FI≥0.6.16): packed uint8 last-dim 584 + swa_topk_lens
    # + extra_sparse_* for compressed pool. bf16 512 fails kernel packed check.
    B, qlen, H = 1, 1, 64
    page_size, swa_pages, comp_pages = 64, 4, 16
    swa_topk, sparse_topk = 128, 128
    packed_dim = 584
    q = torch.randn(B, qlen, H, 512, device="cuda", dtype=torch.bfloat16)
    swa = torch.zeros(swa_pages, 1, page_size, packed_dim, device="cuda", dtype=torch.uint8)
    comp = torch.zeros(comp_pages, 1, page_size, packed_dim, device="cuda", dtype=torch.uint8)
    swa_idx = torch.randint(
        0, swa_pages * page_size, (B * qlen, swa_topk), device="cuda", dtype=torch.int32
    )
    comp_idx = torch.randint(
        0, comp_pages * page_size, (B * qlen, sparse_topk), device="cuda", dtype=torch.int32
    )
    swa_topk_lens = torch.full((B * qlen,), swa_topk, device="cuda", dtype=torch.int32)
    comp_topk_lens = torch.full((B * qlen,), sparse_topk, device="cuda", dtype=torch.int32)
    seq = torch.full((B,), swa_pages * page_size, device="cuda", dtype=torch.int32)
    ws = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    try:
        out = mla.trtllm_batch_decode_sparse_mla_dsv4(
            q,
            swa,
            ws,
            swa_idx,
            comp,
            None,
            seq,
            kv_layout="HND",
            swa_topk_lens=swa_topk_lens,
            extra_sparse_indices=comp_idx,
            extra_sparse_topk_lens=comp_topk_lens,
        )
        print(
            "RUN OK",
            tuple(out.shape),
            float(out.float().abs().mean()),
            bool(torch.isfinite(out).all()),
        )
        return 0
    except Exception as e:
        print("RUN FAIL", type(e).__name__, e)
        print(traceback.format_exc()[-1500:])
        return 1


if __name__ == "__main__":
    # Ensure isolated prefix wins if provided
    prefix = os.environ.get("FI_PREFIX")
    if prefix:
        sys.path.insert(0, prefix)
    raise SystemExit(main())
