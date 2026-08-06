"""Hook official Hybrid ``sparse_attn`` → KernelBackend SM120 sparse MLA (decode).

Prefill (q_len>1) stays on official TileLang ``sparse_attn``. Decode tries
FlashInfer SM120 packed path when capability says so; any failure falls back.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch

from .capability import SparseMlaBackend
from .dsv4_kv_pack import (
    DSV4_PAGE_SIZE,
    pack_dsv4_kv_bf16,
    split_swa_compress_indices,
    to_paged_hnd,
)

logger = logging.getLogger("sglang_lite.v4_sparse_mla")


def _resolve_act_quant() -> Optional[Callable]:
    try:
        import kernel as official_kernel  # type: ignore

        return official_kernel.act_quant
    except Exception:
        return None


def sparse_mla_decode_from_official_tensors(
    backend: Any,
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
    *,
    window_size: int = 128,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run FI SM120 sparse MLA decode from official ``sparse_attn`` tensors.

    q: ``[B, 1, H, 512]`` bf16
    kv: ``[B, T, 512]`` bf16 (SWA ring || compressed)
    topk_idxs: ``[B, 1, K]``
    """
    if q.dim() != 4 or q.shape[1] != 1:
        raise ValueError("SM120 sparse hook supports decode q_len==1 only")
    b, _, h, d = q.shape
    if d != 512:
        raise ValueError(f"expected head_dim 512, got {d}")
    if kv.dim() != 3 or kv.shape[0] != b:
        raise ValueError(f"kv shape {tuple(kv.shape)} incompatible with q batch {b}")

    act_quant = _resolve_act_quant()
    # Pack per-batch rows into one token pool with batch-strided indices.
    # For B==1 (common single-request path) this is a straight map.
    t = kv.shape[1]
    win = min(window_size, t)
    swa_tokens = kv[:, :win, :]
    comp_tokens = kv[:, win:, :] if t > win else kv[:, :0, :]

    # Flatten batch into pages: batch-major token order [b0_t0.., b1_t0..]
    # Indices from official are per-row into that row's cache; for B>1 we
    # offset by row * win / row * comp_len.
    swa_idx, swa_lens, comp_idx, comp_lens = split_swa_compress_indices(
        topk_idxs, window_size=win
    )
    if b > 1:
        row = torch.arange(b, device=q.device, dtype=torch.int32).repeat_interleave(
            q.shape[1]
        )
        swa_off = (row * win).unsqueeze(-1)
        swa_idx = torch.where(swa_idx >= 0, swa_idx + swa_off, swa_idx)
        if comp_idx.numel():
            comp_len = max(comp_tokens.shape[1], 1)
            comp_off = (row * comp_len).unsqueeze(-1)
            comp_idx = torch.where(comp_idx >= 0, comp_idx + comp_off, comp_idx)

    swa_flat = swa_tokens.reshape(b * win, d) if win else swa_tokens.reshape(0, d)
    packed_swa = pack_dsv4_kv_bf16(swa_flat, act_quant_fn=act_quant)
    swa_pages = to_paged_hnd(packed_swa, page_size=DSV4_PAGE_SIZE)

    compressed_pages = None
    if comp_tokens.numel() > 0 and comp_idx.numel() > 0:
        comp_flat = comp_tokens.reshape(b * comp_tokens.shape[1], d)
        packed_comp = pack_dsv4_kv_bf16(comp_flat, act_quant_fn=act_quant)
        compressed_pages = to_paged_hnd(packed_comp, page_size=DSV4_PAGE_SIZE)

    if workspace is None:
        workspace = getattr(backend, "workspace_buffer", None)
    if workspace is None:
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)

    sinks = attn_sink
    if sinks is not None and sinks.dim() == 1:
        # FI may want [H] or [B, H]; keep 1d heads.
        sinks = sinks.to(dtype=torch.float32)

    kwargs = dict(
        query=q.contiguous(),
        swa_kv_cache=swa_pages.contiguous(),
        workspace_buffer=workspace,
        sparse_indices=swa_idx.contiguous(),
        compressed_kv_cache=compressed_pages,
        bmm1_scale=float(softmax_scale),
        bmm2_scale=1.0,
        sinks=sinks,
        kv_layout="HND",
        swa_topk_lens=swa_lens.contiguous(),
        extra_sparse_indices=comp_idx.contiguous() if comp_idx.numel() else None,
        extra_sparse_topk_lens=comp_lens.contiguous() if comp_idx.numel() else None,
    )
    out = backend.sparse_mla_decode_dsv4(**kwargs)
    # Expect [B, 1, H, 512] or [B*1, H, 512]
    if out.dim() == 3:
        out = out.view(b, 1, h, d)
    return out


def attach_v4_sparse_mla(backend: Any, *, window_size: int = 128) -> bool:
    """Monkey-patch official ``model.sparse_attn`` / ``kernel.sparse_attn``.

    Returns True if the FlashInfer SM120 path is armed (may still fall back
    per-call). Returns False if capability is official-only (no patch needed
    beyond documenting fallback).
    """
    try:
        import kernel as kernel_mod  # type: ignore
        import model as model_mod  # type: ignore
    except ImportError as e:
        logger.warning("attach_v4_sparse_mla: official modules not importable: %s", e)
        return False

    orig = getattr(model_mod, "sparse_attn", None) or getattr(
        kernel_mod, "sparse_attn", None
    )
    if orig is None:
        logger.warning("attach_v4_sparse_mla: sparse_attn symbol missing")
        return False

    smla = getattr(backend, "sparse_mla_backend", None)
    use_fi = smla in (
        SparseMlaBackend.FLASHINFER_SPARSE_SM120,
        SparseMlaBackend.FLASHINFER_SPARSE_SM100,
    )
    if not use_fi:
        # Keep official path; wire OFFICIAL_SPARSE_ATTN through backend for tests.
        backend._official_sparse_attn = orig
        logger.info(
            "v4 sparse MLA: using official sparse_attn (%s)",
            getattr(smla, "value", smla),
        )
        return False

    stats = {"fi": 0, "fallback": 0, "disabled_zero_out": False}
    # Opt-out if FI returns an all-zero tensor (seen with FI 0.6.16 + cubin 0.6.12).
    fi_disabled = {"v": False}

    def routed(q, kv, attn_sink, topk_idxs, softmax_scale):
        # Decode only; prefill keeps TileLang.
        if q.shape[1] != 1 or fi_disabled["v"]:
            return orig(q, kv, attn_sink, topk_idxs, softmax_scale)
        try:
            out = sparse_mla_decode_from_official_tensors(
                backend,
                q,
                kv,
                attn_sink,
                topk_idxs,
                float(softmax_scale),
                window_size=window_size,
                workspace=getattr(backend, "workspace_buffer", None),
            )
            if not torch.isfinite(out).all() or float(out.detach().float().abs().max()) < 1e-6:
                fi_disabled["v"] = True
                stats["disabled_zero_out"] = True
                stats["fallback"] += 1
                logger.warning(
                    "SM120 sparse MLA produced empty/non-finite output; "
                    "disabling FI path for this process (use official sparse_attn). "
                    "Check flashinfer/cubin version match and JIT build."
                )
                return orig(q, kv, attn_sink, topk_idxs, softmax_scale)
            stats["fi"] += 1
            return out
        except Exception as e:
            stats["fallback"] += 1
            if stats["fallback"] <= 3:
                logger.warning(
                    "SM120 sparse MLA fallback to official sparse_attn: %s", e
                )
            return orig(q, kv, attn_sink, topk_idxs, softmax_scale)

    routed._sglang_lite_stats = stats  # type: ignore[attr-defined]
    kernel_mod.sparse_attn = routed
    model_mod.sparse_attn = routed
    backend._official_sparse_attn = orig
    backend._v4_sparse_attn_routed = routed
    logger.info(
        "v4 sparse MLA: FlashInfer path armed (%s) for decode; prefill=official",
        getattr(smla, "value", smla),
    )
    return True
