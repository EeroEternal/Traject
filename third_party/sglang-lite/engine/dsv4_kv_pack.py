"""DeepSeek-V4 packed KV (FlashInfer SM120 sparse MLA, 584 B/token).

Layout (matches vLLM fp8_ds_mla / FlashInfer SM120 docs):
  bytes[0:448]   — NoPE as float8_e4m3 (448 dims)
  bytes[448:576] — RoPE as 64×bfloat16 (128 bytes)
  bytes[576:583] — 7×ue8m0 block scales (block_size=64)
  bytes[583:584] — pad
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

DSV4_HEAD_DIM = 512
DSV4_NOPE_DIM = 448
DSV4_ROPE_DIM = 64
DSV4_PACKED_BYTES = 584
DSV4_SCALE_BYTES = 7
DSV4_FP8_BLOCK = 64
DSV4_PAGE_SIZE = 64


def pack_dsv4_kv_bf16(
    kv: torch.Tensor,
    *,
    act_quant_fn=None,
) -> torch.Tensor:
    """Pack bf16 KV ``[..., 512]`` → uint8 ``[..., 584]``.

    Prefers official ``kernel.act_quant`` when ``act_quant_fn`` is provided
    (scale_fmt=ue8m0) so packing matches the Hybrid model's QAT path.
    """
    if kv.shape[-1] != DSV4_HEAD_DIM:
        raise ValueError(f"expected head_dim={DSV4_HEAD_DIM}, got {kv.shape[-1]}")
    if kv.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise TypeError(f"unsupported kv dtype {kv.dtype}")

    nope = kv[..., :DSV4_NOPE_DIM].contiguous()
    rope = kv[..., DSV4_NOPE_DIM:].contiguous()

    if act_quant_fn is not None:
        y, s = act_quant_fn(
            nope, DSV4_FP8_BLOCK, "ue8m0", torch.float8_e8m0fnu, False
        )
    else:
        y, s = _torch_fp8_ue8m0_quant(nope, DSV4_FP8_BLOCK)

    out = torch.empty(
        *kv.shape[:-1], DSV4_PACKED_BYTES, dtype=torch.uint8, device=kv.device
    )
    out[..., :DSV4_NOPE_DIM] = y.view(torch.uint8).view(*y.shape[:-1], DSV4_NOPE_DIM)
    out[..., DSV4_NOPE_DIM : DSV4_NOPE_DIM + DSV4_ROPE_DIM * 2] = rope.view(
        torch.uint8
    ).view(*rope.shape[:-1], DSV4_ROPE_DIM * 2)
    scale_u8 = s.view(torch.uint8).view(*s.shape[:-1], DSV4_SCALE_BYTES)
    out[..., 576 : 576 + DSV4_SCALE_BYTES] = scale_u8
    out[..., 583] = 0
    return out


def _torch_fp8_ue8m0_quant(
    x: torch.Tensor, block_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback block FP8 + power-of-two ue8m0 scales (no TileLang)."""
    n = x.shape[-1]
    if n % block_size != 0:
        raise ValueError(f"last dim {n} not divisible by block_size={block_size}")
    xf = x.reshape(-1, n).to(torch.float32)
    groups = xf.view(-1, n // block_size, block_size)
    amax = groups.abs().amax(dim=-1).clamp(min=1e-12)
    # Power-of-two scale (ue8m0-style).
    scale = torch.exp2(torch.ceil(torch.log2(amax)))
    y = (groups / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    s = scale.to(torch.float8_e8m0fnu)
    y = y.view(*x.shape[:-1], n)
    s = s.view(*x.shape[:-1], n // block_size)
    return y, s


def to_paged_hnd(
    packed_tokens: torch.Tensor,
    page_size: int = DSV4_PAGE_SIZE,
) -> torch.Tensor:
    """``[T, 584]`` uint8 → ``[num_pages, 1, page_size, 584]`` (HND, H=1)."""
    if packed_tokens.dim() != 2 or packed_tokens.shape[-1] != DSV4_PACKED_BYTES:
        raise ValueError(f"expected [T, 584], got {tuple(packed_tokens.shape)}")
    t = packed_tokens.shape[0]
    pad = (page_size - t % page_size) % page_size
    if pad:
        packed_tokens = torch.cat(
            [
                packed_tokens,
                packed_tokens.new_zeros(pad, DSV4_PACKED_BYTES),
            ],
            dim=0,
        )
    n_pages = packed_tokens.shape[0] // page_size
    return packed_tokens.view(n_pages, 1, page_size, DSV4_PACKED_BYTES)


def split_swa_compress_indices(
    topk_idxs: torch.Tensor,
    window_size: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split official concatenated topk into SWA + compressed (relative) indices.

    ``topk_idxs``: ``[B, S, K]`` int; first ``window_size`` columns are SWA.
    Returns ``(swa_idx, swa_lens, comp_idx, comp_lens)`` with shapes
    ``[B*S, *]`` / ``[B*S]``.
    """
    if topk_idxs.dim() != 3:
        raise ValueError(f"expected [B,S,K], got {tuple(topk_idxs.shape)}")
    b, s, k = topk_idxs.shape
    swa_k = min(window_size, k)
    swa = topk_idxs[:, :, :swa_k].reshape(b * s, swa_k).to(torch.int32)
    swa_lens = (swa >= 0).sum(dim=-1).to(torch.int32)

    if k > swa_k:
        comp = topk_idxs[:, :, swa_k:].reshape(b * s, k - swa_k).to(torch.int32)
        # Official compress indices are offset by window_size into the concat cache.
        valid = comp >= 0
        comp = torch.where(valid, comp - window_size, comp)
        comp_lens = valid.sum(dim=-1).to(torch.int32)
    else:
        comp = topk_idxs.new_zeros(b * s, 0, dtype=torch.int32)
        comp_lens = topk_idxs.new_zeros(b * s, dtype=torch.int32)
    return swa, swa_lens, comp, comp_lens
