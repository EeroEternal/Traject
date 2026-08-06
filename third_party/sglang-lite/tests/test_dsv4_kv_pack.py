"""Unit tests for DSV4 584-byte KV packing (CPU, no FlashInfer)."""

from __future__ import annotations

import torch

from sglang_lite.dsv4_kv_pack import (
    DSV4_PACKED_BYTES,
    DSV4_PAGE_SIZE,
    pack_dsv4_kv_bf16,
    split_swa_compress_indices,
    to_paged_hnd,
)


def test_pack_shape_and_rope_bytes():
    kv = torch.randn(3, 512, dtype=torch.bfloat16)
    # Distinct rope pattern for byte check
    kv[:, 448:] = torch.arange(64, dtype=torch.bfloat16)
    packed = pack_dsv4_kv_bf16(kv)
    assert packed.shape == (3, DSV4_PACKED_BYTES)
    assert packed.dtype == torch.uint8
    rope_u8 = kv[:, 448:].contiguous().view(torch.uint8).view(3, 128)
    assert torch.equal(packed[:, 448:576], rope_u8)
    assert (packed[:, 583] == 0).all()


def test_to_paged_pads_to_page_size():
    packed = torch.zeros(10, DSV4_PACKED_BYTES, dtype=torch.uint8)
    pages = to_paged_hnd(packed, page_size=DSV4_PAGE_SIZE)
    assert pages.shape == (1, 1, DSV4_PAGE_SIZE, DSV4_PACKED_BYTES)


def test_split_swa_compress_indices():
    # window=4, plus 2 compress cols (already offset by window in official layout)
    topk = torch.tensor(
        [[[0, 1, 2, -1, 4, 5]]], dtype=torch.int32
    )  # [1,1,6]
    swa, swa_l, comp, comp_l = split_swa_compress_indices(topk, window_size=4)
    assert swa.shape == (1, 4)
    assert int(swa_l[0]) == 3
    assert torch.equal(comp[0], torch.tensor([0, 1], dtype=torch.int32))
    assert int(comp_l[0]) == 2
