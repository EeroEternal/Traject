"""KV layout descriptors for MHA / MLA / DSV4 packed pools."""

from __future__ import annotations

import torch

from sglang_lite.kv_cache import KvLayout, KvLayoutKind, RadixCache


def test_mha_layout_default_cache():
    cache = RadixCache(
        max_tokens=64,
        block_size=16,
        num_layers=2,
        num_kv_heads=4,
        head_dim=64,
        device="cpu",
        dtype=torch.float32,
    )
    assert cache.layout.kind == KvLayoutKind.MHA
    assert cache.ckv_cache is None
    assert cache.k_cache.shape == (2, 4, 16, 4, 64)


def test_mla_compressed_allocates_latent_pages():
    layout = KvLayout.mla_compressed(ckv_dim=512, kpe_dim=64)
    cache = RadixCache(
        max_tokens=64,
        block_size=16,
        num_layers=2,
        num_kv_heads=1,
        head_dim=512,
        device="cpu",
        dtype=torch.float32,
        layout=layout,
    )
    assert cache.ckv_cache is not None
    assert cache.kpe_cache is not None
    assert cache.ckv_cache.shape[-1] == 512
    assert cache.kpe_cache.shape[-1] == 64


def test_dsv4_packed_allocates_uint8_pool():
    layout = KvLayout.dsv4_packed(584)
    cache = RadixCache(
        max_tokens=64,
        block_size=64,
        num_layers=1,
        num_kv_heads=1,
        head_dim=512,
        device="cpu",
        dtype=torch.bfloat16,
        layout=layout,
        swa_layout=layout,
    )
    assert cache.packed_kv_cache is not None
    assert cache.packed_kv_cache.dtype == torch.uint8
    assert cache.packed_kv_cache.shape[-1] == 584
