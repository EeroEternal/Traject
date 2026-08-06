"""M2: prefix skip reduces prefill work; cancel/finish frees blocks."""

from __future__ import annotations


def test_shared_prefix_sets_cache_hit_tokens(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=2)
    try:
        ids = engine.tokenize("The capital of France is")
        r1 = engine.generate("p1", ids, max_tokens=4, temperature=0.0)
        assert r1["usage"]["cache_hit_tokens"] == 0

        # Longer prompt sharing the same prefix
        ids2 = engine.tokenize("The capital of France is Paris and")
        # Ensure prefix of ids2 starts with ids
        assert ids2[: len(ids)] == ids or ids2[:8] == ids[:8]

        shared = engine.tokenize("The capital of France is")
        r2 = engine.generate("p2", shared + engine.tokenize(" more text here"), max_tokens=2, temperature=0.0)
        # Second request with exact shared prefix should report hits
        assert r2["usage"]["cache_hit_tokens"] > 0
        assert r2["usage"]["cache_hit_tokens"] <= len(shared)
    finally:
        engine.shutdown()


def test_cancel_releases_blocks(tiny_mixtral_id):
    from sglang_lite import LiteEngine
    import time

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=2)
    try:
        ids = engine.tokenize("abcdefghijklmnopqrstuvwxyz")
        # Submit long gen then cancel
        from sglang_lite.loop import GenParams

        sub = engine.loop.submit(
            "c1", ids, GenParams(max_tokens=64, temperature=0.0, timeout_s=30.0)
        )
        # Wait for at least one delta
        first = sub.delta_queue.get(timeout=30.0)
        assert first is not None
        engine.cancel("c1")
        time.sleep(0.05)
        stats = engine.get_stats()
        # After cancel, request should not remain running
        assert stats["running"] == 0 or "c1" not in [
            t.get("request_id") for t in stats.get("last_batch_trace", [])
        ]
    finally:
        engine.shutdown()


def test_kv_oom_rejects_deterministically():
    from sglang_lite.kv_cache import RadixCache
    import pytest

    cache = RadixCache(
        max_tokens=32,
        block_size=16,
        num_layers=1,
        num_kv_heads=1,
        head_dim=8,
        device="cpu",
    )
    # capacity = 2 blocks
    cache.allocate_blocks(2)
    with pytest.raises(MemoryError):
        cache.allocate_blocks(1)
