"""Model-level batching: one tensor forward covers multiple sequences."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import torch


def test_decode_batch_uses_tensor_batch_dim(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=8)
    try:
        # Warm shared prefix so later decodes share cached_len
        base = engine.tokenize("batch decode")
        engine.generate("warm", base, max_tokens=1, temperature=0.0)
        engine.runner.model_forward_count = 0
        engine.runner.last_model_forward_size = 0

        def one(i: int):
            # Distinct suffixes so prompts differ but after prefill decode lens align
            ids = base + [10 + (i % 20)]
            return engine.generate(f"bd{i}", ids, max_tokens=2, temperature=0.0)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(one, i) for i in range(8)]
            for f in as_completed(futs):
                f.result(timeout=120)

        stats = engine.get_stats()
        assert stats["multi_request_batches"] >= 1
        assert stats["paged_rebuild_count"] >= 1
        assert engine.runner.model_forward_count >= 1

        # Prove multi-seq tensor forward: equal cached_len decode batch on runner.
        from sglang_lite.scheduler import Sequence

        radix = engine.radix
        runner = engine.runner
        seqs = []
        for i in range(4):
            ids = engine.tokenize(f"sync {i}")
            s = Sequence(seq_id=100 + i, request_id=f"s{i}", input_ids=ids)
            tok = runner._prefill_one(s, radix)
            assert tok is not None
            s.output_ids.append(tok)
            seqs.append(s)
        cl = seqs[0].cached_len
        for s in seqs:
            assert s.cached_len == cl
        runner.last_model_forward_size = 0
        results = [None] * 4
        runner._batch_decode(seqs, list(range(4)), radix, results)
        assert runner.last_model_forward_size == 4, runner.last_model_forward_size
        assert all(t is not None for t in results)
    finally:
        engine.shutdown()


def test_paged_kv_roundtrip_is_attention_source(tiny_mixtral_id):
    """If page writes are broken, generation after a hit must fail or diverge — pages are required."""
    from sglang_lite import LiteEngine
    from sglang_lite.kv_cache import RadixCache

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=2)
    try:
        ids = engine.tokenize("page source")
        r1 = engine.generate("ps1", ids, max_tokens=3, temperature=0.0)
        assert r1["usage"]["cache_hit_tokens"] == 0
        rebuilds_before = engine.runner.paged_rebuild_count

        r2 = engine.generate("ps2", ids, max_tokens=3, temperature=0.0)
        assert r2["usage"]["cache_hit_tokens"] == len(ids)
        assert engine.runner.paged_rebuild_count > rebuilds_before

        # Direct round-trip: write/read pages
        cache: RadixCache = engine.radix
        # Use first sequence's committed path
        node = cache.tree.find_node(ids)
        assert node is not None and node.block_ids
        rebuilt = cache.read_kv(node.block_ids, len(ids))
        assert len(rebuilt) == cache.num_layers
        assert rebuilt[0][0].shape[-2] == len(ids)

        # Corrupting pages must change rebuilt tensors (proves reads hit real storage)
        bid = node.block_ids[0]
        before = cache.k_cache[0, bid, 0].clone()
        cache.k_cache[0, bid, 0].zero_()
        after = cache.read_kv(node.block_ids, len(ids))[0][0][0, :, 0, :]
        assert not torch.allclose(before, after)
        cache.k_cache[0, bid, 0].copy_(before)
    finally:
        engine.shutdown()
