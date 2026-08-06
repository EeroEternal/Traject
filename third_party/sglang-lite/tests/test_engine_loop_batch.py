"""M3: concurrent requests share one continuous batching loop."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def test_concurrent_requests_multi_batch(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=16)
    try:
        base = engine.tokenize("batch")

        def one(i: int):
            ids = base + [ (i % 50) + 10 ]
            return engine.generate(
                f"r{i}",
                ids,
                max_tokens=2,
                temperature=0.0,
            )

        n = 32
        with ThreadPoolExecutor(max_workers=32) as pool:
            futs = [pool.submit(one, i) for i in range(n)]
            results = [f.result(timeout=120) for f in as_completed(futs)]
        assert len(results) == n
        assert all(r.get("finish_reason") in ("stop", "length") for r in results)
        stats = engine.get_stats()
        assert stats["steps"] > 0
        # Prove at least one scheduler step saw multiple requests
        assert stats["multi_request_batches"] >= 1, stats
    finally:
        engine.shutdown()


def test_no_global_generation_lock(tiny_mixtral_id):
    """Overlapping submits make progress without serializing on a process-wide lock."""
    from sglang_lite import LiteEngine
    import time

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=8)
    try:
        started = []
        lock = threading.Lock()

        def worker(i):
            with lock:
                started.append(time.time())
            return engine.generate(
                f"g{i}",
                engine.tokenize(f"hello {i}"),
                max_tokens=2,
                temperature=0.0,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(worker, range(8)))
        # All submits happened close together (not fully serial wall times of generate)
        assert max(started) - min(started) < 2.0
        assert engine.get_stats()["multi_request_batches"] >= 1
    finally:
        engine.shutdown()
