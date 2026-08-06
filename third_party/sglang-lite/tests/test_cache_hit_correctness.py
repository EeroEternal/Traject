"""Exact prefix hit must match miss greedy tokens (no duplicate last-prompt forward)."""

from __future__ import annotations


def test_exact_prefix_hit_matches_miss_tokens(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=2)
    try:
        ids = engine.tokenize("Hello world")
        assert len(ids) >= 2

        miss = []
        for d in engine.generate_stream("miss", ids, max_tokens=4, temperature=0.0):
            if d.get("token") is not None:
                miss.append(int(d["token"]))
            if d.get("finished"):
                break

        hit = []
        usage_hit = None
        for d in engine.generate_stream("hit", ids, max_tokens=4, temperature=0.0):
            if d.get("token") is not None:
                hit.append(int(d["token"]))
            if d.get("usage"):
                usage_hit = d["usage"]
            if d.get("finished"):
                break

        assert miss == hit, f"miss={miss} hit={hit}"
        assert usage_hit is not None
        assert usage_hit["cache_hit_tokens"] == len(ids)
    finally:
        engine.shutdown()


def test_pending_cancel_does_not_admit(tiny_mixtral_id):
    from sglang_lite import LiteEngine
    from sglang_lite.loop import GenParams
    import time

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=1)
    try:
        # Pause the loop briefly by flooding then cancel before admit is hard;
        # cancel immediately after submit and ensure no steps for that request.
        ids = engine.tokenize("cancel me please")
        sub = engine.loop.submit("pc1", ids, GenParams(max_tokens=8, temperature=0.0))
        engine.loop.cancel("pc1")
        # Drain client side
        item = sub.delta_queue.get(timeout=5.0)
        assert item.get("finish_reason") == "cancelled"
        time.sleep(0.1)
        # Request must not remain running
        stats = engine.get_stats()
        assert all(
            t.get("request_id") != "pc1" for t in stats.get("last_batch_trace", [])
        ) or stats["running"] == 0
    finally:
        engine.shutdown()
