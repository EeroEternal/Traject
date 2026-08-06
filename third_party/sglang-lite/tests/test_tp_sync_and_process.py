"""TP sync helpers + EngineLoop.mark_ready (no GPU)."""

from sglang_lite.loop import EngineLoop, GenParams
from sglang_lite.runner import ModelRunner
from sglang_lite.tp_sync import is_tp, rank, world_size


def test_tp_sync_defaults_single_process():
    assert world_size() == 1
    assert rank() == 0
    assert is_tp() is False


def test_mark_ready_without_background_thread():
    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)
    loop = EngineLoop(runner, max_batch_size=2)
    assert loop.ready is False
    loop.mark_ready()
    assert loop.ready is True
    # Admission works without a background thread; caller owns pump_until_idle.
    loop.submit("r0", [1, 2, 3], GenParams(max_tokens=1, temperature=0.0))
    loop._admit_pending()
    assert len(loop.scheduler.waiting) == 1
