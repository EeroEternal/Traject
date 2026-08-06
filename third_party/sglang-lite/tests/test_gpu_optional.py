"""CUDA / FlashInfer / popular MoE checks — skipped when hardware/deps absent."""

from __future__ import annotations

import os

import pytest
import torch


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


def test_cuda_tiny_mixtral_greedy(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cuda", max_batch_size=2)
    try:
        ids = engine.tokenize("cuda hello")
        out = engine.generate("g1", ids, max_tokens=4, temperature=0.0)
        assert out["finish_reason"] in ("stop", "length")
        assert out["usage"]["completion_tokens"] >= 1
    finally:
        engine.shutdown()


@pytest.mark.skipif(
    os.environ.get("SGLANG_LITE_POPULAR_MOE") is None,
    reason="Set SGLANG_LITE_POPULAR_MOE=<hub-id> to run popular MoE smoke",
)
def test_popular_moe_smoke():
    from sglang_lite import LiteEngine

    model = os.environ["SGLANG_LITE_POPULAR_MOE"]
    engine = LiteEngine(model, device="cuda" if torch.cuda.is_available() else "cpu", max_batch_size=1)
    try:
        ids = engine.tokenize("Hello")
        out = engine.generate("pop", ids, max_tokens=8, temperature=0.0)
        assert out["usage"]["completion_tokens"] >= 1
    finally:
        engine.shutdown()


def test_flashinfer_import_on_cuda():
    pytest.importorskip("flashinfer")
    import flashinfer  # noqa: F401

    assert torch.cuda.is_available()
