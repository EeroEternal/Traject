"""Mock tests for DeepSeek-V4 Hybrid forward adapter (no weights)."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest
import torch

from sglang_lite.kv_cache import RadixCache
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence


class _FakeV4(torch.nn.Module):
    """Mimics official Transformer.forward(input_ids, start_pos) → [B, vocab]."""

    def __init__(self, vocab: int = 32):
        super().__init__()
        self.vocab = vocab
        self.calls: List[tuple] = []

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        self.calls.append((tuple(input_ids.shape), int(start_pos)))
        b = input_ids.shape[0]
        logits = torch.zeros(b, self.vocab)
        # Prefer token id 3 so sampling is deterministic under greedy.
        logits[:, 3] = 10.0
        return logits


def _make_hybrid_runner(fake: _FakeV4) -> ModelRunner:
    from sglang_lite.v4_prefix_cache import V4PrefixCache

    runner = ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=4)
    runner.model = fake
    runner._is_real = True
    runner._v4_hybrid = True
    runner._v4_prefix_cache = V4PrefixCache()
    runner.use_paged_as_source = False
    runner.vocab_size = fake.vocab
    runner.eos_token_id = 2
    runner.kernel_backend = MagicMock()
    runner.kernel_backend.supports_paged_attention = True
    return runner


def _seq(ids: List[int], cached_len: int = 0) -> Sequence:
    return Sequence(
        seq_id=0,
        request_id="r0",
        input_ids=list(ids),
        max_tokens=8,
        temperature=0.0,
        cached_len=cached_len,
    )


def test_use_paged_attention_false_for_v4_hybrid():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    assert runner._use_paged_attention() is False


def test_v4_prefill_calls_start_pos_and_skips_pages():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    radix = RadixCache(
        max_tokens=256,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    seq = _seq([10, 11, 12], cached_len=0)
    results: List[Optional[int]] = [None]
    runner._batch_prefill([seq], [0], radix, results)
    assert fake.calls == [((1, 3), 0)]
    assert seq.cached_len == 3
    assert seq.block_table == []
    assert results[0] == 3
    assert len(radix._allocated_blocks) == 0


def test_v4_decode_uses_cached_len_as_start_pos():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    radix = RadixCache(
        max_tokens=256,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )
    seq = _seq([10, 11, 12], cached_len=3)
    seq.output_ids = [7]
    results: List[Optional[int]] = [None]
    runner._batch_decode([seq], [0], radix, results)
    assert fake.calls == [((1, 1), 3)]
    assert seq.cached_len == 4
    assert results[0] == 3


def test_model_forward_v4_accepts_3d_logits():
    class _ThreeD(torch.nn.Module):
        def forward(self, input_ids, start_pos=0):
            b, t = input_ids.shape
            out = torch.zeros(b, t, 16)
            out[:, -1, 5] = 10.0
            return out

    runner = _make_hybrid_runner(_FakeV4())
    runner.model = _ThreeD()
    ids = torch.tensor([[1, 2, 3]])
    logits = runner._model_forward_v4(ids, start_pos=0)
    assert logits.shape == (1, 16)
    assert int(logits.argmax()) == 5


def test_admit_ignores_radix_prefix_for_v4():
    from sglang_lite.loop import EngineLoop, GenParams

    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    loop = EngineLoop(runner, max_batch_size=2)
    assert loop.ready is False
    # Poison radix with a fake tree hit — Hybrid must ignore it (no V4 snapshot).
    loop.radix.insert_or_update(
        [1, 2, 3, 4],
        None,
        4,
        block_ids=[],
        last_logits=torch.zeros(32),
    )
    loop.submit("a", [1, 2, 3, 4], GenParams(max_tokens=2, temperature=0.0))
    loop._admit_pending()
    seq = loop.scheduler.waiting[0]
    assert seq.cached_len == 0
    assert seq.cache_hit_tokens == 0
    assert seq.last_logits is None
