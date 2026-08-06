"""Unit tests for DeepSeek-V4 Hybrid prefix snapshot/restore (no weights)."""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import torch
import torch.nn as nn

from sglang_lite.kv_cache import RadixCache
from sglang_lite.loop import EngineLoop, GenParams
from sglang_lite.runner import ModelRunner
from sglang_lite.scheduler import Sequence
from sglang_lite.v4_prefix_cache import (
    V4PrefixCache,
    clear_v4_kv_slot,
    restore_v4_kv,
    snapshot_v4_kv,
)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.kv_cache = torch.zeros(2, 4, 8)
        self.kv_state = torch.zeros(2, 3)
        self.score_state = torch.zeros(2, 1)


class _FakeV4(nn.Module):
    def __init__(self, vocab: int = 32):
        super().__init__()
        self.vocab = vocab
        self.attn = _Attn()
        self.calls: List[tuple] = []

    def forward(self, input_ids: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        self.calls.append((tuple(input_ids.shape), int(start_pos)))
        # Mutate slot 0 so snapshots differ across prefills.
        self.attn.kv_cache[0, 0, 0] = float(start_pos + input_ids.shape[1])
        b = input_ids.shape[0]
        logits = torch.zeros(b, self.vocab)
        logits[:, 3] = 10.0
        return logits


def _make_hybrid_runner(fake: _FakeV4) -> ModelRunner:
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


def _radix() -> RadixCache:
    return RadixCache(
        max_tokens=256,
        block_size=16,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float32,
        device="cpu",
    )


def test_match_longest_exact_prefix_only():
    cache = V4PrefixCache()
    buf = {"attn.kv_cache": torch.ones(1, 2)}
    cache.insert([1, 2, 3], last_logits=torch.zeros(4), buffers=buf)
    cache.insert([1, 2, 3, 4, 5], last_logits=torch.zeros(4), buffers=buf)
    n, e = cache.match([1, 2, 3, 4, 9])
    assert n == 3
    assert e is not None
    assert e.token_ids == [1, 2, 3]
    n2, e2 = cache.match([9, 1, 2])
    assert n2 == 0 and e2 is None


def test_snapshot_restore_roundtrip():
    m = _FakeV4()
    m.attn.kv_cache[0] = 7.0
    m.attn.kv_state[0] = 2.0
    snap = snapshot_v4_kv(m, batch_slot=0)
    assert "attn.kv_cache" in snap
    m.attn.kv_cache.zero_()
    m.attn.kv_state.zero_()
    n = restore_v4_kv(m, snap, batch_slot=0)
    assert n >= 2
    assert torch.allclose(m.attn.kv_cache[0], torch.full((4, 8), 7.0))
    assert torch.allclose(m.attn.kv_state[0], torch.full((3,), 2.0))


def test_clear_v4_kv_slot_zeros_row():
    m = _FakeV4()
    m.attn.kv_cache[0] = 3.0
    m.attn.kv_cache[1] = 9.0
    m.attn.kv_state[0] = 4.0
    n = clear_v4_kv_slot(m, batch_slot=0)
    assert n >= 2
    assert torch.count_nonzero(m.attn.kv_cache[0]) == 0
    assert torch.allclose(m.attn.kv_cache[1], torch.full((4, 8), 9.0))
    assert torch.count_nonzero(m.attn.kv_state[0]) == 0


def test_prefill_saves_and_exact_hit_skips_forward():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    radix = _radix()
    seq1 = _seq([10, 11, 12], cached_len=0)
    results: List[Optional[int]] = [None]
    runner._batch_prefill([seq1], [0], radix, results)
    assert len(runner._v4_prefix_cache) == 1
    assert fake.calls == [((1, 3), 0)]
    fwd = runner.model_forward_count

    # Exact hit: restore + sample last_logits, no new forward in prefill.
    match_len, entry = runner.v4_match_prefix([10, 11, 12])
    assert match_len == 3
    seq2 = _seq([10, 11, 12], cached_len=3)
    seq2.cache_hit_tokens = 3
    seq2.last_logits = entry.last_logits
    seq2._v4_prefix_entry = entry
    seq2._v4_kv_pending_restore = True
    results2: List[Optional[int]] = [None]
    runner._batch_prefill([seq2], [0], radix, results2)
    assert runner.model_forward_count == fwd
    assert seq2.prefill_tokens == 0
    assert seq2._v4_kv_pending_restore is False


def test_partial_hit_shortens_prefill_one_token_suffix():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    radix = _radix()
    seq1 = _seq([10, 11, 12], cached_len=0)
    results: List[Optional[int]] = [None]
    runner._batch_prefill([seq1], [0], radix, results)
    match_len, entry = runner.v4_match_prefix([10, 11, 12, 13, 14])
    assert match_len == 3
    seq2 = _seq([10, 11, 12, 13, 14], cached_len=3)
    seq2.cache_hit_tokens = 3
    seq2.last_logits = entry.last_logits
    seq2._v4_prefix_entry = entry
    seq2._v4_kv_pending_restore = True
    fake.calls.clear()
    results2: List[Optional[int]] = [None]
    runner._batch_prefill([seq2], [0], radix, results2)
    # start_pos>0 ⇒ one-token steps for the 2-token suffix.
    assert fake.calls == [((1, 1), 3), ((1, 1), 4)]
    assert seq2.prefill_tokens == 2
    assert seq2.cached_len == 5


def test_admit_uses_v4_prefix_not_radix():
    fake = _FakeV4()
    runner = _make_hybrid_runner(fake)
    loop = EngineLoop(runner, max_batch_size=2)
    loop.radix.insert_or_update(
        [1, 2, 3, 4],
        None,
        4,
        block_ids=[],
        last_logits=torch.zeros(32),
    )
    # No V4 snapshot yet → miss despite Radix hit.
    loop.submit("a", [1, 2, 3, 4], GenParams(max_tokens=2, temperature=0.0))
    loop._admit_pending()
    seq = loop.scheduler.waiting[0]
    assert seq.cached_len == 0
    assert seq.cache_hit_tokens == 0

    buffers = snapshot_v4_kv(fake, batch_slot=0)
    runner._v4_prefix_cache.insert(
        [1, 2, 3, 4],
        last_logits=torch.zeros(32),
        buffers=buffers,
    )
    loop.submit("b", [1, 2, 3, 4, 5], GenParams(max_tokens=2, temperature=0.0))
    loop._admit_pending()
    seq_b = loop.scheduler.waiting[1]
    assert seq_b.cache_hit_tokens == 4
    assert seq_b.cached_len == 4
    assert seq_b._v4_kv_pending_restore is True
