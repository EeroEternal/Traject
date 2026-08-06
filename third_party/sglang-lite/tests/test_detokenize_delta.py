"""Incremental detokenize must not re-emit full text or lone U+FFFD."""

from __future__ import annotations

from sglang_lite.runner import ModelRunner


def _runner() -> ModelRunner:
    return ModelRunner(model_name="stub", device="cpu", allow_stub=True, max_batch=2)


def test_extends_prev_emits_suffix_only():
    r = _runner()
    r.detokenize = lambda ids: {1: "Hell", 2: "Hello", 3: "Hello!"}[len(ids)]  # type: ignore[method-assign]
    assert r.detokenize_delta([1], "") == "Hell"
    assert r.detokenize_delta([1, 2], "Hell") == "o"
    assert r.detokenize_delta([1, 2, 3], "Hello") == "!"


def test_incomplete_multibyte_waits():
    r = _runner()
    # Tokenizer briefly shortens while assembling a glyph.
    r.detokenize = lambda ids: "你好" if len(ids) < 3 else "你好！"  # type: ignore[method-assign]
    assert r.detokenize_delta([1], "你好？") == ""
    assert r.detokenize_delta([1, 2, 3], "你好") == "！"


def test_divergent_rewrite_does_not_dump_full():
    r = _runner()
    r.detokenize = lambda ids: "完全不同的句子"  # type: ignore[method-assign]
    assert r.detokenize_delta([1, 2], "previous stream") == ""


def test_lcp_emits_stable_suffix():
    r = _runner()
    r.detokenize = lambda ids: "你好！"  # type: ignore[method-assign]
    assert r.detokenize_delta([1], "你好X") == "！"


def test_lone_replacement_char_suppressed():
    r = _runner()
    r.detokenize = lambda ids: "ab\ufffd" if len(ids) == 1 else "ab😊"  # type: ignore[method-assign]
    assert r.detokenize_delta([1], "ab") == ""
    assert r.detokenize_delta([1, 2], "ab") == "😊"


def test_empty_ids():
    r = _runner()
    assert r.detokenize_delta([], "x") == ""
