"""DynamicCache API compat: Transformers 4.x to_legacy_cache vs 5.x layers."""

from __future__ import annotations

import torch


class _CacheWithoutToLegacy:
    """Mimic Transformers 5.x DynamicCache when to_legacy_cache is absent."""

    def __init__(self, inner):
        self.layers = getattr(inner, "layers", None)
        self._inner = inner

    def __len__(self):
        return len(self._inner)

    def __getitem__(self, i):
        return self._inner[i]

    def get_seq_length(self):
        return int(self._inner.get_seq_length())


def test_to_legacy_kv_layers_fallback(tiny_mixtral_id):
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=1)
    try:
        ids = engine.tokenize("compat")
        with torch.no_grad():
            out = engine.runner.model(
                torch.tensor([ids], dtype=torch.long),
                use_cache=True,
            )
        past = _CacheWithoutToLegacy(out.past_key_values)
        assert not hasattr(past, "to_legacy_cache")
        legacy = engine.runner._to_legacy_kv(past)
        assert legacy is not None and len(legacy) > 0
        assert legacy[0][0].shape[-2] == len(ids)

        r = engine.generate("compat-gen", ids, max_tokens=2, temperature=0.0)
        assert r.get("finish_reason") != "error", r
        assert r.get("text") or r.get("usage", {}).get("completion_tokens", 0) > 0, r
    finally:
        engine.shutdown()
