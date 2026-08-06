"""M1: greedy/fixed-seed output matches Transformers reference on tiny Mixtral."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def test_greedy_matches_transformers(tiny_mixtral_id, tiny_mixtral_path):
    from sglang_lite import LiteEngine

    tokenizer = AutoTokenizer.from_pretrained(tiny_mixtral_path)
    ref_model = AutoModelForCausalLM.from_pretrained(tiny_mixtral_path)
    ref_model.eval()

    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_new = 8

    with torch.no_grad():
        ref_out = ref_model.generate(
            torch.tensor([input_ids], dtype=torch.long),
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    ref_tokens = ref_out[0].tolist()[len(input_ids) :]

    engine = LiteEngine(tiny_mixtral_id, device="cpu", max_batch_size=2, allow_stub=False)
    try:
        got = []
        for delta in engine.generate_stream(
            "corr-1", input_ids, max_tokens=max_new, temperature=0.0
        ):
            if delta.get("token") is not None:
                got.append(int(delta["token"]))
            if delta.get("finished"):
                break
        got = got[:max_new]
        ref_tokens = ref_tokens[:max_new]
        assert got == ref_tokens, f"got={got} ref={ref_tokens}"
    finally:
        engine.shutdown()


def test_load_failure_does_not_fallback_to_stub():
    from sglang_lite.runner import ModelRunner
    import pytest

    with pytest.raises(Exception):
        ModelRunner("fixture:/nonexistent/path/moe", device="cpu", allow_stub=False)


def test_stub_requires_allow_flag():
    from sglang_lite.runner import ModelRunner
    import pytest

    with pytest.raises(ValueError):
        ModelRunner("stub", device="cpu", allow_stub=False)
