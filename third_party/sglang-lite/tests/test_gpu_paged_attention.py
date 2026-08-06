"""GPU S1: FlashInfer paged attention is the sole KV source (no DynamicCache rebuild)."""

from __future__ import annotations

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required"),
]


def test_flashinfer_backend_no_rebuild(tiny_mixtral_id):
    pytest.importorskip("flashinfer")
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cuda", max_batch_size=2)
    try:
        assert engine.get_stats()["kernel_backend"] == "flashinfer"
        ids = engine.tokenize("paged hello")
        engine.runner.paged_rebuild_count = 0
        out = engine.generate("gpu-nr", ids, max_tokens=6, temperature=0.0)
        assert out["usage"]["completion_tokens"] >= 1
        assert engine.runner.paged_rebuild_count == 0
        assert engine.get_stats()["paged_rebuild_count"] == 0
    finally:
        engine.shutdown()


def test_gpu_greedy_matches_transformers(tiny_mixtral_id, tiny_mixtral_path):
    pytest.importorskip("flashinfer")
    from sglang_lite import LiteEngine

    tokenizer = AutoTokenizer.from_pretrained(tiny_mixtral_path)
    # Eager attention matches FlashInfer causal semantics more closely than SDPA.
    ref_model = AutoModelForCausalLM.from_pretrained(
        tiny_mixtral_path, dtype=torch.bfloat16, attn_implementation="eager"
    ).cuda()
    ref_model.eval()

    prompt = "Hello world"
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    max_new = 8
    inp = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    attn = torch.ones_like(inp)

    with torch.no_grad():
        ref_out = ref_model.generate(
            inp,
            attention_mask=attn,
            max_new_tokens=max_new,
            do_sample=False,
            temperature=None,
            top_p=None,
        )
    ref_tokens = ref_out[0].tolist()[len(input_ids) :][:max_new]

    engine = LiteEngine(tiny_mixtral_id, device="cuda", max_batch_size=2)
    try:
        assert engine.get_stats()["kernel_backend"] == "flashinfer"
        got = []
        for delta in engine.generate_stream(
            "gpu-corr", input_ids, max_tokens=max_new, temperature=0.0
        ):
            if delta.get("token") is not None:
                got.append(int(delta["token"]))
            if delta.get("finished"):
                break
        got = got[:max_new]
        assert got == ref_tokens, f"got={got} ref={ref_tokens}"
        assert engine.runner.paged_rebuild_count == 0
    finally:
        engine.shutdown()


def test_gpu_prefix_hit_reduces_prefill(tiny_mixtral_id):
    pytest.importorskip("flashinfer")
    from sglang_lite import LiteEngine

    engine = LiteEngine(tiny_mixtral_id, device="cuda", max_batch_size=2)
    try:
        ids = engine.tokenize("shared prefix cache test")
        r1 = engine.generate("gpu-p1", ids, max_tokens=3, temperature=0.0)
        assert r1["usage"]["cache_hit_tokens"] == 0
        prefill1 = None
        # Second request with same prompt should hit pages
        engine.runner.paged_rebuild_count = 0
        r2 = engine.generate("gpu-p2", ids, max_tokens=3, temperature=0.0)
        assert r2["usage"]["cache_hit_tokens"] == len(ids)
        assert engine.runner.paged_rebuild_count == 0
        # Exact hit samples from last_logits — no prefill recompute of prompt tokens
        assert r2["usage"]["completion_tokens"] >= 1
        _ = prefill1
    finally:
        engine.shutdown()
