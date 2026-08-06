"""ModelLoader TP shard plan tests (no GPU / no weights required)."""

from __future__ import annotations

import pytest

from sglang_lite.model_loader import build_tp_shard_plan, default_tp8_plans, find_rank_shard
from sglang_lite.models import DEEPSEEK_V4, assert_moe_supported, family_for_model_type


def test_tp8_expert_split_covers_all():
    plans = default_tp8_plans(256)
    assert len(plans) == 8
    covered = []
    for p in plans:
        assert p.n_local_experts == 32
        covered.extend(p.expert_ids())
    assert covered == list(range(256))


def test_tp_plan_rejects_bad_rank():
    with pytest.raises(ValueError):
        build_tp_shard_plan(8, 8, n_experts=256)


def test_tp_plan_requires_divisible_experts():
    with pytest.raises(ValueError):
        build_tp_shard_plan(8, 0, n_experts=255)


def test_deepseek_v4_family_registered():
    assert family_for_model_type("deepseek_v4") is DEEPSEEK_V4
    fam = assert_moe_supported("deepseek-ai/DeepSeek-V4-Flash", "deepseek_v4")
    assert fam.name == "deepseek_v4"
    fam2 = assert_moe_supported("/data/ds-v4-flash")
    assert fam2.name == "deepseek_v4"


def test_find_rank_shard_naming(tmp_path):
    p = tmp_path / "model3-mp8.safetensors"
    p.write_bytes(b"x")
    assert find_rank_shard(tmp_path, 3, 8) == p
