"""MoE model registry for sglang-lite.

Only popular MoE families are in scope. Dense models are rejected.
Verified ids are those successfully loaded in this process (plus explicit register).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set


@dataclass(frozen=True)
class MoEFamily:
    name: str
    model_types: frozenset
    example_ids: frozenset


MIXTRAL = MoEFamily(
    name="mixtral",
    model_types=frozenset({"mixtral"}),
    example_ids=frozenset(
        {
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "mistralai/Mixtral-8x22B-Instruct-v0.1",
        }
    ),
)

QWEN_MOE = MoEFamily(
    name="qwen_moe",
    model_types=frozenset({"qwen2_moe", "qwen3_moe"}),
    example_ids=frozenset(
        {
            "Qwen/Qwen1.5-MoE-A2.7B-Chat",
            "Qwen/Qwen2-57B-A14B-Instruct",
        }
    ),
)

DEEPSEEK_MOE = MoEFamily(
    name="deepseek_moe",
    model_types=frozenset({"deepseek_v2", "deepseek_v3", "deepseek_moe"}),
    example_ids=frozenset(
        {
            "deepseek-ai/DeepSeek-V2-Lite-Chat",
            "deepseek-ai/DeepSeek-V2-Chat",
        }
    ),
)

DEEPSEEK_V4 = MoEFamily(
    name="deepseek_v4",
    model_types=frozenset({"deepseek_v4"}),
    example_ids=frozenset(
        {
            "deepseek-ai/DeepSeek-V4-Flash",
            "local:ds-v4-flash",
        }
    ),
)

FAMILIES: List[MoEFamily] = [MIXTRAL, QWEN_MOE, DEEPSEEK_MOE, DEEPSEEK_V4]

# Only ids that successfully loaded (or were explicitly registered) this process.
_VERIFIED: Set[str] = set()


def is_fixture_model(model_id: str) -> bool:
    return model_id.startswith("fixture:") or model_id.startswith("local:")


def register_verified(model_id: str) -> None:
    """Mark a model id as verified for this process (after load succeeds)."""
    _VERIFIED.add(model_id)


def list_verified_models() -> List[str]:
    return sorted(_VERIFIED)


def known_example_ids() -> List[str]:
    """Documented example hub ids (not automatically advertised)."""
    out: Set[str] = set()
    for fam in FAMILIES:
        out |= set(fam.example_ids)
    return sorted(out)


def family_for_model_type(model_type: Optional[str]) -> Optional[MoEFamily]:
    if not model_type:
        return None
    mt = model_type.lower()
    for fam in FAMILIES:
        if mt in fam.model_types:
            return fam
    return None


def assert_moe_supported(model_id: str, model_type: Optional[str] = None) -> MoEFamily:
    """Raise ValueError if the model is not an allowed MoE family."""
    if is_fixture_model(model_id) or model_id == "stub":
        return MIXTRAL

    fam = family_for_model_type(model_type)
    if fam is not None:
        return fam

    lower = model_id.lower()
    if "mixtral" in lower:
        return MIXTRAL
    if "qwen" in lower and "moe" in lower:
        return QWEN_MOE
    if "deepseek" in lower and "v4" in lower:
        return DEEPSEEK_V4
    if "ds-v4" in lower or "deepseek-v4" in lower:
        return DEEPSEEK_V4
    if "deepseek" in lower and ("moe" in lower or "v2" in lower or "v3" in lower):
        return DEEPSEEK_MOE

    raise ValueError(
        f"model '{model_id}' is not a supported MoE family "
        f"(mixtral / qwen-moe / deepseek-moe / deepseek_v4). Dense models are out of scope."
    )


def default_serving_model_list() -> List[str]:
    return list_verified_models()
