"""ModelLoader: shard mapping + Hybrid path for DeepSeek-V4 official inference.

Owns "which shard lands on which rank" for the controlled single-node TP=8
topology. Does not implement full EP / EPLB. Kernel compute stays in KernelBackend.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class TpShardPlan:
    """Single-node tensor-parallel placement."""

    world_size: int
    rank: int
    local_device: str
    n_experts: int
    n_local_experts: int
    expert_start: int
    expert_end: int  # exclusive

    def expert_ids(self) -> List[int]:
        return list(range(self.expert_start, self.expert_end))


def build_tp_shard_plan(
    world_size: int,
    rank: int,
    n_experts: int = 256,
    device_prefix: str = "cuda",
) -> TpShardPlan:
    """Map rank → local expert range (matches official convert.py MP split)."""
    if world_size <= 0:
        raise ValueError("world_size must be > 0")
    if not (0 <= rank < world_size):
        raise ValueError(f"rank {rank} out of range for world_size={world_size}")
    if n_experts % world_size != 0:
        raise ValueError(
            f"n_experts ({n_experts}) must be divisible by world_size ({world_size})"
        )
    n_local = n_experts // world_size
    start = rank * n_local
    return TpShardPlan(
        world_size=world_size,
        rank=rank,
        local_device=f"{device_prefix}:{rank}",
        n_experts=n_experts,
        n_local_experts=n_local,
        expert_start=start,
        expert_end=start + n_local,
    )


def default_tp8_plans(n_experts: int = 256) -> List[TpShardPlan]:
    return [build_tp_shard_plan(8, r, n_experts=n_experts) for r in range(8)]


@dataclass
class DeepseekV4Paths:
    """Locations for Hybrid DeepSeek-V4 loading."""

    hf_ckpt: Path
    inference_dir: Path
    converted_ckpt: Optional[Path] = None

    @property
    def config_path(self) -> Path:
        return self.hf_ckpt / "config.json"

    def load_config(self) -> Dict[str, Any]:
        """Prefer official inference/config.json (ModelArgs shape) when present."""
        infer_cfg = self.inference_dir / "config.json"
        path = infer_cfg if infer_cfg.is_file() else self.config_path
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_hf_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)


def resolve_v4_paths(
    hf_ckpt: Optional[str] = None,
    converted_ckpt: Optional[str] = None,
) -> DeepseekV4Paths:
    """Resolve HF + official inference/ paths from env or arguments."""
    hf = Path(
        hf_ckpt
        or os.environ.get("SGLANG_LITE_DSV4_HF", "")
        or os.environ.get("HF_CKPT", "")
        or os.path.expanduser("~/models/ds-v4-flash")
    ).expanduser()
    infer = Path(
        os.environ.get("SGLANG_LITE_DSV4_INFER", str(hf / "inference"))
    ).expanduser()
    conv_env = converted_ckpt or os.environ.get("SGLANG_LITE_DSV4_CONVERTED", "")
    conv = Path(conv_env).expanduser() if conv_env else None
    return DeepseekV4Paths(hf_ckpt=hf, inference_dir=infer, converted_ckpt=conv)


def import_official_inference(inference_dir: Path):
    """Import official ``model`` module from DeepSeek inference/."""
    import sys

    inference_dir = inference_dir.resolve()
    if not inference_dir.is_dir():
        raise FileNotFoundError(f"inference dir not found: {inference_dir}")
    key = str(inference_dir)
    if key not in sys.path:
        sys.path.insert(0, key)
    import importlib

    return importlib.import_module("model")


# HF config.json → official ModelArgs field names (partial map).
_HF_TO_ARGS = {
    "hidden_size": "dim",
    "num_hidden_layers": "n_layers",
    "num_attention_heads": "n_heads",
    "num_hash_layers": "n_hash_layers",
    "num_nextn_predict_layers": "n_mtp_layers",
    "moe_intermediate_size": "moe_inter_dim",
    "n_routed_experts": "n_routed_experts",
    "n_shared_experts": "n_shared_experts",
    "num_experts_per_tok": "n_activated_experts",
    "scoring_func": "score_func",
    "routed_scaling_factor": "route_scale",
    "qk_rope_head_dim": "rope_head_dim",
    "rms_norm_eps": "norm_eps",
    "sliding_window": "window_size",
    "vocab_size": "vocab_size",
    "q_lora_rank": "q_lora_rank",
    "o_lora_rank": "o_lora_rank",
    "o_groups": "o_groups",
    "head_dim": "head_dim",
    "index_n_heads": "index_n_heads",
    "index_head_dim": "index_head_dim",
    "index_topk": "index_topk",
    "hc_mult": "hc_mult",
    "hc_sinkhorn_iters": "hc_sinkhorn_iters",
    "hc_eps": "hc_eps",
    "swiglu_limit": "swiglu_limit",
    "compress_ratios": "compress_ratios",
    "compress_rope_theta": "compress_rope_theta",
    "rope_theta": "rope_theta",
    "expert_dtype": "expert_dtype",
}


def model_args_from_config(ModelArgs: Any, cfg: Dict[str, Any]) -> Any:
    """Build official ModelArgs from inference/ or HF config.json."""
    field_names = {f.name for f in fields(ModelArgs)}
    # Already ModelArgs-shaped (inference/config.json)
    if "dim" in cfg and "n_layers" in cfg:
        kwargs = {k: v for k, v in cfg.items() if k in field_names}
        if "compress_ratios" in kwargs and isinstance(kwargs["compress_ratios"], list):
            kwargs["compress_ratios"] = tuple(kwargs["compress_ratios"])
        return ModelArgs(**kwargs)

    kwargs: Dict[str, Any] = {}
    for hf_key, args_key in _HF_TO_ARGS.items():
        if hf_key in cfg and args_key in field_names:
            val = cfg[hf_key]
            if args_key == "compress_ratios" and isinstance(val, list):
                val = tuple(val)
            if args_key == "expert_dtype" and val == "fp4":
                val = "fp4"
            kwargs[args_key] = val
    rope = cfg.get("rope_scaling") or {}
    if isinstance(rope, dict):
        if "factor" in rope and "rope_factor" in field_names:
            kwargs["rope_factor"] = rope["factor"]
        if "beta_fast" in rope and "beta_fast" in field_names:
            kwargs["beta_fast"] = rope["beta_fast"]
        if "beta_slow" in rope and "beta_slow" in field_names:
            kwargs["beta_slow"] = rope["beta_slow"]
        if (
            "original_max_position_embeddings" in rope
            and "original_seq_len" in field_names
        ):
            kwargs["original_seq_len"] = rope["original_max_position_embeddings"]
    qcfg = cfg.get("quantization_config") or {}
    if qcfg.get("scale_fmt") and "scale_fmt" in field_names:
        kwargs["scale_fmt"] = qcfg["scale_fmt"]
    if "dtype" in field_names:
        kwargs.setdefault("dtype", "fp8")
    return ModelArgs(**kwargs)


# Back-compat alias
model_args_from_hf_config = model_args_from_config


def find_rank_shard(converted_ckpt: Path, rank: int, world_size: int) -> Path:
    """Locate convert.py output shard for a TP rank."""
    candidates = [
        converted_ckpt / f"model{rank}-mp{world_size}.safetensors",
        converted_ckpt / f"model{rank}.safetensors",
        converted_ckpt / f"mp{rank}.safetensors",
    ]
    for p in candidates:
        if p.is_file():
            return p
    rank_dir = converted_ckpt / f"rank{rank}"
    if rank_dir.is_dir():
        return rank_dir
    raise FileNotFoundError(
        f"no shard for rank={rank} under {converted_ckpt}; tried "
        f"{[str(p) for p in candidates]}"
    )


def ensure_tp_process_group() -> Tuple[int, int, str]:
    """Init NCCL (if needed) before constructing official Transformer.

    Official ``model.py`` reads ``dist.get_world_size()`` during ``Transformer.__init__``.
    Must run before model construction. Returns ``(rank, world_size, local_device)``.

    When the entrypoint remaps ``CUDA_VISIBLE_DEVICES`` to a single GPU (required by
    TileLang sparse_attn ABI expecting device_id 0), this uses ``cuda:0``.
    """
    import torch
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    remapped = bool(visible) and "," not in visible
    # After single-GPU remap, the only visible device is always cuda:0.
    device_index = 0 if remapped else local_rank
    local_device = f"cuda:{device_index}"

    if world_size > 1:
        if not dist.is_initialized():
            if not torch.cuda.is_available():
                raise RuntimeError("deepseek_v4 TP requires CUDA + NCCL")
            torch.cuda.set_device(device_index)
            dist.init_process_group("nccl")
        else:
            torch.cuda.set_device(device_index)
    elif torch.cuda.is_available():
        torch.cuda.set_device(device_index)

    if torch.cuda.is_available():
        torch.set_default_device("cuda")

    return rank, world_size, local_device


def load_v4_hybrid_model(
    paths: Optional[DeepseekV4Paths] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    device: Optional[str] = None,
    max_batch_size: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any], TpShardPlan]:
    """Load official Transformer for the given TP rank (Hybrid entry).

    Requires converted checkpoint at ``paths.converted_ckpt`` (from convert.py).
    Initializes the process group before ``Transformer(...)``.
    Returns (model, config_dict, shard_plan).
    """
    import torch
    from safetensors.torch import load_model

    env_rank, env_world, local_device = ensure_tp_process_group()
    rank = env_rank if rank is None else int(rank)
    world_size = env_world if world_size is None else int(world_size)
    if rank != env_rank or world_size != env_world:
        raise ValueError(
            f"rank/world_size mismatch: args=({rank},{world_size}) "
            f"env=({env_rank},{env_world})"
        )

    paths = paths or resolve_v4_paths()
    cfg = paths.load_config()
    n_experts = int(cfg.get("n_routed_experts", 256))
    plan = build_tp_shard_plan(world_size, rank, n_experts=n_experts)
    # Single-node torchrun: device follows LOCAL_RANK, not necessarily RANK.
    dev = device or local_device
    plan = TpShardPlan(
        world_size=plan.world_size,
        rank=plan.rank,
        local_device=dev,
        n_experts=plan.n_experts,
        n_local_experts=plan.n_local_experts,
        expert_start=plan.expert_start,
        expert_end=plan.expert_end,
    )

    model_mod = import_official_inference(paths.inference_dir)
    args = model_args_from_config(model_mod.ModelArgs, cfg)
    # Prefer HF n_experts when inference config omitted it
    if "n_routed_experts" not in cfg:
        hf_cfg = paths.load_hf_config()
        n_experts = int(hf_cfg.get("n_routed_experts", n_experts))
        plan = build_tp_shard_plan(world_size, rank, n_experts=n_experts)
        plan = TpShardPlan(
            world_size=plan.world_size,
            rank=plan.rank,
            local_device=dev,
            n_experts=plan.n_experts,
            n_local_experts=plan.n_local_experts,
            expert_start=plan.expert_start,
            expert_end=plan.expert_end,
        )
    if max_batch_size is not None and hasattr(args, "max_batch_size"):
        args.max_batch_size = int(max_batch_size)

    # Match official generate.py dtype defaults before allocating module buffers.
    if torch.cuda.is_available():
        torch.set_default_dtype(torch.bfloat16)
        # Prefer current CUDA device ("cuda") so TileLang sees device_id 0 after remap.
        torch_dev = torch.device("cuda")
    else:
        torch_dev = torch.device(dev)
    with torch_dev:
        model = model_mod.Transformer(args)

    ckpt = paths.converted_ckpt
    if ckpt is None:
        raise FileNotFoundError(
            "converted checkpoint required (run scripts/v4_official_smoke.sh). "
            "Set SGLANG_LITE_DSV4_CONVERTED=/path/to/mp-shards"
        )
    shard = find_rank_shard(Path(ckpt), rank, world_size)
    load_model(model, str(shard), strict=False)
    model.eval()
    return model, cfg, plan
