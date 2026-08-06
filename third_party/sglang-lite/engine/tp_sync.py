"""Minimal TP coordination for the engine process (torchrun).

Rank 0 owns the HTTP surface; all ranks must submit identical work and
``pump_until_idle`` together so Hybrid V4 NCCL forwards stay aligned.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_tp() -> bool:
    return world_size() > 1


def remap_visible_device_for_tilelang() -> None:
    """TileLang sparse_attn expects process-local device_id==0."""
    if "LOCAL_RANK" in os.environ and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]


def broadcast_obj(msg: Optional[Dict[str, Any]], *, src: int = 0) -> Optional[Dict[str, Any]]:
    """Broadcast a small JSON-able dict (or None to shut down workers)."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return msg
    payload = [msg]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
