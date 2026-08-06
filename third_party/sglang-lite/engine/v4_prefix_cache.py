"""Prefix cache for DeepSeek-V4 Hybrid (official Attention in-module KV).

Stores CPU snapshots of Attention / Compressor / Indexer buffers keyed by
prompt token prefixes. On hit, restore into the sequence's batch slot and
skip (exact) or shorten (suffix) prefill. Divergent concurrent prefixes share
the same official buffer layout — prefer sequential / shared-prefix CB.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class V4PrefixEntry:
    token_ids: List[int]
    last_logits: Optional[torch.Tensor]  # CPU float
    # name -> tensor snapshot (CPU)
    buffers: Dict[str, torch.Tensor] = field(default_factory=dict)


def snapshot_v4_kv(model: torch.nn.Module, *, batch_slot: int = 0) -> Dict[str, torch.Tensor]:
    """Copy V4 stateful buffers for one batch slot to CPU."""
    out: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, mod in model.named_modules():
        for attr in ("kv_cache", "kv_state", "score_state"):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}"
            if key in seen:
                continue
            buf = getattr(mod, attr)
            if not torch.is_tensor(buf) or buf.numel() == 0:
                continue
            seen.add(key)
            # [max_batch, ...] — keep only the used slot to shrink CPU RAM.
            # clone() is required: detach().to("cpu") may still share storage.
            if buf.dim() >= 1 and buf.shape[0] > batch_slot:
                piece = buf[batch_slot : batch_slot + 1].detach().to("cpu").contiguous().clone()
            else:
                piece = buf.detach().to("cpu").contiguous().clone()
            out[key] = piece
    return out


def clear_v4_kv_slot(model: torch.nn.Module, *, batch_slot: int = 0) -> int:
    """Zero one batch row of official V4 KV / compressor state.

    Call on cache miss (before prefill) and when a sequence finishes so the
    next occupant of that slot cannot read stale compressor state.
    """
    n = 0
    seen = set()
    for name, mod in model.named_modules():
        for attr in ("kv_cache", "kv_state", "score_state"):
            if not hasattr(mod, attr):
                continue
            key = f"{name}.{attr}"
            if key in seen:
                continue
            buf = getattr(mod, attr)
            if not torch.is_tensor(buf) or buf.numel() == 0:
                continue
            seen.add(key)
            if buf.dim() >= 1 and buf.shape[0] > batch_slot:
                buf[batch_slot].zero_()
                n += 1
            elif batch_slot == 0:
                buf.zero_()
                n += 1
    return n


def restore_v4_kv(
    model: torch.nn.Module,
    buffers: Dict[str, torch.Tensor],
    *,
    batch_slot: int = 0,
) -> int:
    """Restore CPU snapshots into ``batch_slot``. Returns number of tensors written."""
    n = 0
    for key, cpu_t in buffers.items():
        # key = "layers.3.attn.kv_cache"
        if "." not in key:
            continue
        mod_path, attr = key.rsplit(".", 1)
        mod = model
        try:
            for part in mod_path.split("."):
                if not part:
                    continue
                mod = getattr(mod, part)
        except AttributeError:
            continue
        if not hasattr(mod, attr):
            continue
        buf = getattr(mod, attr)
        if not torch.is_tensor(buf):
            continue
        src = cpu_t.to(device=buf.device, dtype=buf.dtype)
        if buf.dim() >= 1 and buf.shape[0] > batch_slot and src.shape[0] == 1:
            buf[batch_slot : batch_slot + 1].copy_(src)
        elif buf.shape == src.shape:
            buf.copy_(src)
        else:
            # Shape drift — skip rather than corrupt.
            continue
        n += 1
    return n


class V4PrefixCache:
    """Longest-prefix store for Hybrid V4 prompts (exact + partial)."""

    def __init__(self, max_entries: int = 64):
        self.max_entries = max_entries
        self._entries: List[V4PrefixEntry] = []

    def clear(self) -> None:
        self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    def insert(
        self,
        token_ids: List[int],
        *,
        last_logits: Optional[torch.Tensor],
        buffers: Dict[str, torch.Tensor],
    ) -> None:
        if not token_ids or not buffers:
            return
        # Replace equal-length exact key if present.
        for i, e in enumerate(self._entries):
            if e.token_ids == token_ids:
                self._entries[i] = V4PrefixEntry(
                    token_ids=list(token_ids),
                    last_logits=last_logits.detach().float().cpu().clone()
                    if last_logits is not None
                    else None,
                    buffers=buffers,
                )
                return
        self._entries.append(
            V4PrefixEntry(
                token_ids=list(token_ids),
                last_logits=last_logits.detach().float().cpu().clone()
                if last_logits is not None
                else None,
                buffers=buffers,
            )
        )
        if len(self._entries) > self.max_entries:
            self._entries.pop(0)

    def match(self, token_ids: List[int]) -> Tuple[int, Optional[V4PrefixEntry]]:
        """Longest entry whose tokens are an exact prefix of ``token_ids``.

        Snapshots are taken after fully processing ``entry.token_ids``, so a
        partial overlap with a longer stored prompt is not reusable.
        """
        best_len = 0
        best: Optional[V4PrefixEntry] = None
        for e in self._entries:
            elen = len(e.token_ids)
            if elen == 0 or elen > len(token_ids):
                continue
            if token_ids[:elen] == e.token_ids and elen > best_len:
                best_len = elen
                best = e
        return best_len, best
