"""Generation sampling helpers (temperature / top_p / top_k / seed)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def sample_logits(
    logits: torch.Tensor,
    *,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Sample a token id from last-step logits.

    temperature <= 0 or ~0 → greedy (argmax).
    """
    if logits.dim() > 1:
        logits = logits.view(-1)

    if temperature is None or temperature <= 1e-5:
        return int(torch.argmax(logits).item())

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.numel())
        values, indices = torch.topk(logits, k)
        mask = torch.full_like(logits, float("-inf"))
        mask.scatter_(0, indices, values)
        logits = mask

    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        # Keep tokens with cumulative prob <= top_p, plus the first above threshold
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    if generator is not None:
        idx = torch.multinomial(probs, num_samples=1, generator=generator)
    else:
        idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def make_generator(device: str, seed: Optional[int]) -> Optional[torch.Generator]:
    if seed is None:
        return None
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    # multinomial on CUDA still accepts CPU generator in recent torch; keep CPU gen
    return gen
