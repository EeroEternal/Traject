"""Continuous batching scheduler with token-budget batch formation."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from .kv_cache import PastKV, RadixCache


@dataclass
class Sequence:
    seq_id: int
    request_id: str
    input_ids: List[int]
    output_ids: List[int] = field(default_factory=list)
    cached_len: int = 0
    cache_hit_tokens: int = 0
    block_table: List[int] = field(default_factory=list)
    kv_state: Optional[PastKV] = None
    last_logits: Optional[object] = None  # torch.Tensor; logits after last prompt token
    created_ts: float = field(default_factory=time.time)
    last_token_ts: float = field(default_factory=time.time)
    finished: bool = False
    finish_reason: str = ""
    prefill_tokens: int = 0
    decode_tokens: int = 0
    cancelled: bool = False
    # Generation params
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop_token_ids: List[int] = field(default_factory=list)
    stop_strings: List[str] = field(default_factory=list)


class SequenceTable:
    def __init__(self):
        self.sequences: Dict[int, Sequence] = {}
        self._next_id = 1

    def create(self, request_id: str, input_ids: List[int]) -> Sequence:
        seq = Sequence(seq_id=self._next_id, request_id=request_id, input_ids=list(input_ids))
        self._next_id += 1
        self.sequences[seq.seq_id] = seq
        return seq

    def get(self, seq_id: int) -> Optional[Sequence]:
        return self.sequences.get(seq_id)

    def remove(self, seq_id: int):
        self.sequences.pop(seq_id, None)


class Scheduler:
    def __init__(
        self,
        radix_cache: RadixCache,
        max_batch_size: int = 8,
        max_tokens_per_batch: int = 512,
        max_waiting: int = 128,
        max_prompt_tokens: int = 8192,
    ):
        self.radix = radix_cache
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_waiting = max_waiting
        self.max_prompt_tokens = max_prompt_tokens

        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.batch_former = BatchFormer(max_batch_size, max_tokens_per_batch)
        self._next_seq_id = 1
        self._by_request: Dict[str, Sequence] = {}
        self.last_batch_trace: List[Dict] = []

    def add_request(
        self,
        request_id: str,
        input_ids: List[int],
        *,
        max_tokens: int = 128,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        seed: Optional[int] = None,
        stop_token_ids: Optional[List[int]] = None,
        stop_strings: Optional[List[str]] = None,
    ) -> Sequence:
        if len(self.waiting) >= self.max_waiting:
            raise RuntimeError(f"waiting queue full (max_waiting={self.max_waiting})")
        if len(input_ids) > self.max_prompt_tokens:
            raise ValueError(
                f"prompt length {len(input_ids)} exceeds max_prompt_tokens={self.max_prompt_tokens}"
            )

        seq = Sequence(
            seq_id=self._next_seq_id,
            request_id=request_id,
            input_ids=list(input_ids),
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            stop_token_ids=list(stop_token_ids or []),
            stop_strings=list(stop_strings or []),
        )
        self._next_seq_id += 1

        matched, matched_len, _remaining, block_ids, past_kv, last_logits = self.radix.match_prefix(
            input_ids
        )
        seq.cached_len = matched_len
        # Hit counts when paged prefix blocks are reusable (rebuild past from pages).
        seq.cache_hit_tokens = matched_len if block_ids else 0
        if block_ids:
            seq.block_table = list(block_ids)
        # Prefer paged rebuild; keep forked HF past only as fallback
        if past_kv is not None:
            seq.kv_state = past_kv
        if last_logits is not None:
            seq.last_logits = last_logits

        self.waiting.append(seq)
        self._by_request[request_id] = seq
        return seq

    def cancel(self, request_id: str) -> bool:
        seq = self._by_request.get(request_id)
        if seq is None:
            return False
        seq.cancelled = True
        self.mark_finished(seq, "cancelled")
        return True

    def step(self) -> Tuple[List[Sequence], List[bool]]:
        batch, is_prefill = self.batch_former.form_batch(self.waiting, self.running)
        self.last_batch_trace = [
            {
                "request_id": s.request_id,
                "seq_id": s.seq_id,
                "prefill": p,
                "cached_len": s.cached_len,
                "prompt_len": len(s.input_ids),
                "output_len": len(s.output_ids),
            }
            for s, p in zip(batch, is_prefill)
        ]
        # Merge into running
        running_ids: Set[int] = {s.seq_id for s in self.running}
        for s in batch:
            if s.seq_id not in running_ids and not s.finished:
                self.running.append(s)
        self.running = [s for s in self.running if not s.finished]
        return batch, is_prefill

    def update_after_prefill(
        self,
        seq: Sequence,
        new_tokens: List[int],
        new_kv_state: PastKV,
        new_block_ids: Optional[List[int]] = None,
    ) -> None:
        seq.cached_len = len(seq.input_ids)
        seq.kv_state = new_kv_state
        if new_block_ids:
            seq.block_table.extend(new_block_ids)
        seq.prefill_tokens += len(new_tokens)
        full_prompt = seq.input_ids[: seq.cached_len]
        self.radix.insert_or_update(
            full_prompt,
            new_kv_state,
            seq.cached_len,
            block_ids=list(seq.block_table),
            last_logits=getattr(seq, "last_logits", None),
        )

    def update_after_decode(
        self,
        seq: Sequence,
        new_token: int,
        new_kv_state: PastKV,
        new_block_ids: Optional[List[int]] = None,
    ) -> None:
        seq.output_ids.append(new_token)
        # cached_len tracks KV length and is updated by the runner after each forward
        seq.kv_state = new_kv_state
        if new_block_ids:
            seq.block_table.extend(new_block_ids)
        seq.decode_tokens += 1
        seq.last_token_ts = time.time()

    def mark_finished(self, seq: Sequence, reason: str = "stop") -> None:
        if seq.finished and seq.finish_reason:
            # Still ensure cleanup once
            pass
        seq.finished = True
        seq.finish_reason = reason or seq.finish_reason or "stop"
        if seq in self.running:
            self.running.remove(seq)
        # Remove from waiting if still there
        self.waiting = deque(s for s in self.waiting if s.seq_id != seq.seq_id)
        # Release private pages (shared prefix pages keep refcount)
        if seq.block_table:
            self.radix.release_blocks(seq.block_table)
            seq.block_table = []
        seq.kv_state = None
        self._by_request.pop(seq.request_id, None)


class BatchFormer:
    def __init__(self, max_batch_size: int = 8, max_tokens_per_batch: int = 512):
        self.max_batch_size = max_batch_size
        self.max_tokens_per_batch = max_tokens_per_batch

    def form_batch(
        self, waiting: deque, running: List[Sequence]
    ) -> Tuple[List[Sequence], List[bool]]:
        batch: List[Sequence] = []
        is_prefill: List[bool] = []
        token_budget = self.max_tokens_per_batch

        for seq in list(running):
            if len(batch) >= self.max_batch_size or token_budget <= 0:
                break
            if seq.finished or seq.cancelled:
                continue
            # decode costs 1 token of budget
            batch.append(seq)
            is_prefill.append(False)
            token_budget -= 1

        while waiting and len(batch) < self.max_batch_size and token_budget > 0:
            seq = waiting[0]
            if seq.finished or seq.cancelled:
                waiting.popleft()
                continue
            remaining_prompt = max(0, len(seq.input_ids) - seq.cached_len)
            need_prefill = remaining_prompt > 0
            cost = max(1, remaining_prompt) if need_prefill else 1
            if cost > token_budget and batch:
                # leave for later step (fairness: don't starve decode)
                break
            waiting.popleft()
            batch.append(seq)
            is_prefill.append(need_prefill)
            token_budget -= min(cost, token_budget)

        return batch, is_prefill
