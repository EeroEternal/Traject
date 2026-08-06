"""Central continuous-batching engine loop.

HTTP handlers only submit work and consume deltas; this loop owns scheduling.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from .kv_cache import RadixCache
from .runner import ModelRunner
from .scheduler import Scheduler, Sequence


@dataclass
class GenParams:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    timeout_s: float = 300.0
    ignore_eos: bool = False


@dataclass
class SubmitResult:
    request_id: str
    delta_queue: "queue.Queue[Dict[str, Any]]"
    seq: Sequence


@dataclass
class _Pending:
    request_id: str
    input_ids: List[int]
    params: GenParams
    delta_queue: "queue.Queue[Dict[str, Any]]"
    enqueued_at: float = field(default_factory=time.time)


class EngineLoop:
    """Long-running loop composing RadixCache + Scheduler + ModelRunner."""

    def __init__(
        self,
        runner: ModelRunner,
        radix: Optional[RadixCache] = None,
        max_batch_size: int = 8,
        max_tokens_per_batch: int = 512,
        max_waiting: int = 128,
        max_prompt_tokens: int = 8192,
        idle_sleep_s: float = 0.001,
    ):
        self.runner = runner
        if radix is None:
            # Modest page pool for the HF-cache prototype (paged tensors are mirrors only).
            # Full 64k prealloc is too large for tiny CPU fixtures / single-GPU demos.
            max_tokens = 4096 if runner.device == "cpu" else 16384
            radix = RadixCache(
                max_tokens=max_tokens,
                block_size=16,
                num_layers=runner.num_layers,
                num_kv_heads=runner.num_kv_heads,
                head_dim=runner.head_dim,
                dtype=getattr(runner, "torch_dtype", None)
                or (torch.float32 if runner.device == "cpu" else torch.bfloat16),
                device=runner.device if runner.device != "cpu" else "cpu",
            )
        self.radix = radix
        self.scheduler = Scheduler(
            self.radix,
            max_batch_size=max_batch_size,
            max_tokens_per_batch=max_tokens_per_batch,
            max_waiting=max_waiting,
            max_prompt_tokens=max_prompt_tokens,
        )
        self.idle_sleep_s = idle_sleep_s

        self._submit_q: queue.Queue[Optional[_Pending]] = queue.Queue()
        self._delta_qs: Dict[str, queue.Queue] = {}
        self._prev_text: Dict[str, str] = {}
        self._deadlines: Dict[str, float] = {}
        self._cancelled: set = set()
        self._lock = threading.Lock()
        self._ready = False
        self._stopping = False
        self._thread: Optional[threading.Thread] = None
        self.steps = 0
        self.multi_request_batches = 0

    @property
    def ready(self) -> bool:
        return self._ready

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stopping = False
        self._thread = threading.Thread(target=self._run, name="sglang-lite-engine-loop", daemon=True)
        self._thread.start()
        self._ready = True

    def mark_ready(self) -> None:
        """Mark ready without a background thread (TP sync / pump_until_idle)."""
        self._ready = True

    def stop(self, drain: bool = True) -> None:
        self._stopping = True
        self._submit_q.put(None)
        if self._thread:
            self._thread.join(timeout=30.0)
        self._ready = False

    def submit(self, request_id: str, input_ids: List[int], params: Optional[GenParams] = None) -> SubmitResult:
        if self._stopping:
            raise RuntimeError("engine is draining; not accepting new requests")
        params = params or GenParams()
        if params.max_tokens <= 0:
            raise ValueError("max_tokens must be >= 1")
        if params.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not (0.0 < params.top_p <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        if params.top_k is not None and params.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if not input_ids:
            raise ValueError("input_ids must be non-empty")
        dq: queue.Queue = queue.Queue(maxsize=256)
        pending = _Pending(
            request_id=request_id,
            input_ids=input_ids,
            params=params,
            delta_queue=dq,
        )
        with self._lock:
            self._delta_qs[request_id] = dq
            self._prev_text[request_id] = ""
            self._deadlines[request_id] = time.time() + params.timeout_s
        self._submit_q.put(pending)
        # seq filled after admission; return placeholder — caller uses delta_queue
        return SubmitResult(request_id=request_id, delta_queue=dq, seq=None)  # type: ignore[arg-type]

    def cancel(self, request_id: str) -> bool:
        """Cancel a request whether it is pending, waiting, or running."""
        with self._lock:
            self._cancelled.add(request_id)
            had_delta = request_id in self._delta_qs
        ok = self.scheduler.cancel(request_id)
        # Always notify client if we still own a delta queue
        if had_delta or ok:
            self._emit(
                request_id,
                {
                    "text": "",
                    "finish_reason": "cancelled",
                    "usage": None,
                    "error": None,
                },
                final=True,
            )
        return True

    def _emit(self, request_id: str, payload: Dict[str, Any], final: bool = False) -> None:
        dq = self._delta_qs.get(request_id)
        if dq is None:
            return
        try:
            dq.put(payload, timeout=1.0)
        except queue.Full:
            # backpressure: drop slow client by cancelling
            self.scheduler.cancel(request_id)
        if final:
            with self._lock:
                self._delta_qs.pop(request_id, None)
            # V4: free official Attention batch slot (prefix snapshot already on CPU).
            if getattr(self.runner, "_v4_hybrid", False):
                for s in list(self.scheduler.running) + list(self.scheduler.waiting):
                    if s.request_id == request_id:
                        self.runner.v4_release_seq(s, batch_slot=0)
                        break
                self._prev_text.pop(request_id, None)
                self._deadlines.pop(request_id, None)

    def _admit_pending(self) -> None:
        while True:
            try:
                item = self._submit_q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._stopping = True
                break
            with self._lock:
                if item.request_id in self._cancelled:
                    self._cancelled.discard(item.request_id)
                    continue
            try:
                seq = self.scheduler.add_request(
                    item.request_id,
                    item.input_ids,
                    max_tokens=item.params.max_tokens,
                    temperature=item.params.temperature,
                    top_p=item.params.top_p,
                    top_k=item.params.top_k,
                    seed=item.params.seed,
                    stop_strings=item.params.stop,
                )
                # V4 Hybrid KV lives in official Attention buffers — ignore
                # Radix hits and match against V4PrefixCache snapshots instead.
                if getattr(self.runner, "_v4_hybrid", False):
                    seq.block_table = []
                    seq.kv_state = None
                    match_len, entry = self.runner.v4_match_prefix(seq.input_ids)
                    if match_len > 0 and entry is not None:
                        seq.cached_len = match_len
                        seq.cache_hit_tokens = match_len
                        seq.last_logits = entry.last_logits
                        seq._v4_prefix_entry = entry
                        seq._v4_kv_pending_restore = True
                    else:
                        seq.cached_len = 0
                        seq.cache_hit_tokens = 0
                        seq.last_logits = None
                        seq._v4_prefix_entry = None
                        seq._v4_kv_pending_restore = False
                eos = self.runner.eos_token_id
                if eos is not None and not item.params.ignore_eos:
                    seq.stop_token_ids = [eos]
            except MemoryError as e:
                self._emit(
                    item.request_id,
                    {
                        "text": "",
                        "finish_reason": "error",
                        "usage": None,
                        "error": f"oom: {e}",
                    },
                    final=True,
                )
            except Exception as e:
                self._emit(
                    item.request_id,
                    {
                        "text": "",
                        "finish_reason": "error",
                        "usage": None,
                        "error": str(e),
                    },
                    final=True,
                )

    def _check_timeouts(self) -> None:
        now = time.time()
        expired = [rid for rid, dl in list(self._deadlines.items()) if now > dl]
        for rid in expired:
            self.scheduler.cancel(rid)
            self._emit(
                rid,
                {"text": "", "finish_reason": "timeout", "usage": None, "error": None},
                final=True,
            )

    def _usage(self, seq: Sequence) -> Dict[str, int]:
        prompt = len(seq.input_ids)
        completion = len(seq.output_ids)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
            "cache_hit_tokens": int(getattr(seq, "cache_hit_tokens", 0) or 0),
        }

    def _apply_stop_and_limits(
        self, seq: Sequence, tok: int, prev_text: str
    ) -> tuple[bool, str, Optional[int]]:
        """After appending tok: return (finished, delta_text, emit_token).

        Stop strings / EOS are trimmed so they are not leaked to the client.
        """
        if seq.cancelled:
            return True, "", None
        if seq.max_tokens <= 0 or len(seq.output_ids) > seq.max_tokens:
            # overshoot: drop the last token if we exceeded max_tokens
            if seq.max_tokens <= 0:
                seq.output_ids.clear()
                self.scheduler.mark_finished(seq, "length")
                return True, "", None
            if len(seq.output_ids) > seq.max_tokens:
                seq.output_ids.pop()
                self.scheduler.mark_finished(seq, "length")
                delta = self.runner.detokenize_delta(seq.output_ids, prev_text)
                return True, delta, tok if delta else None
        if len(seq.output_ids) >= seq.max_tokens:
            self.scheduler.mark_finished(seq, "length")
            delta = self.runner.detokenize_delta(seq.output_ids, prev_text)
            return True, delta, tok if delta else None

        if tok in (seq.stop_token_ids or []):
            if seq.output_ids and seq.output_ids[-1] == tok:
                seq.output_ids.pop()
            self.scheduler.mark_finished(seq, "stop")
            delta = self.runner.detokenize_delta(seq.output_ids, prev_text)
            return True, delta, None

        full = self.runner.detokenize(seq.output_ids)
        if seq.stop_strings:
            for s in seq.stop_strings:
                if s and s in full:
                    trimmed = full[: full.find(s)]
                    while seq.output_ids and len(self.runner.detokenize(seq.output_ids)) > len(
                        trimmed
                    ):
                        seq.output_ids.pop()
                    self.scheduler.mark_finished(seq, "stop")
                    delta = self.runner.detokenize_delta(seq.output_ids, prev_text)
                    return True, delta, tok if delta else None

        delta = self.runner.detokenize_delta(seq.output_ids, prev_text)
        return False, delta, tok

    def _run(self) -> None:
        while not self._stopping or self.scheduler.waiting or self.scheduler.running:
            self._admit_pending()
            self._check_timeouts()
            batch, is_prefill = self.scheduler.step()
            if not batch:
                if self._stopping:
                    break
                time.sleep(self.idle_sleep_s)
                continue

            self.steps += 1
            if len(batch) > 1:
                self.multi_request_batches += 1

            try:
                next_tokens = self.runner.run_batch(batch, self.radix, is_prefill)
            except Exception as e:
                for seq in batch:
                    self.scheduler.mark_finished(seq, "error")
                    self._emit(
                        seq.request_id,
                        {
                            "text": "",
                            "finish_reason": "error",
                            "usage": self._usage(seq),
                            "error": str(e),
                        },
                        final=True,
                    )
                continue

            for seq, tok, pre in zip(batch, next_tokens, is_prefill):
                if seq.finished or tok is None:
                    continue
                with self._lock:
                    if seq.request_id in self._cancelled:
                        self.scheduler.mark_finished(seq, "cancelled")
                        self._emit(
                            seq.request_id,
                            {
                                "text": "",
                                "finish_reason": "cancelled",
                                "usage": self._usage(seq),
                                "error": None,
                            },
                            final=True,
                        )
                        continue
                if pre:
                    if getattr(self.runner, "_v4_hybrid", False):
                        # Bypass radix insert; official model owns KV.
                        seq.cached_len = len(seq.input_ids)
                        seq.kv_state = None
                    else:
                        self.scheduler.update_after_prefill(seq, [], seq.kv_state)
                self.scheduler.update_after_decode(seq, tok, seq.kv_state)

                prev = self._prev_text.get(seq.request_id, "")
                finished, delta_text, emit_tok = self._apply_stop_and_limits(seq, tok, prev)
                self._prev_text[seq.request_id] = self.runner.detokenize(seq.output_ids)

                payload = {
                    "text": delta_text,
                    "token": emit_tok,
                    "finish_reason": seq.finish_reason if finished else None,
                    "usage": self._usage(seq) if finished else None,
                    "error": None,
                }
                self._emit(seq.request_id, payload, final=finished)

            self.scheduler.running = [s for s in self.scheduler.running if not s.finished]

    def pump_once(self) -> bool:
        """Run one scheduler step synchronously. Returns True if work was done."""
        self._admit_pending()
        self._check_timeouts()
        batch, is_prefill = self.scheduler.step()
        if not batch:
            return False
        self.steps += 1
        if len(batch) > 1:
            self.multi_request_batches += 1
        try:
            next_tokens = self.runner.run_batch(batch, self.radix, is_prefill)
        except Exception as e:
            for seq in batch:
                self.scheduler.mark_finished(seq, "error")
                self._emit(
                    seq.request_id,
                    {
                        "text": "",
                        "finish_reason": "error",
                        "usage": self._usage(seq),
                        "error": str(e),
                    },
                    final=True,
                )
            return True

        for seq, tok, pre in zip(batch, next_tokens, is_prefill):
            if seq.finished or tok is None:
                continue
            with self._lock:
                if seq.request_id in self._cancelled:
                    self.scheduler.mark_finished(seq, "cancelled")
                    self._emit(
                        seq.request_id,
                        {
                            "text": "",
                            "finish_reason": "cancelled",
                            "usage": self._usage(seq),
                            "error": None,
                        },
                        final=True,
                    )
                    continue
            if pre:
                if getattr(self.runner, "_v4_hybrid", False):
                    seq.cached_len = len(seq.input_ids)
                    seq.kv_state = None
                else:
                    self.scheduler.update_after_prefill(seq, [], seq.kv_state)
            self.scheduler.update_after_decode(seq, tok, seq.kv_state)

            prev = self._prev_text.get(seq.request_id, "")
            finished, delta_text, emit_tok = self._apply_stop_and_limits(seq, tok, prev)
            self._prev_text[seq.request_id] = self.runner.detokenize(seq.output_ids)

            payload = {
                "text": delta_text,
                "token": emit_tok,
                "finish_reason": seq.finish_reason if finished else None,
                "usage": self._usage(seq) if finished else None,
                "error": None,
            }
            self._emit(seq.request_id, payload, final=finished)

        self.scheduler.running = [s for s in self.scheduler.running if not s.finished]
        return True

    def pump_until_idle(self, timeout_s: float = 600.0) -> None:
        """Synchronously drain pending/running work (TP-safe; no background thread)."""
        deadline = time.time() + timeout_s
        while True:
            self._admit_pending()
            busy = bool(self.scheduler.waiting or self.scheduler.running or not self._submit_q.empty())
            if not busy:
                return
            if time.time() > deadline:
                raise TimeoutError("pump_until_idle exceeded timeout")
            if not self.pump_once():
                time.sleep(self.idle_sleep_s)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ready": self._ready,
            "waiting": len(self.scheduler.waiting),
            "running": len(self.scheduler.running),
            "steps": self.steps,
            "multi_request_batches": self.multi_request_batches,
            "last_batch_trace": list(self.scheduler.last_batch_trace),
            "cache": self.radix.get_cache_stats(),
            "model": self.runner.model_name,
            "device": self.runner.device,
            "last_model_forward_size": getattr(self.runner, "last_model_forward_size", 0),
            "model_forward_count": getattr(self.runner, "model_forward_count", 0),
            "paged_rebuild_count": getattr(self.runner, "paged_rebuild_count", 0),
            "kernel_backend": getattr(
                getattr(self.runner, "kernel_backend", None), "name", "unknown"
            ),
            "arch_family": getattr(
                getattr(getattr(self.runner, "kernel_backend", None), "arch_family", None),
                "value",
                "unknown",
            ),
            "sparse_mla_backend": getattr(
                getattr(
                    getattr(self.runner, "kernel_backend", None), "sparse_mla_backend", None
                ),
                "value",
                "none",
            ),
            "moe_gemm_backend": getattr(
                getattr(
                    getattr(self.runner, "kernel_backend", None), "moe_gemm_backend", None
                ),
                "value",
                "none",
            ),
            "v4_hybrid": bool(getattr(self.runner, "_v4_hybrid", False)),
        }
