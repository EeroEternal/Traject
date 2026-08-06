"""Internal engine process HTTP server (GenerationRequest / TokenDelta).

Not an OpenAI surface — that lives in Rust control/serving.
Run: python -m sglang_lite.process --model <moe> --port 9001

TP / DeepSeek-V4 (torchrun)::

  torchrun --nproc-per-node=8 -m sglang_lite.process \\
    --model ~/models/ds-v4-flash --device cuda --port 9001
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional

# Must run before torch import (TileLang device_id==0).
from .tp_sync import broadcast_obj, is_tp, rank, remap_visible_device_for_tilelang, world_size

remap_visible_device_for_tilelang()

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field, field_validator

from .loop import EngineLoop, GenParams
from .models import list_verified_models, register_verified
from .runner import ModelRunner

logger = logging.getLogger("sglang_lite.process")

app = FastAPI(title="sglang-lite engine process")
LOOP: Optional[EngineLoop] = None
READY = False
MODEL_NAME = "stub"
# Serialize TP generate so all ranks stay on one shared schedule.
_TP_LOCK = threading.Lock()
_TP_MODE = False
# Rank-0: one CUDA-owning thread runs broadcast+submit+pump (TileLang is
# thread-affine; asyncio default executors break device binding).
_TP_CUDA_Q: "queue.Queue" = None  # type: ignore[assignment]
_TP_CUDA_THREAD: Optional[threading.Thread] = None
# Traject MemoryManager alignment: pin/ref + last prompt tokens per session.
_PREFIX_PINS: Dict[str, Dict[str, Any]] = {}
_SESSION_LAST_IDS: Dict[str, List[int]] = {}
# prefix_id → last prompt token ids (for free/evict of V4 snapshots).
_PREFIX_TOKEN_IDS: Dict[str, List[int]] = {}


class ChatMessage(BaseModel):
    role: str
    content: Optional[str] = None


class GenerationRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model: str = ""
    messages: Optional[List[Dict[str, Any]]] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    stream: bool = True
    timeout_s: float = 300.0
    # Traject session tracking (optional; ignored by core loop except logging / request_id).
    trajectory_id: Optional[str] = None
    session_id: Optional[str] = None
    prefix_id: Optional[str] = None
    step_id: Optional[str] = None

    @field_validator("max_tokens")
    @classmethod
    def _max_tokens(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_tokens must be >= 1")
        return v

    @field_validator("temperature")
    @classmethod
    def _temperature(cls, v: float) -> float:
        if v < 0:
            raise ValueError("temperature must be >= 0")
        return v

    @field_validator("top_p")
    @classmethod
    def _top_p(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError("top_p must be in (0, 1]")
        return v


class CancelRequest(BaseModel):
    request_id: str


class PrefixPinRequest(BaseModel):
    """Traject MemoryManager → engine pin for tool gaps / prefetch."""

    prefix_id: str
    session_id: Optional[str] = None
    trajectory_id: Optional[str] = None
    ttl_ms: int = 30_000
    reason: str = "WaitingTool"


class PrefixUnpinRequest(BaseModel):
    prefix_id: str


class PrefixFreeRequest(BaseModel):
    """Traject MemoryManager eviction: drop pin + any V4 snapshot for this handle."""

    prefix_id: str
    session_id: Optional[str] = None


def _input_ids_from_req(req: GenerationRequest) -> List[int]:
    assert LOOP is not None
    if req.input_ids:
        return list(req.input_ids)
    messages = req.messages or []
    return LOOP.runner.apply_chat_template(messages)


def _params_from_req(req: GenerationRequest) -> GenParams:
    return GenParams(
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        seed=req.seed,
        stop=req.stop,
        timeout_s=req.timeout_s,
    )


def _msg_generate(req: GenerationRequest, input_ids: List[int]) -> Dict[str, Any]:
    return {
        "op": "generate",
        "request_id": req.request_id,
        "input_ids": list(input_ids),
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
        "top_p": req.top_p,
        "top_k": req.top_k,
        "seed": req.seed,
        "stop": req.stop,
        "timeout_s": req.timeout_s,
    }


def _apply_generate_msg(msg: Dict[str, Any]):
    assert LOOP is not None
    params = GenParams(
        max_tokens=int(msg["max_tokens"]),
        temperature=float(msg["temperature"]),
        top_p=float(msg["top_p"]),
        top_k=msg.get("top_k"),
        seed=msg.get("seed"),
        stop=msg.get("stop"),
        timeout_s=float(msg.get("timeout_s", 300.0)),
    )
    return LOOP.submit(str(msg["request_id"]), list(msg["input_ids"]), params)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "service": "sglang-lite-engine"}


@app.get("/readyz")
async def readyz():
    if not READY or LOOP is None or not LOOP.ready:
        return JSONResponse({"status": "not_ready"}, status_code=503)
    return {
        "status": "ready",
        "model": MODEL_NAME,
        "tp_world_size": world_size(),
        "rank": rank(),
    }


@app.get("/metrics")
async def metrics():
    if LOOP is None:
        body = "# no engine\n"
    else:
        stats = LOOP.get_stats()
        lines = [
            "# HELP sglang_lite_up Engine process up",
            "# TYPE sglang_lite_up gauge",
            "sglang_lite_up 1",
            f"sglang_lite_ready {1 if READY else 0}",
            f"sglang_lite_waiting_requests {stats['waiting']}",
            f"sglang_lite_running_requests {stats['running']}",
            f"sglang_lite_engine_steps {stats['steps']}",
            f"sglang_lite_multi_request_batches {stats['multi_request_batches']}",
            f"sglang_lite_cache_hit_count {stats['cache'].get('hit_count', 0)}",
            f"sglang_lite_cache_miss_count {stats['cache'].get('miss_count', 0)}",
            f"sglang_lite_kv_blocks_used {stats['cache'].get('blocks_used', 0)}",
            f"sglang_lite_oom_reject_count {stats['cache'].get('oom_reject_count', 0)}",
            f"sglang_lite_tp_world_size {world_size()}",
        ]
        body = "\n".join(lines) + "\n"
    return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")


@app.get("/v1/models")
async def models():
    return {"object": "list", "data": [{"id": m, "object": "model"} for m in list_verified_models()]}


@app.get("/stats")
async def stats():
    if LOOP is None:
        return {}
    return LOOP.get_stats()


@app.post("/v1/cancel")
async def cancel(req: CancelRequest):
    if LOOP is None:
        return JSONResponse({"ok": False}, status_code=503)
    # Local cancel only — TP workers may finish the current pump step; do not
    # take _TP_LOCK (would deadlock with an in-flight generate).
    ok = LOOP.cancel(req.request_id)
    return {"ok": ok, "request_id": req.request_id, "tp": _TP_MODE}


@app.post("/v1/prefix/pin")
async def prefix_pin(req: PrefixPinRequest):
    """Mark a prefix handle as pinned (Traject tool-wait / prefetch)."""
    import time

    until = time.time() + max(0.001, req.ttl_ms / 1000.0)
    entry = _PREFIX_PINS.get(req.prefix_id) or {"refs": 0}
    entry["refs"] = int(entry.get("refs", 0)) + 1
    entry["until"] = max(float(entry.get("until", 0)), until)
    entry["reason"] = req.reason
    entry["session_id"] = req.session_id
    entry["trajectory_id"] = req.trajectory_id
    _PREFIX_PINS[req.prefix_id] = entry
    logger.info(
        "prefix pin id=%s refs=%s until=%.0f reason=%s session=%s",
        req.prefix_id,
        entry["refs"],
        entry["until"],
        req.reason,
        req.session_id,
    )
    return {"ok": True, "prefix_id": req.prefix_id, "refs": entry["refs"]}


@app.post("/v1/prefix/unpin")
async def prefix_unpin(req: PrefixUnpinRequest):
    entry = _PREFIX_PINS.get(req.prefix_id)
    if not entry:
        return {"ok": True, "prefix_id": req.prefix_id, "refs": 0}
    entry["refs"] = max(0, int(entry.get("refs", 1)) - 1)
    if entry["refs"] == 0:
        _PREFIX_PINS.pop(req.prefix_id, None)
    else:
        _PREFIX_PINS[req.prefix_id] = entry
    logger.info("prefix unpin id=%s refs=%s", req.prefix_id, entry["refs"])
    return {"ok": True, "prefix_id": req.prefix_id, "refs": entry["refs"]}


@app.get("/v1/prefix/stats")
async def prefix_stats():
    import time

    now = time.time()
    live = {
        k: v
        for k, v in _PREFIX_PINS.items()
        if float(v.get("until", 0)) > now or int(v.get("refs", 0)) > 0
    }
    return {
        "pinned": len(live),
        "sessions_tracked": len(_SESSION_LAST_IDS),
        "prefix_token_maps": len(_PREFIX_TOKEN_IDS),
        "pins": {
            k: {"refs": v.get("refs"), "reason": v.get("reason")} for k, v in list(live.items())[:32]
        },
    }


@app.post("/v1/prefix/free")
async def prefix_free(req: PrefixFreeRequest):
    """Physical + logical free for a Traject prefix handle.

    1. Drop pin + session token map
    2. Drop V4 CPU snapshots (and zero associated GPU slot when hybrid)
    3. Free radix paged GPU blocks for private leaves matching the token path
    """
    _PREFIX_PINS.pop(req.prefix_id, None)
    token_ids = _PREFIX_TOKEN_IDS.pop(req.prefix_id, None)
    if req.session_id:
        prev = _SESSION_LAST_IDS.get(req.session_id)
        if prev is not None and token_ids is not None and prev == token_ids:
            _SESSION_LAST_IDS.pop(req.session_id, None)

    v4_dropped = 0
    radix_stats: Dict[str, int] = {"nodes_unlinked": 0, "blocks_released": 0}
    gpu_slot_cleared = 0

    if token_ids and LOOP is not None:
        runner = getattr(LOOP, "runner", None)
        # V4 hybrid: drop CPU snapshot + zero live GPU batch slot (physical free).
        if runner is not None:
            cache = getattr(runner, "_v4_prefix_cache", None)
            if cache is not None and hasattr(cache, "drop_exact"):
                v4_dropped = int(cache.drop_exact(token_ids) or 0)
            elif cache is not None and hasattr(cache, "drop_prefix"):
                v4_dropped = int(cache.drop_prefix(token_ids) or 0)
            if getattr(runner, "_v4_hybrid", False) and getattr(runner, "model", None) is not None:
                try:
                    from .v4_prefix_cache import clear_v4_kv_slot

                    gpu_slot_cleared = int(clear_v4_kv_slot(runner.model, batch_slot=0) or 0)
                except Exception as e:  # noqa: BLE001
                    logger.debug("clear_v4_kv_slot failed: %s", e)

        # Non-V4 / radix path: free private GPU pages for this token path.
        radix = getattr(LOOP, "radix", None)
        if radix is None and runner is not None:
            radix = getattr(runner, "radix", None)
        if radix is not None and hasattr(radix, "free_prefix_tokens"):
            try:
                radix_stats = dict(radix.free_prefix_tokens(token_ids) or {})
            except Exception as e:  # noqa: BLE001
                logger.debug("radix free_prefix_tokens failed: %s", e)

    logger.info(
        "prefix free id=%s tokens=%s v4_dropped=%s radix=%s gpu_slot_cleared=%s session=%s",
        req.prefix_id,
        len(token_ids) if token_ids else 0,
        v4_dropped,
        radix_stats,
        gpu_slot_cleared,
        req.session_id,
    )
    return {
        "ok": True,
        "prefix_id": req.prefix_id,
        "token_len": len(token_ids) if token_ids else 0,
        "v4_dropped": v4_dropped,
        "radix_nodes_unlinked": int(radix_stats.get("nodes_unlinked", 0)),
        "radix_blocks_released": int(radix_stats.get("blocks_released", 0)),
        "gpu_slot_tensors_cleared": gpu_slot_cleared,
    }


def _common_prefix_len(a: List[int], b: List[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@app.post("/v1/generate")
async def generate(req: GenerationRequest, request: Request):
    if LOOP is None or not READY:
        return JSONResponse(
            {"error": "engine not ready"},
            status_code=503,
        )
    if req.trajectory_id or req.session_id or req.prefix_id:
        logger.info(
            "traject session trajectory_id=%s session_id=%s prefix_id=%s step_id=%s request_id=%s",
            req.trajectory_id,
            req.session_id,
            req.prefix_id,
            req.step_id,
            req.request_id,
        )
    if req.prefix_id and req.prefix_id in _PREFIX_PINS:
        logger.info("prefix_id %s is pinned refs=%s", req.prefix_id, _PREFIX_PINS[req.prefix_id].get("refs"))
    if req.model and req.model != MODEL_NAME and req.model not in list_verified_models():
        return JSONResponse(
            {
                "error": {
                    "message": f"model '{req.model}' is not loaded (loaded={MODEL_NAME})",
                    "type": "invalid_request_error",
                    "code": "model_not_found",
                }
            },
            status_code=400,
        )
    if not req.input_ids and not req.messages:
        return JSONResponse(
            {"error": {"message": "messages or input_ids required", "type": "invalid_request_error"}},
            status_code=400,
        )
    try:
        input_ids = _input_ids_from_req(req)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    if not input_ids:
        return JSONResponse(
            {"error": {"message": "empty prompt after tokenization", "type": "invalid_request_error"}},
            status_code=400,
        )

    # Session continuity: log LCP vs last turn (helps diagnose Traject multi-turn).
    session_lcp = 0
    if req.session_id:
        prev = _SESSION_LAST_IDS.get(req.session_id)
        if prev:
            session_lcp = _common_prefix_len(prev, input_ids)
            logger.info(
                "session %s prompt_lcp=%s prev_len=%s new_len=%s",
                req.session_id,
                session_lcp,
                len(prev),
                len(input_ids),
            )
        _SESSION_LAST_IDS[req.session_id] = list(input_ids)
    if req.prefix_id:
        _PREFIX_TOKEN_IDS[req.prefix_id] = list(input_ids)
    # Stash on request for streaming final usage enrichment.
    req._session_lcp = session_lcp  # type: ignore[attr-defined]

    if _TP_MODE:
        return await _generate_tp(req, request, input_ids)
    return await _generate_local(req, request, input_ids)


def _enrich_usage(item: Dict[str, Any], session_lcp: int) -> Dict[str, Any]:
    """Inject session LCP into final usage; raise cache_hit_tokens floor to max(v4, lcp)."""
    if session_lcp <= 0:
        return item
    usage = item.get("usage")
    if not isinstance(usage, dict):
        if item.get("finish_reason") is None and not item.get("error"):
            return item
        usage = {}
        item = dict(item)
        item["usage"] = usage
    else:
        item = dict(item)
        usage = dict(usage)
        item["usage"] = usage
    usage["session_lcp_tokens"] = int(session_lcp)
    v4_hit = int(usage.get("cache_hit_tokens") or 0)
    usage["cache_hit_tokens"] = max(v4_hit, int(session_lcp))
    return item


async def _generate_local(req: GenerationRequest, request: Request, input_ids: List[int]):
    assert LOOP is not None
    params = _params_from_req(req)
    session_lcp = int(getattr(req, "_session_lcp", 0) or 0)
    try:
        submitted = LOOP.submit(req.request_id, input_ids, params)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=429)

    async def ndjson_stream():
        dq = submitted.delta_queue
        while True:
            if await request.is_disconnected():
                LOOP.cancel(req.request_id)
                break
            try:
                item = await asyncio.get_event_loop().run_in_executor(None, dq.get, True, 0.5)
            except Exception:
                continue
            if item.get("finish_reason") is not None or item.get("usage") is not None:
                item = _enrich_usage(item, session_lcp)
            yield json.dumps(item, ensure_ascii=True) + "\n"
            if item.get("finish_reason") is not None or item.get("error"):
                break

    if req.stream:
        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
    return await _aggregate_ndjson(ndjson_stream(), input_ids)


def _tp_cuda_bind() -> None:
    """Bind the remapped single visible GPU (cuda:0)."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            torch.set_default_device("cuda")
    except Exception:
        pass


def _start_tp_cuda_thread() -> None:
    """Start the rank-0 CUDA worker that owns all Hybrid forwards."""
    global _TP_CUDA_Q, _TP_CUDA_THREAD
    if _TP_CUDA_THREAD is not None:
        return
    q: queue.Queue = queue.Queue()
    _TP_CUDA_Q = q

    def _loop() -> None:
        _tp_cuda_bind()
        logger.info("tp cuda worker thread bound device=0")
        while True:
            job = q.get()
            if job is None:
                break
            msg, holder = job
            try:
                with _TP_LOCK:
                    broadcast_obj(msg, src=0)
                    submitted = _apply_generate_msg(msg)
                    holder["submitted"] = submitted
                    holder["event"].set()  # allow HTTP side to start reading deltas
                    LOOP.pump_until_idle(timeout_s=float(msg.get("timeout_s", 300.0)))
                holder["done"].set()
            except Exception as e:
                holder["error"] = e
                holder["event"].set()
                holder["done"].set()

    _TP_CUDA_THREAD = threading.Thread(
        target=_loop, name="sglang-lite-tp-cuda", daemon=True
    )
    _TP_CUDA_THREAD.start()


async def _generate_tp(req: GenerationRequest, request: Request, input_ids: List[int]):
    """Rank-0 only: enqueue CUDA work, stream NDJSON deltas."""
    assert LOOP is not None
    if _TP_CUDA_Q is None:
        _start_tp_cuda_thread()
    session_lcp = int(getattr(req, "_session_lcp", 0) or 0)
    msg = _msg_generate(req, input_ids)
    holder: Dict[str, Any] = {
        "event": threading.Event(),
        "done": threading.Event(),
        "submitted": None,
        "error": None,
    }
    _TP_CUDA_Q.put((msg, holder))

    # Wait until submit finished (or error) without blocking the event loop hard.
    loop = asyncio.get_event_loop()
    while not holder["event"].is_set():
        await asyncio.sleep(0.005)
    if holder["error"] is not None:
        return JSONResponse({"error": str(holder["error"])}, status_code=500)
    submitted = holder["submitted"]

    async def ndjson_stream():
        dq = submitted.delta_queue
        try:
            while True:
                if await request.is_disconnected():
                    LOOP.cancel(req.request_id)
                    break
                try:
                    item = await loop.run_in_executor(None, dq.get, True, 0.5)
                except Exception:
                    if holder["done"].is_set() and dq.empty():
                        break
                    continue
                if item.get("finish_reason") is not None or item.get("usage") is not None:
                    item = _enrich_usage(item, session_lcp)
                yield json.dumps(item, ensure_ascii=True) + "\n"
                if item.get("finish_reason") is not None or item.get("error"):
                    break
        finally:
            while not holder["done"].is_set():
                await asyncio.sleep(0.01)

    if req.stream:
        return StreamingResponse(ndjson_stream(), media_type="application/x-ndjson")
    return await _aggregate_ndjson(ndjson_stream(), input_ids)


async def _aggregate_ndjson(ndjson_stream, input_ids: List[int]):
    text_parts: List[str] = []
    finish = "stop"
    usage = None
    error = None
    async for chunk in ndjson_stream:
        data = json.loads(chunk)
        if data.get("text"):
            text_parts.append(data["text"])
        if data.get("finish_reason"):
            finish = data["finish_reason"]
        if data.get("usage"):
            usage = data["usage"]
        if data.get("error"):
            error = data["error"]
    if error:
        return JSONResponse({"error": error}, status_code=500)
    return {
        "text": "".join(text_parts),
        "finish_reason": finish,
        "usage": usage
        or {
            "prompt_tokens": len(input_ids),
            "completion_tokens": 0,
            "total_tokens": len(input_ids),
            "cache_hit_tokens": 0,
        },
    }


def _tp_worker_loop() -> None:
    """Non-zero ranks: wait for broadcast ops and pump in lockstep with rank 0."""
    assert LOOP is not None
    logger.info("tp worker rank=%s waiting for broadcast", rank())
    while True:
        msg = broadcast_obj(None, src=0)
        if msg is None:
            logger.info("tp worker rank=%s shutdown", rank())
            break
        op = msg.get("op")
        if op == "generate":
            _apply_generate_msg(msg)
            LOOP.pump_until_idle(timeout_s=float(msg.get("timeout_s", 300.0)))
        elif op == "cancel":
            LOOP.cancel(str(msg["request_id"]))
        elif op == "warmup":
            _apply_generate_msg(msg)
            LOOP.pump_until_idle(timeout_s=float(msg.get("timeout_s", 60.0)))
        else:
            logger.warning("tp worker unknown op=%s", op)


def build_loop(
    model: str,
    device: str,
    allow_stub: bool,
    max_batch_size: int,
    *,
    background: bool,
) -> EngineLoop:
    runner = ModelRunner(model, device=device, max_batch=max_batch_size, allow_stub=allow_stub)
    loop = EngineLoop(runner, max_batch_size=max_batch_size)
    if background:
        loop.start()
    else:
        loop.mark_ready()
    return loop


def _warmup(loop: EngineLoop, *, tp: bool) -> None:
    runner = loop.runner
    if not runner._is_real:
        return
    ids = runner.tokenize("hi")[:4] or [1, 2]
    msg = {
        "op": "warmup",
        "request_id": "warmup",
        "input_ids": list(ids),
        "max_tokens": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "seed": None,
        "stop": None,
        "timeout_s": 60.0,
    }
    if tp:
        if rank() == 0:
            broadcast_obj(msg, src=0)
            sub = _apply_generate_msg(msg)
            loop.pump_until_idle(timeout_s=60.0)
            while True:
                item = sub.delta_queue.get(timeout=60.0)
                if item.get("error"):
                    raise RuntimeError(f"warmup failed: {item['error']}")
                if item.get("finish_reason") is not None:
                    break
        else:
            # Worker path handles warmup via _tp_worker_loop — but warmup runs
            # before the worker loop starts, so workers must participate here.
            got = broadcast_obj(None, src=0)
            assert got is not None and got.get("op") == "warmup"
            _apply_generate_msg(got)
            loop.pump_until_idle(timeout_s=60.0)
        return

    sub = loop.submit(
        "warmup",
        ids,
        GenParams(max_tokens=1, temperature=0.0, timeout_s=60.0),
    )
    while True:
        item = sub.delta_queue.get(timeout=60.0)
        if item.get("error"):
            raise RuntimeError(f"warmup failed: {item['error']}")
        if item.get("finish_reason") is not None:
            break


def main(argv: Optional[List[str]] = None) -> None:
    global LOOP, READY, MODEL_NAME, _TP_MODE
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(description="sglang-lite engine process")
    p.add_argument("--model", required=True, help="MoE model id or fixture:<path>")
    p.add_argument("--device", default="cpu")
    p.add_argument("--port", type=int, default=9001)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--allow-stub", action="store_true")
    args = p.parse_args(argv)

    MODEL_NAME = args.model
    READY = False
    tp = is_tp()
    _TP_MODE = tp
    background = not tp

    try:
        LOOP = build_loop(
            args.model,
            args.device,
            args.allow_stub,
            args.max_batch_size,
            background=background,
        )
        _warmup(LOOP, tp=tp)
        register_verified(args.model)
        READY = True
    except Exception:
        READY = False
        logger.exception("engine failed to become ready")
        raise

    logger.info(
        "engine ready model=%s device=%s port=%s tp=%s rank=%s/%s",
        args.model,
        args.device,
        args.port,
        tp,
        rank(),
        world_size(),
    )

    if tp and rank() == 0:
        _start_tp_cuda_thread()

    if tp and rank() != 0:
        try:
            _tp_worker_loop()
        finally:
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass
        return

    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if tp and rank() == 0:
            try:
                broadcast_obj(None, src=0)
            except Exception:
                pass
            try:
                import torch.distributed as dist

                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass


if __name__ == "__main__":
    main()
