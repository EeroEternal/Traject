"""
Demo / legacy example server (NOT the production path).

Production standalone entry:
  cargo run -p sglang-lite-serving -- serve --model <moe> --port 8000
  # which spawns: python -m sglang_lite.process

This file may still use a process-wide lock and is kept only for demos.
Do not treat it as the official continuous-batching serving path.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel

import sys

sys.path.insert(0, "engine")

from sglang_lite.config import Config
from sglang_lite import LiteEngine

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sglang_lite_server")


def _log(msg: str, **kwargs):
    log_data = {"ts": time.time(), "msg": msg, **kwargs}
    logger.info(json.dumps(log_data))


app = FastAPI(title="sglang-lite Python Core (real engine)")

# Global engine
ENGINE: Optional[LiteEngine] = None
ENGINE_LOCK = asyncio.Lock()


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    req_id = request.headers.get("x-request-id") or str(uuid.uuid4())[:12]
    request.state.request_id = req_id
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    _log(
        "http_request",
        request_id=req_id,
        method=request.method,
        path=str(request.url.path),
        duration=duration,
        status=response.status_code,
    )
    response.headers["x-request-id"] = req_id
    return response


class GenRequest(BaseModel):
    request_id: Optional[str] = None
    model: str = "stub"
    messages: Optional[List[dict]] = None
    input_ids: Optional[List[int]] = None
    max_tokens: int = 128
    temperature: float = 0.7
    stream: bool = False


class GenResult(BaseModel):
    text: str
    finish_reason: Optional[str] = "stop"
    usage: dict


def _extract_input_ids(req: GenRequest) -> List[int]:
    if req.input_ids:
        return req.input_ids

    # crude extraction from messages (last user message)
    last_user = ""
    if req.messages:
        for m in reversed(req.messages):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    last_user = content
                break
    if not last_user:
        last_user = "Hello"

    if ENGINE and ENGINE.runner.tokenizer:
        return ENGINE.runner.tokenize(last_user)
    return [hash(c) % 32000 for c in last_user[:80]]


@app.post("/generate", response_model=GenResult)
async def generate(req: GenRequest, request: Request):
    global ENGINE
    rid = req.request_id or getattr(request.state, "request_id", f"req-{uuid.uuid4().hex[:12]}")
    input_ids = _extract_input_ids(req)

    _log("generate_start", request_id=rid, max_tokens=req.max_tokens)
    async with ENGINE_LOCK:
        result = ENGINE.generate(
            rid,
            input_ids,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    _log("generate_end", request_id=rid, finish_reason=result.get("finish_reason"))
    return GenResult(
        text=result["text"],
        finish_reason=result.get("finish_reason"),
        usage=result.get("usage", {}),
    )


# OpenAI compatible endpoint for UniGateway HTTP mode
class OpenAIChatRequest(BaseModel):
    model: str
    messages: List[dict]
    max_tokens: Optional[int] = 128
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(req: OpenAIChatRequest, request: Request):
    global ENGINE
    rid = getattr(request.state, "request_id", f"chatcmpl-{uuid.uuid4().hex[:12]}")
    input_ids = _extract_input_ids(
        GenRequest(
            model=req.model,
            messages=req.messages,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )
    )

    if req.stream:

        async def stream_gen():
            async with ENGINE_LOCK:
                for delta in ENGINE.generate_stream(
                    rid,
                    input_ids,
                    max_tokens=req.max_tokens or 128,
                    temperature=req.temperature or 0.7,
                ):
                    chunk = {
                        "id": rid,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": req.model,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": delta.get("text", "")}
                                if delta.get("text")
                                else {},
                                "finish_reason": delta.get("finish_reason"),
                            }
                        ],
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        async with ENGINE_LOCK:
            result = ENGINE.generate(
                rid, input_ids, max_tokens=req.max_tokens or 128, temperature=req.temperature or 0.7
            )
        response = {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": result["text"]},
                    "finish_reason": result.get("finish_reason", "stop"),
                }
            ],
            "usage": result.get("usage", {}),
        }
        return response


@app.post("/generate_stream")
async def generate_stream(req: GenRequest):
    rid = req.request_id or f"req-{uuid.uuid4().hex[:12]}"
    input_ids = _extract_input_ids(req)

    async def event_generator():
        async with ENGINE_LOCK:
            for delta in ENGINE.generate_stream(
                rid, input_ids, max_tokens=req.max_tokens, temperature=req.temperature
            ):
                import json

                yield f"data: {json.dumps(delta)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    stats = ENGINE.get_stats() if ENGINE else {}
    return {
        "status": "ok",
        "phase": "0",
        "core": "sglang-lite-python",
        "stats": stats,
    }


@app.get("/stats")
async def stats():
    return ENGINE.get_stats() if ENGINE else {}


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=None,
        help="Enable gRPC on this port (corresponds to unigateway sglang-lite grpc driver)",
    )
    parser.add_argument("--model", type=str, default="stub")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-concurrent", type=int, default=32)
    args = parser.parse_args()

    cfg = Config.from_env("lite")
    # Override with CLI args if provided
    if args.model != "stub":
        cfg.model = args.model
    if args.device != "cpu":
        cfg.device = args.device
    cfg.port = args.port
    cfg.max_batch_size = args.max_batch_size
    cfg.max_concurrent = args.max_concurrent

    global ENGINE
    ENGINE = LiteEngine(model_name=cfg.model, device=cfg.device)

    print(f"[sglang-lite-core] Phase 1 engine starting on :{args.port}")
    print(f"  config: {cfg.to_dict()}")

    if args.grpc_port:
        import threading

        t = threading.Thread(target=_start_grpc_server, args=(args.grpc_port,), daemon=True)
        t.start()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level=cfg.log_level.lower())


# ---------------- gRPC support (to correspond with unigateway gRPC client skeleton) ----------------
HAS_GRPC = False
try:
    # gRPC generated code is kept out of the engine package; it is an integration artifact.
    sys.path.insert(0, "examples/grpc")
    import sglang_lite_pb2 as pb2
    import sglang_lite_pb2_grpc as pb2_grpc
    import grpc
    from concurrent import futures

    # standard health for unigateway client
    from grpc_health.v1 import health_pb2, health_pb2_grpc

    HAS_GRPC = True
except Exception:
    pass


if HAS_GRPC:

    class _SglangLiteServicer(pb2_grpc.SglangLiteServiceServicer):
        def ChatCompletions(self, request, context):
            global ENGINE
            if ENGINE is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "engine not ready")
            rid = f"chatcmpl-grpc-{uuid.uuid4().hex[:12]}"
            # map messages (basic)
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            # tools accepted but not executed in core (per scope); prompt construction stays in caller or upper layer
            if request.tools:
                _log("grpc_tools_received", count=len(request.tools))
            try:
                gen_req = GenRequest(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens or 128,
                    temperature=request.temperature or 0.7,
                )
                input_ids = _extract_input_ids(gen_req)
                result = ENGINE.generate(
                    rid, input_ids, max_tokens=gen_req.max_tokens, temperature=gen_req.temperature
                )
                usage = result.get("usage", {})
                # ensure cache_hit_tokens if present in result
                resp = pb2.ChatCompletionsResponse(
                    id=rid,
                    object="chat.completion",
                    created=int(time.time()),
                    model=request.model,
                    choices=[
                        pb2.Choice(
                            index=0,
                            message=pb2.Message(role="assistant", content=result.get("text", "")),
                            finish_reason=result.get("finish_reason", "stop"),
                        )
                    ],
                    usage=pb2.Usage(
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        total_tokens=usage.get("total_tokens", 0),
                        cache_hit_tokens=usage.get("cache_hit_tokens"),
                    ),
                )
                return resp
            except Exception as e:
                context.abort(grpc.StatusCode.INTERNAL, str(e))

        def ChatCompletionsStream(self, request, context):
            global ENGINE
            if ENGINE is None:
                context.abort(grpc.StatusCode.UNAVAILABLE, "engine not ready")
            rid = f"chatcmpl-grpc-{uuid.uuid4().hex[:12]}"
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            try:
                gen_req = GenRequest(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens or 128,
                    temperature=request.temperature or 0.7,
                )
                input_ids = _extract_input_ids(gen_req)
                for delta in ENGINE.generate_stream(
                    rid, input_ids, max_tokens=gen_req.max_tokens, temperature=gen_req.temperature
                ):
                    chunk = pb2.ChatCompletionChunk(
                        id=rid,
                        object="chat.completion.chunk",
                        created=int(time.time()),
                        model=request.model,
                        choices=[
                            pb2.ChunkChoice(
                                index=0,
                                delta=pb2.Delta(content=delta.get("text", ""))
                                if delta.get("text")
                                else None,
                                finish_reason=delta.get("finish_reason"),
                            )
                        ],
                    )
                    # if last delta had usage, attach (engine may put in final)
                    if "usage" in delta:
                        u = delta["usage"]
                        chunk.usage.CopyFrom(
                            pb2.Usage(
                                prompt_tokens=u.get("prompt_tokens", 0),
                                completion_tokens=u.get("completion_tokens", 0),
                                total_tokens=u.get("total_tokens", 0),
                                cache_hit_tokens=u.get("cache_hit_tokens"),
                            )
                        )
                    yield chunk
            except Exception as e:
                context.abort(grpc.StatusCode.INTERNAL, str(e))

        def Embeddings(self, request, context):
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "embeddings not implemented in core")

        def ListModels(self, request, context):
            # minimal
            return pb2.ListModelsResponse(object="list", data=[])


def _start_grpc_server(port: int):
    if not HAS_GRPC:
        print("gRPC not available (install grpcio grpcio-tools)")
        return
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pb2_grpc.add_SglangLiteServiceServicer_to_server(_SglangLiteServicer(), server)

    # Register standard gRPC health v1 (for unigateway client health checks)
    health_servicer = health_pb2_grpc.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    health_servicer.set("", health_pb2.HealthCheckResponse.SERVING)  # overall
    # Optionally set for the sglang service name too

    server.add_insecure_port(f"[::]:{port}")
    server.start()
    print(f"[sglang-lite] gRPC listening on :{port} (with standard health v1)")
    server.wait_for_termination()


# ---------------- end gRPC ----------------


if __name__ == "__main__":
    main()
