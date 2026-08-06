#!/usr/bin/env bash
# Standalone OpenAI SSE for DeepSeek-V4-Flash (TP=8 Hybrid).
#
# Terminal A — engine (rank0 HTTP :9001):
#   export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
#   export SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8
#   PATH=/usr/local/cuda/bin:$PATH CPATH=/usr/local/cuda/include \
#     bash scripts/v4_serve_sse.sh engine
#
# Terminal B — Rust control plane (:8000):
#   bash scripts/v4_serve_sse.sh control
#
# Client:
#   curl -N http://127.0.0.1:8000/v1/chat/completions \
#     -H 'Content-Type: application/json' \
#     -d '{"model":"'"${MODEL}"'","messages":[{"role":"user","content":"Hello"}],"max_tokens":16,"stream":true}'

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL="${SGLANG_LITE_DSV4_HF:-$HOME/models/ds-v4-flash}"
ENGINE_PORT="${ENGINE_PORT:-9001}"
CONTROL_PORT="${CONTROL_PORT:-8000}"
TP="${TP:-8}"
MODE="${1:-all}"

export PATH="/usr/local/cuda/bin:${PATH:-}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export SGLANG_LITE_DSV4_HF="${SGLANG_LITE_DSV4_HF:-$MODEL}"
export SGLANG_LITE_DSV4_CONVERTED="${SGLANG_LITE_DSV4_CONVERTED:-/tmp/ds-v4-mp8}"

run_engine() {
  echo "[v4-sse] torchrun TP=${TP} engine → :${ENGINE_PORT} model=${MODEL}"
  exec torchrun --nproc-per-node="${TP}" -m sglang_lite.process \
    --model "${MODEL}" \
    --device cuda \
    --port "${ENGINE_PORT}" \
    --host 127.0.0.1
}

run_control() {
  echo "[v4-sse] rust serving → :${CONTROL_PORT} engine=http://127.0.0.1:${ENGINE_PORT}"
  exec cargo run -p sglang-lite-serving --release -- serve \
    --model "${MODEL}" \
    --engine-url "http://127.0.0.1:${ENGINE_PORT}" \
    --port "${CONTROL_PORT}"
}

run_all() {
  echo "[v4-sse] single-process: serving spawns torchrun TP=${TP}"
  exec cargo run -p sglang-lite-serving --release -- serve \
    --model "${MODEL}" \
    --device cuda \
    --port "${CONTROL_PORT}" \
    --engine-port "${ENGINE_PORT}" \
    --tp "${TP}"
}

case "${MODE}" in
  engine) run_engine ;;
  control) run_control ;;
  all) run_all ;;
  *)
    echo "usage: $0 [engine|control|all]" >&2
    exit 2
    ;;
esac
