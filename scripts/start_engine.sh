#!/usr/bin/env bash
# Start co-located sglang-lite GPU engine for Traject (TP=8 DeepSeek-V4-Flash).
# Usage: bash scripts/start_engine.sh
set -euo pipefail

VENV="${SGLANG_VENV:-/home/bodesi/sglang-dflash-venv}"
MODEL="${SGLANG_LITE_DSV4_HF:-/home/bodesi/models/ds-v4-flash}"
CONVERTED="${SGLANG_LITE_DSV4_CONVERTED:-/tmp/ds-v4-mp8}"
PORT="${ENGINE_PORT:-9001}"
TP="${TP:-8}"
LOG_DIR="${TRAJECT_LOG_DIR:-/tmp/traject-logs}"
LITE_ROOT="${SGLANG_LITE_ROOT:-$(cd "$(dirname "$0")/.." && pwd)/third_party/sglang-lite}"

mkdir -p "$LOG_DIR"
export PATH="/usr/local/cuda/bin:${VENV}/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export SGLANG_LITE_DSV4_HF="$MODEL"
export SGLANG_LITE_DSV4_CONVERTED="$CONVERTED"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"

if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
  echo "engine already ready on :${PORT}"
  curl -s "http://127.0.0.1:${PORT}/readyz"
  echo
  exit 0
fi

cd "$LITE_ROOT"
echo "starting torchrun TP=${TP} → :${PORT}"
nohup "${VENV}/bin/torchrun" --nproc-per-node="${TP}" -m sglang_lite.process \
  --model "${MODEL}" \
  --device cuda \
  --port "${PORT}" \
  --host 127.0.0.1 \
  > "${LOG_DIR}/engine-${PORT}.log" 2>&1 &
echo $! > "${LOG_DIR}/engine-${PORT}.pid"
echo "pid=$(cat "${LOG_DIR}/engine-${PORT}.pid") log=${LOG_DIR}/engine-${PORT}.log"

for i in $(seq 1 180); do
  if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
    echo "engine ready"
    curl -s "http://127.0.0.1:${PORT}/readyz"
    echo
    exit 0
  fi
  if ! kill -0 "$(cat "${LOG_DIR}/engine-${PORT}.pid")" 2>/dev/null; then
    echo "engine died; tail log:" >&2
    tail -40 "${LOG_DIR}/engine-${PORT}.log" >&2
    exit 1
  fi
  sleep 5
done
echo "timeout waiting for engine" >&2
exit 1
