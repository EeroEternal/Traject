#!/usr/bin/env bash
# Start co-located sglang-lite GPU engine for Traject (TP=8 DeepSeek-V4-Flash).
# Usage: bash scripts/start_engine.sh
#
# Env overrides (pro6000 defaults shown):
#   SGLANG_VENV              /home/bodesi/venvs/sglang-lite
#   SGLANG_LITE_DSV4_HF      /home/bodesi/models/ds-v4-flash
#   SGLANG_LITE_DSV4_CONVERTED /home/bodesi/models/ds-v4-mp8
#   ENGINE_PORT              9001
#   TP                       8
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# Prefer machine venv; fall back to legacy path.
if [[ -z "${SGLANG_VENV:-}" ]]; then
  if [[ -x /home/bodesi/venvs/sglang-lite/bin/python ]]; then
    SGLANG_VENV=/home/bodesi/venvs/sglang-lite
  elif [[ -x /home/bodesi/sglang-dflash-venv/bin/python ]]; then
    SGLANG_VENV=/home/bodesi/sglang-dflash-venv
  else
    SGLANG_VENV=""
  fi
fi
VENV="${SGLANG_VENV}"
MODEL="${SGLANG_LITE_DSV4_HF:-/home/bodesi/models/ds-v4-flash}"
if [[ -z "${SGLANG_LITE_DSV4_CONVERTED:-}" ]]; then
  if [[ -d /home/bodesi/models/ds-v4-mp8 ]]; then
    CONVERTED=/home/bodesi/models/ds-v4-mp8
  else
    CONVERTED=/tmp/ds-v4-mp8
  fi
else
  CONVERTED="${SGLANG_LITE_DSV4_CONVERTED}"
fi
PORT="${ENGINE_PORT:-9001}"
TP="${TP:-8}"
LOG_DIR="${TRAJECT_LOG_DIR:-/tmp/traject-logs}"
LITE_ROOT="${SGLANG_LITE_ROOT:-${ROOT}/third_party/sglang-lite}"

mkdir -p "$LOG_DIR"
if [[ -z "$VENV" || ! -x "${VENV}/bin/python" ]]; then
  echo "SGLANG_VENV not found (set SGLANG_VENV to a torch+cuda venv)" >&2
  exit 1
fi
export PATH="/usr/local/cuda/bin:${VENV}/bin:${PATH}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export SGLANG_LITE_DSV4_HF="$MODEL"
export SGLANG_LITE_DSV4_CONVERTED="$CONVERTED"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
# Ensure vendored package is importable even if not pip-installed.
export PYTHONPATH="${LITE_ROOT}/engine:${LITE_ROOT}:${PYTHONPATH:-}"

if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
  echo "engine already ready on :${PORT}"
  curl -s "http://127.0.0.1:${PORT}/readyz"
  echo
  exit 0
fi

# package-dir maps sglang_lite → engine/; PYTHONPATH needs parent or pip -e.
cd "$LITE_ROOT"
echo "starting torchrun TP=${TP} → :${PORT}"
echo "  VENV=$VENV"
echo "  MODEL=$MODEL"
echo "  CONVERTED=$CONVERTED"
echo "  LITE_ROOT=$LITE_ROOT"
# Prefer installed module; PYTHONPATH covers vendored tree via package-dir layout.
# setuptools package-dir means import path is still sglang_lite when installed -e.
# For uninstalled tree, add a sitecustomize-style path via PYTHONPATH pointing at
# parent and relying on pip -e; try both.
if ! "${VENV}/bin/python" -c "import sglang_lite" 2>/dev/null; then
  echo "installing vendored sglang-lite editable into venv..."
  "${VENV}/bin/pip" install -e "$LITE_ROOT" --no-deps -q
fi

nohup "${VENV}/bin/torchrun" --nproc-per-node="${TP}" -m sglang_lite.process \
  --model "${MODEL}" \
  --device cuda \
  --port "${PORT}" \
  --host 127.0.0.1 \
  > "${LOG_DIR}/engine-${PORT}.log" 2>&1 &
echo $! > "${LOG_DIR}/engine-${PORT}.pid"
echo "pid=$(cat "${LOG_DIR}/engine-${PORT}.pid") log=${LOG_DIR}/engine-${PORT}.log"

# Model load on 8×96GB can take many minutes.
for i in $(seq 1 360); do
  if curl -sf "http://127.0.0.1:${PORT}/readyz" >/dev/null 2>&1; then
    echo "engine ready after ~$((i * 5))s"
    curl -s "http://127.0.0.1:${PORT}/readyz"
    echo
    exit 0
  fi
  if ! kill -0 "$(cat "${LOG_DIR}/engine-${PORT}.pid")" 2>/dev/null; then
    echo "engine died; tail log:" >&2
    tail -80 "${LOG_DIR}/engine-${PORT}.log" >&2
    exit 1
  fi
  if (( i % 12 == 0 )); then
    echo "still waiting for engine... ($((i * 5))s) last log:"
    tail -3 "${LOG_DIR}/engine-${PORT}.log" || true
  fi
  sleep 5
done
echo "timeout waiting for engine" >&2
tail -80 "${LOG_DIR}/engine-${PORT}.log" >&2 || true
exit 1
