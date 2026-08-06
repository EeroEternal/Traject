#!/usr/bin/env bash
# Official DeepSeek-V4-Flash reference smoke (P0).
#
# Converts HF safetensors via inference/convert.py, then short-generates with
# torchrun generate.py. Output is the gold baseline for Hybrid alignment.
#
# Example (8×5090):
#   HF_CKPT=~/models/ds-v4-flash \
#   SAVE_PATH=/tmp/ds-v4-mp8 \
#   MP=8 \
#   bash scripts/v4_official_smoke.sh
#
# Optional: SKIP_CONVERT=1 if SAVE_PATH already has converted shards.

set -euo pipefail

HF_CKPT="${HF_CKPT:-${HOME}/models/ds-v4-flash}"
INFER_DIR="${INFER_DIR:-${HF_CKPT}/inference}"
SAVE_PATH="${SAVE_PATH:-/tmp/ds-v4-mp${MP:-8}}"
MP="${MP:-8}"
EXPERTS="${EXPERTS:-256}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
PROMPT="${PROMPT:-Hello}"
SKIP_CONVERT="${SKIP_CONVERT:-0}"
PYTHON="${PYTHON:-python}"

if [[ ! -d "${INFER_DIR}" ]]; then
  echo "missing inference dir: ${INFER_DIR}" >&2
  exit 2
fi
if [[ ! -f "${HF_CKPT}/config.json" ]]; then
  echo "missing HF config: ${HF_CKPT}/config.json" >&2
  exit 2
fi

echo "[v4-smoke] HF_CKPT=${HF_CKPT}"
echo "[v4-smoke] SAVE_PATH=${SAVE_PATH} MP=${MP} EXPERTS=${EXPERTS}"
"${PYTHON}" - <<'PY'
import importlib, sys
need = ["torch", "transformers", "safetensors", "tilelang"]
optional = ["fast_hadamard_transform"]
missing = []
for n in need:
    try:
        importlib.import_module(n)
    except Exception as e:
        missing.append(f"{n}: {e}")
for n in optional:
    try:
        importlib.import_module(n)
        print(f"[v4-smoke] optional ok: {n}")
    except Exception as e:
        print(f"[v4-smoke] optional missing (may still run): {n}: {e}")
if missing:
    print("[v4-smoke] missing required deps:", file=sys.stderr)
    for m in missing:
        print(" ", m, file=sys.stderr)
    sys.exit(3)
import torch
print(f"[v4-smoke] torch={torch.__version__} cuda={torch.cuda.is_available()} n={torch.cuda.device_count() if torch.cuda.is_available() else 0}")
PY

mkdir -p "${SAVE_PATH}"
# generate.py expects ModelArgs-shaped JSON (inference/config.json), NOT HF config.json.
INFER_CONFIG="${INFER_DIR}/config.json"
if [[ ! -f "${INFER_CONFIG}" ]]; then
  echo "missing ${INFER_CONFIG}" >&2
  exit 2
fi
cp -f "${INFER_CONFIG}" "${SAVE_PATH}/config.json"

# TileLang JIT: use system CUDA toolkit only. Mixing pip nvidia/cu13 headers with
# /usr/local/cuda nvcc causes "CUDA compiler and CUDA toolkit headers are incompatible".
if [[ -x /usr/local/cuda/bin/nvcc ]]; then
  export PATH="/usr/local/cuda/bin:${PATH}"
  export CUDA_HOME=/usr/local/cuda
  export CUDA_PATH=/usr/local/cuda
fi
export CPATH=/usr/local/cuda/include
export CPLUS_INCLUDE_PATH=/usr/local/cuda/include
# Optional: fast_hadamard_transform (required by official model.py HC path).
#   PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=12.0 \
#     pip install --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git

if [[ "${SKIP_CONVERT}" != "1" ]]; then
  if [[ -f "${SAVE_PATH}/model0.safetensors" ]] || [[ -f "${SAVE_PATH}/model0-mp${MP}.safetensors" ]] || compgen -G "${SAVE_PATH}/*.safetensors" > /dev/null; then
    echo "[v4-smoke] SAVE_PATH already has safetensors; set SKIP_CONVERT=0 FORCE_CONVERT=1 to redo"
    if [[ "${FORCE_CONVERT:-0}" == "1" ]]; then
      echo "[v4-smoke] FORCE_CONVERT=1 — converting"
      "${PYTHON}" "${INFER_DIR}/convert.py" \
        --hf-ckpt-path "${HF_CKPT}" \
        --save-path "${SAVE_PATH}" \
        --n-experts "${EXPERTS}" \
        --model-parallel "${MP}"
    fi
  else
    echo "[v4-smoke] converting HF → MP=${MP} (long running)…"
    "${PYTHON}" "${INFER_DIR}/convert.py" \
      --hf-ckpt-path "${HF_CKPT}" \
      --save-path "${SAVE_PATH}" \
      --n-experts "${EXPERTS}" \
      --model-parallel "${MP}"
  fi
else
  echo "[v4-smoke] SKIP_CONVERT=1"
fi

# Write a one-line prompt file for batch mode (non-interactive).
PROMPT_FILE="${SAVE_PATH}/smoke_prompt.txt"
printf '%s\n' "${PROMPT}" > "${PROMPT_FILE}"

echo "[v4-smoke] torchrun generate MP=${MP} max_new_tokens=${MAX_NEW_TOKENS}"
# generate.py flags vary slightly by snapshot; prefer --input-file batch path.
cd "${INFER_DIR}"
torchrun --nproc-per-node "${MP}" generate.py \
  --ckpt-path "${SAVE_PATH}" \
  --config "${INFER_CONFIG}" \
  --input-file "${PROMPT_FILE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --temperature 0.0 \
  2>&1 | tee "${SAVE_PATH}/smoke_generate.log"

echo "[v4-smoke] done; log=${SAVE_PATH}/smoke_generate.log"
