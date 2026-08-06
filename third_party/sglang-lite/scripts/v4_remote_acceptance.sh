#!/usr/bin/env bash
# 8×GPU acceptance: official gold → Lite token align → OpenAI SSE smoke.
#
#   export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
#   export SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8
#   bash scripts/v4_remote_acceptance.sh
#
# Steps: align | sse | all (default)

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
HF="${SGLANG_LITE_DSV4_HF:-$HOME/models/ds-v4-flash}"
CONV="${SGLANG_LITE_DSV4_CONVERTED:-/tmp/ds-v4-mp8}"
TP="${TP:-8}"
PROMPT="${PROMPT:-Hello}"
MAX_NEW="${MAX_NEW:-8}"
ENGINE_PORT="${ENGINE_PORT:-9001}"
CONTROL_PORT="${CONTROL_PORT:-8000}"
GOLD_FILE="${GOLD_FILE:-/tmp/v4_align_gold.txt}"
SUMMARY="${SUMMARY:-/tmp/v4_align_summary.json}"
SSE_LOG="${SSE_LOG:-/tmp/v4_sse_smoke.log}"

export SGLANG_LITE_DSV4_HF="$HF"
export SGLANG_LITE_DSV4_CONVERTED="$CONV"
# Numerical align: stay on official TileLang sparse_attn (FI SM120 still blocked).
export SGLANG_LITE_V4_DISABLE_FI_SPARSE="${SGLANG_LITE_V4_DISABLE_FI_SPARSE:-1}"
export PATH="/usr/local/cuda/bin:${PATH:-}"
export CPATH="/usr/local/cuda/include:${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
if [[ -f "$HOME/sglang-dflash-venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/sglang-dflash-venv/bin/activate"
fi
# Editable install keeps ``import sglang_lite`` → engine/ mapping.
pip install -e "$ROOT" --no-deps -q 2>/dev/null || true

run_official_gold() {
  local prompt_file="$CONV/accept_prompt.txt"
  local log="$CONV/accept_official.log"
  printf '%s\n' "$PROMPT" >"$prompt_file"
  echo "[accept] official generate.py → gold"
  torchrun --nproc-per-node="$TP" \
    "$HF/inference/generate.py" \
    --ckpt-path "$CONV" \
    --config "$HF/inference/config.json" \
    --input-file "$prompt_file" \
    --max-new-tokens "$MAX_NEW" \
    --temperature 0.0 \
    >"$log" 2>&1 || {
    echo "[accept] official generate failed; tail:" >&2
    tail -n 80 "$log" >&2
    return 1
  }
  # "Completion: …"
  local gold
  gold="$(python3 - <<'PY' "$log"
import re, sys
text = open(sys.argv[1], encoding="utf-8", errors="replace").read()
m = re.findall(r"^Completion:\s*(.*)$", text, flags=re.M)
if not m:
    raise SystemExit("no Completion: line in official log")
print(m[-1].strip())
PY
)"
  printf '%s\n' "$gold" >"$GOLD_FILE"
  echo "[accept] gold=$gold"
}

run_align() {
  run_official_gold
  echo "[accept] LiteEngine align"
  torchrun --nproc-per-node="$TP" scripts/v4_align_tokens.py \
    --hf-ckpt "$HF" \
    --prompt "$PROMPT" \
    --max-new-tokens "$MAX_NEW" \
    --gold-file "$GOLD_FILE" \
    --out "$SUMMARY"
  python3 - <<'PY' "$SUMMARY"
import json, sys
s = json.load(open(sys.argv[1]))
assert s.get("match") is True, s
kind = "exact" if s.get("match_exact") else "soft_top5"
print(f"[accept] ALIGN PASS ({kind})")
print("prefill_top5=", s.get("prefill_top5"))
PY
}

wait_ready() {
  local url="$1" timeout="${2:-1800}"
  local start
  start="$(date +%s)"
  while true; do
    if curl -sf "$url/readyz" >/dev/null 2>&1; then
      return 0
    fi
    if (( "$(date +%s)" - start > timeout )); then
      echo "[accept] timeout waiting for $url/readyz" >&2
      return 1
    fi
    sleep 2
  done
}

run_sse() {
  echo "[accept] building rust serving (release)"
  cargo build -p sglang-lite-serving --release

  echo "[accept] starting torchrun engine on :$ENGINE_PORT"
  torchrun --nproc-per-node="$TP" -m sglang_lite.process \
    --model "$HF" --device cuda --port "$ENGINE_PORT" --host 127.0.0.1 \
    >"$SSE_LOG" 2>&1 &
  local eng_pid=$!

  cleanup() {
    kill "$ctrl_pid" 2>/dev/null || true
    kill "$eng_pid" 2>/dev/null || true
    # torchrun children
    pkill -P "$eng_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  wait_ready "http://127.0.0.1:$ENGINE_PORT" 1800

  echo "[accept] starting rust control on :$CONTROL_PORT"
  ./target/release/sglang-lite-serving serve \
    --model "$HF" \
    --engine-url "http://127.0.0.1:$ENGINE_PORT" \
    --port "$CONTROL_PORT" \
    >/tmp/v4_sse_control.log 2>&1 &
  local ctrl_pid=$!

  wait_ready "http://127.0.0.1:$CONTROL_PORT" 120

  local body sse_out
  sse_out="$(mktemp)"
  body="$(python3 - <<PY
import json
print(json.dumps({
  "model": "$HF",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 16,
  "temperature": 0.0,
  "stream": True,
}))
PY
)"
  echo "[accept] curl SSE"
  curl -sN --max-time 300 \
    "http://127.0.0.1:$CONTROL_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "$body" | tee "$sse_out"

  grep -q 'text/event-stream\|chat.completion.chunk' "$sse_out" || true
  grep -q '\[DONE\]' "$sse_out"
  grep -q 'chat.completion.chunk' "$sse_out"
  echo
  echo "[accept] SSE PASS (saw chat.completion.chunk + [DONE])"
  printf '%s\n' "ok" > /tmp/v4_sse_smoke.ok
}

case "$MODE" in
  align) run_align ;;
  sse) run_sse ;;
  all)
    run_align
    run_sse
    ;;
  *)
    echo "usage: $0 [align|sse|all]" >&2
    exit 2
    ;;
esac

echo "[accept] DONE mode=$MODE"
