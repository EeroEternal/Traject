#!/usr/bin/env bash
# End-to-end agent on multi-GPU host (PRO 6000 / 5090) with Traject-owned sglang-lite.
# Prerequisites: engine ready on ENGINE_URL (bash scripts/start_engine.sh).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKDIR="${WORKDIR:-/tmp/traject-zene-demo}"
MODEL="${MODEL:-/home/bodesi/models/ds-v4-flash}"
ENGINE="${ENGINE_URL:-http://127.0.0.1:9001}"
BIN="${TRAJECT_BIN:-./target/release/traject}"
MAX_TURNS="${MAX_TURNS:-6}"
LOG="${E2E_LOG:-/tmp/traject-logs/e2e-agent.log}"
mkdir -p "$WORKDIR" "$(dirname "$LOG")"
echo "hello from traject+zene" > "$WORKDIR/hello.txt"
echo "second file" > "$WORKDIR/note.txt"

if [[ ! -x "$BIN" ]]; then
  echo "missing binary $BIN — run: cargo build -p traject-cli --release" >&2
  exit 1
fi
if ! curl -sf "${ENGINE}/readyz" >/dev/null 2>&1; then
  echo "engine not ready at $ENGINE — run: bash scripts/start_engine.sh" >&2
  exit 1
fi

export RUST_LOG="${RUST_LOG:-info,traject_runtime=info,traject_zene=info,traject_inference=info}"
set -x
"$BIN" agent \
  --engine-url "$ENGINE" \
  --model "$MODEL" \
  --workdir "$WORKDIR" \
  --max-turns "$MAX_TURNS" \
  --max-tokens "${MAX_TOKENS:-512}" \
  "先用 Glob 列出 *.txt，再 Read hello.txt，最后用一句话中文总结文件内容后停止。" \
  2>&1 | tee "$LOG"
set +x

echo "--- e2e checks ---"
# Accept either Chinese or English log fragments from Driver path.
if grep -Eq "generate step via driver/scheduler|sglang-lite generate finished|recorded generate" "$LOG"; then
  echo "OK: generate went through Traject inference"
else
  echo "WARN: did not find generate log markers" >&2
fi
if grep -Eq "tool step via driver/scheduler|recorded tool step|ToolDone|tool_steps" "$LOG"; then
  echo "OK: tool steps recorded on trajectory"
else
  echo "WARN: did not find tool step markers (model may have answered without tools)" >&2
fi
if grep -Eq "cache_hit|cache_hit_tokens" "$LOG"; then
  echo "OK: cache_hit fields present in logs"
  grep -E "cache_hit_tokens" "$LOG" | tail -5 || true
else
  echo "NOTE: no cache_hit log line (first turn may be cold)"
fi
# Soft success criteria: history should show multi-step agent work when tools used.
if grep -Eq "generate_steps=[1-9]|history_len=[2-9]|history_len=[1-9][0-9]" "$LOG"; then
  echo "OK: multi-step trajectory observed"
fi
echo "log: $LOG"
