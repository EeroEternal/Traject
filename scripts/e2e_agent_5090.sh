#!/usr/bin/env bash
# End-to-end agent on a 5090 host with Traject-owned sglang-lite.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
WORKDIR="${WORKDIR:-/tmp/traject-zene-demo}"
MODEL="${MODEL:-/home/bodesi/models/ds-v4-flash}"
ENGINE="${ENGINE_URL:-http://127.0.0.1:9001}"
mkdir -p "$WORKDIR"
echo "hello from traject+zene" > "$WORKDIR/hello.txt"
echo "second file" > "$WORKDIR/note.txt"

./target/release/traject agent \
  --engine-url "$ENGINE" \
  --model "$MODEL" \
  --workdir "$WORKDIR" \
  --max-turns 6 \
  "先用 Glob 列出 *.txt，再 Read hello.txt，最后用一句话中文总结文件内容后停止。"
