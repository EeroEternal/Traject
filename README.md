# Traject

Agent-native Runtime：以 **Trajectory** 为调度单位，把 Agent 轨迹执行与 LLM 推理合并在同一系统。

## 自有运行时（已揉入）

- **Zene agent 栈**：`crates/zene-*`（config/core/llm/tools/sandbox/session/mcp），无 cloud
- **sglang-lite**：`third_party/sglang-lite/`（Python `engine/` + Rust `control`/`serving`）
- Agent LLM 主路径：`TrajectLlmProvider` → **Driver / Scheduler** → Generate/Tool 步 → 引擎 `:9001`（`trajectory_id`/`session_id`/`prefix_id`，回写 `cache_hit_tokens` 到 MemoryManager）
- MemoryManager 持有 prefix 句柄（pin / 复用 / 淘汰评分）；物理 KV 仍在 sglang-lite 进程
- 执行计划：[docs/merge-zene-sglite.md](docs/merge-zene-sglite.md)
- 真机 e2e 记录：[docs/e2e-pro6000.md](docs/e2e-pro6000.md)

## 进程内 runner（物理 paged KV，玩具权重）

```bash
cargo run -p traject-cli -- --local-runner --max-tokens 16 "hello from local runner"
```

## 远端真实推理

```bash
# 引擎默认用仓库内 third_party/sglang-lite
bash scripts/start_engine.sh

# 简单 Generate
cargo run -p traject-cli --release -- \
  --engine-url http://127.0.0.1:9001 \
  --model /home/bodesi/models/ds-v4-flash \
  --max-tokens 48 \
  "用一句话介绍你自己"

# 完整 Zene coding agent（Traject 拥有 session / prefix）
cargo run -p traject-cli --release -- agent \
  --engine-url http://127.0.0.1:9001 \
  --model /home/bodesi/models/ds-v4-flash \
  --workdir /tmp/traject-zene-demo \
  --max-turns 6 \
  "先用 Glob 列出 *.txt，再 Read hello.txt，最后中文总结"

# 兼容旧 OpenAI 控制面（:8000 + tool-bridge）
cargo run -p traject-cli --release -- agent --legacy-http --backend-url http://127.0.0.1:8000/v1 ...
```
