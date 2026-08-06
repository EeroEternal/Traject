# sglang-lite Serving Module (serving/)

**重要**：这个目录**只做极薄的包装和组合**。

## 定位（严格边界）

- sglang-lite 的 `engine/` 是**纯引擎库**（Token Factory）。
- sglang-lite 的 Rust 控制面库在 `control/`，包含 OpenAI 协议最小适配、内部协议、engine 客户端。
- sglang-lite 必须能够独立运行；Unigateway 是可选的高级网关。
- `serving/` 只提供：
  - 组合 `control/` 形成的官方 standalone wrapper
  - 启动 Python engine process、等待 `/readyz`、暴露 OpenAI 表面
  - 单引擎启动配置、健康/就绪、优雅退出

**不允许**在这里实现模型执行、跨后端 routing、auth、业务逻辑或宽 OpenAI 表面。

## 用法

```bash
# 真实 MoE（会 spawn: python -m sglang_lite.process）
cargo run -p sglang-lite-serving -- serve \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --device cuda \
  --port 8000

# 已有 engine process
cargo run -p sglang-lite-serving -- serve \
  --model mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --engine-url http://127.0.0.1:9001 \
  --port 8000

# 控制面 stub（无 GPU / 无模型）
cargo run -p sglang-lite-serving -- serve --stub --port 8000

# DeepSeek-V4-Flash（TP=8 Hybrid，spawn torchrun engine）
export SGLANG_LITE_DSV4_HF=~/models/ds-v4-flash
export SGLANG_LITE_DSV4_CONVERTED=/tmp/ds-v4-mp8
cargo run -p sglang-lite-serving --release -- serve \
  --model "$SGLANG_LITE_DSV4_HF" \
  --device cuda --port 8000 --tp 8

# 或分两步：scripts/v4_serve_sse.sh engine / control
```

流式路径：Rust `HttpEngineClient` 消费 Python NDJSON `TokenDelta`，再打成 OpenAI SSE（含最终 `usage` 与 `data: [DONE]`）。

## 开发规则

- 这里**只用 Rust**。
- 保持极薄：只做组合与进程生命周期。
- 参考 `AGENTS.md` 中的“模块边界定义”。
