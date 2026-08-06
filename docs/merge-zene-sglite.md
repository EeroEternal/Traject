# 揉入 Zene + sglang-lite 执行计划

把 Zene agent crates 与 sglang-lite（Python 引擎 + Rust control/serving）整仓并入 Traject，打通 Trajectory 感知的推理路径，使 Inference 能跟踪并优化 agent session，而不再依赖外部 HTTP 黑盒。

状态：本阶段代码已合入 `main`（见 PR #1）。5090 上的 GPU 端到端验收仍需在机器上按下方 Phase D 执行。

## 问题

当前形态曾是 **外部依赖 + HTTP 旁路**：Zene 经 OpenAI 打 sglang-lite `:8000`，`ChunkRequest.trajectory_id` / `prefix` 被忽略，Agent 绕过 `Driver` / `MemoryManager`。Inference 看不到 session，无法做跨 tool 步的 prefix / KV 优化。

## 选定形态（本阶段）

**仓库内自有、由 Traject 拥有并编排**；GPU 权重执行仍走 sglang-lite 的 Python 引擎子进程（已有 TP=8 / DeepSeek-V4 能力），但：

- 源码进 Traject 树，不再依赖 `~/project/sglang-lite` 或 `../zene`
- Agent 每步 LLM / Tool 进入同一 `Trajectory`
- 引擎协议携带 `trajectory_id` + prefix / session 句柄，与 [`MemoryManager`](../crates/traject-memory/src/manager.rs) 对齐

完整「Rust 同进程加载 MoE 权重」仍是后续 Phase（见 [roadmap.md](roadmap.md)），本阶段不改写整个 runner。

```mermaid
flowchart LR
  subgraph trajectProc [Traject process]
    Agent[Zene Agent vendored]
    Driver[Driver Scheduler]
    Mem[MemoryManager PrefixTree]
    Ctrl[sglang-lite control serving]
  end
  subgraph gpuProc [Owned engine subprocess]
    Eng[Python engine TP]
    Radix[v4_prefix_cache KV]
  end
  Agent -->|"Generate or Tool Step"| Driver
  Driver --> Mem
  Driver -->|"generate with trajectory_id prefix"| Ctrl
  Ctrl --> Eng
  Eng --> Radix
  Radix -->|"prefix hit metrics"| Mem
```

## Phase A — 源码并入仓库

**Zene（agent 栈，不含 cloud）**

- 将 `zene/crates/{config,core,sandbox,session,tools,llm,mcp}` 复制为 Traject workspace members：`crates/zene-*`（包名暂保留 `zene-*`）
- `crates/traject-zene` 改为 workspace 内依赖，去掉对外部 `../zene` 的 path
- 不引入 `zene/cloud`；Keel / sandbox 保持 “off” 默认以便 5090 集成跑通

**sglang-lite**

- 同步进 `third_party/sglang-lite/`：
  - `engine/` + `pyproject.toml`（Python 包）
  - Rust：`control`、`serving` 作为 workspace members（crate 名 `sglang-lite-control` / `sglang-lite-serving`）
- [`LocalEngineHandle`](../crates/traject-inference/src/backend/local_engine.rs) / [`scripts/start_engine.sh`](../scripts/start_engine.sh) 默认 `SGLANG_LITE_ROOT` 指向仓库内 `third_party/sglang-lite`

## Phase B — Agent 接到 Traject 调度主路径

- 改造 [`crates/traject-zene/src/runner.rs`](../crates/traject-zene/src/runner.rs)：不再让 `zene-llm` 直连外部 `base_url` 作为主路径
- 新增 `TrajectLlmProvider`（实现 zene `Provider`）：每次 `chat` 映射为同一 Trajectory 上的 `Generate` 步；tool 结果映射为 `Tool` 步并 `MemoryManager` pin / append
- `traject agent` 默认走引擎路径（`:9001`）；`--legacy-http` / tool-bridge 仅作兼容

## Phase C — Inference 侧 session / prefix 可见

- 扩展 [`ChunkRequest`](../crates/traject-inference/src/chunked.rs) → `SglangLiteEngineBackend` 请求体：带上 `trajectory_id`、`prefix_id`、`session_id`
- vendored `engine/` 的 generate 入口记录 session，并返回 `cache_hit_tokens` 写回 Trajectory / MemoryManager
- Agent 主路径优先 engine `/v1/generate`，不再经无状态 chat + tools 黑盒

## Phase D — 5090 验收

- 拉最新 `main`，用仓库内引擎：`bash scripts/start_engine.sh`
- `cargo build -p traject-cli --release`
- 跑 [`scripts/e2e_agent_5090.sh`](../scripts/e2e_agent_5090.sh)（Glob → Read 多步）
- 验收指标：日志出现 `trajectory_id` / `session_id` / `prefix_id`；同 Trajectory 后续 generate 有 `cache_hit_tokens`；history 可见连续 Generate / Tool 步

## 本阶段明确不做

- 不把完整 MoE runner 改写成纯 Rust 同进程
- 不 merge zene cloud / 多租户控制面
- 不删除 sglang-lite 原仓库；以 Traject 树内副本为唯一集成源

## 后续工作

- 引擎 radix / KV 与 Traject `MemoryManager` 真正对齐（可 pin / 复用 / 淘汰）
- Zene 每步完整走 `Driver` / `Scheduler`，而不只是 Trajectory 记账
- 同进程权重 runner（见 [kernels.md](kernels.md)）
- 降级并最终移除 `--legacy-http` / tool-bridge 主路径依赖
