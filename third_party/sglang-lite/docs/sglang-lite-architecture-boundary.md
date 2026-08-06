# sglang-lite 架构边界与进一步演进方向

## 1. 总体定位

sglang-lite 是一个可独立运行的极简 MoE inference backend；其中 `engine/` 是纯
Token Factory 库，`control/` + `serving/` 是官方薄服务入口。

**核心原则**：
- 只保留三个高内聚构建块：RadixKVCache、BatchingScheduler、MoEModelRunner。
- 内部允许进一步拆分以提升可组合性。
- engine 主循环、KV/调度/执行编排、engine-local admission/cancel 必须留在
  sglang-lite；只有多后端 routing、auth、全局 policy 等上移到可选 gateway。
- 保持代码量极小，专注 MoE 下的 Radix prefix sharing 和连续批处理。

## 2. 三个核心构建块的进一步内部拆分

为方便 sglang-lite 自身的 composition root 进行组合和测试，我们对三个组件进行
内部模块化拆分；跨进程 gateway 不直接操作这些 Python 类：

### RadixKVCache
- RadixTree（纯 token 前缀树逻辑）
- KVAllocator + MemoryManager（block 分配、引用计数、内存预算）
- EvictionPolicy（可插拔的驱逐策略）

### BatchingScheduler
- SequenceTable（序列生命周期管理）
- BatchFormer（批次形成策略）
- engine-local admission、超时/取消清理、队列上限和 backpressure

### MoEModelRunner
- ModelLoader
- MoERouter
- PrefillExecutor
- DecodeExecutor（CUDA graph 可选、保守实现）
- KernelBackend（attention 执行委托给 FlashInfer / Triton 等外部库）

**LiteEngine** 可保持薄 facade，但真正的编排、主循环和 sequence 生命周期必须由
sglang-lite engine process 内的 composition root 负责。Rust 控制面负责外部请求
生命周期并通过 HTTP/gRPC 传播 cancel、timeout 和 token delta。

## 3. 独立运行与 UniGateway 的关系

- **sglang-lite engine**：直接组合三个构建块，拥有中央 continuous batching loop。
- **sglang-lite control/serving**：提供最小 OpenAI chat/stream/models/health/readiness、
  engine-local queue/timeout、错误和优雅退出。
- **UniGateway（可选）**：
  - 只通过 HTTP/gRPC 把 sglang-lite 当作一个 local-inference backend 调用。
  - 负责多后端 routing、auth、全局 rate-limit、租户 policy、failover、metrics 聚合。
  - 不直接 import、组合或替换 sglang-lite 的内部构建块。

**目标**：用户不部署 UniGateway 也能直接使用 sglang-lite；需要统一多个 backend 时
再选择 UniGateway。

### 与 vLLM 的兼容边界

sglang-lite 不应被设计成 SGLang 专用后端，也不应追求 vLLM 功能面全兼容。更合理的
定位是：sglang-lite 与 vLLM 是同层级的 `local-inference` backend，可独立运行，也可
由 UniGateway 统一管理。

- unigateway core 使用通用抽象：`PrefixCache`、`BlockKVCache`、`ContinuousScheduler`、`ModelExecutor`、`BackendCapabilities`。
- sglang-lite 内部继续保持 RadixKVCache + MoE-aware batching；vLLM 内部可使用 APC + PagedAttention。
- 两者在协议层共享 OpenAI-compatible chat/stream/models/health、request id 透传、prefix-cache metrics（如 `usage.cache_hit_tokens`）。
- vLLM 的宽 API（Responses、multimodal、LoRA、spec decode、disagg 等）不进入 sglang-lite core。

详细评估见 `docs/vllm-positioning-compatibility.md`。

## 4. 核心外的薄层与可选上游职责

- Tokenizer/chat template：属于模型执行语义，由 sglang-lite engine 持有，避免调用方
  与模型版本不一致。
- Metrics：engine/control 暴露 standalone 必需指标；UniGateway 可选聚合。
- 配置：standalone 提供最小启动配置；跨后端和租户策略由 UniGateway 控制。
- Batching policy：必须在 engine 内执行；外部只能通过稳定协议传递高层 hint。
- Attention backend：由 sglang-lite 内部根据模型、硬件和配置选择。

## 5. 演进建议

1. 在 sglang-lite engine process 内建立中央 async engine loop。
2. 用真实 TokenDelta 协议连接 Rust control/serving 与 Python engine。
3. 在 UniGateway 侧可选实现通用 LocalInferenceDriver，只调用独立服务接口。
4. 持续内部拆分，提升可测试性和可替换性，但不要过度解耦影响性能。

具体补齐路线见 `docs/standalone-inference-service-roadmap.md`。

此文档用于 sglang-lite 团队内部对齐演进方向。
