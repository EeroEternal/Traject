# UniGateway 支持 sglang-lite 作为后端的需求描述

**重要边界声明**：
- 本文档及未来任何相关需求，**均不包含 PyO3、直接 Python 嵌入或同进程库调用**。
- sglang-lite 必须作为独立服务运行，UniGateway 仅通过 HTTP 或 gRPC 进行交互。
- 此边界旨在保护 UniGateway 作为通用 Rust SDK 的构建体验和可嵌入性。

## 背景

`sglang-lite` 是一个极简的纯 MoE 推理引擎库（Token Factory），其设计目标是保持高内聚和极小的代码量。核心仅包含三个组件：

- RadixKVCache（前缀共享）
- BatchingScheduler（MoE-aware 连续批处理）
- MoEModelRunner（专家路由 + 执行）

sglang-lite 自带独立运行所需的最小 chat/stream/models/health/readiness、engine-local
准入/取消、配置和生命周期能力。多后端 routing、鉴权、全局限流/策略和聚合能力
剥离到上层。

UniGateway 作为面向嵌入的**通用 LLM 网关 SDK**（库 workspace），通过可插拔的 `ProviderDriver` 机制支持不同后端。目标是让 UniGateway 能够将 sglang-lite 作为一种**本地 MoE 执行后端**接入，同时保持 UniGateway 自身的通用性、可扩展性和可嵌入性。

同时需要明确：sglang-lite 不应成为 UniGateway 中唯一被特殊对待的本地推理后端。vLLM 与 sglang-lite 应被视作同一层级的 `local-inference` backend：

- vLLM：通用 LLM/VLM serving engine，API/模型/硬件覆盖更宽。
- sglang-lite：可独立运行、MoE-only、prefix-heavy workload 优先的极简 inference backend。
- UniGateway：统一 OpenAI surface、routing、auth、metrics、capability negotiation。

详细比较见 `docs/vllm-positioning-compatibility.md`。

## 总体目标

1. 在 UniGateway 中支持 sglang-lite 作为一种可插拔的 backend/driver。
2. 用户可通过 UniGateway 统一的 OpenAI 兼容接口访问本地 MoE 模型。
3. 发挥 sglang-lite 在 Radix 前缀缓存和 MoE 连续批处理上的优势。
4. 所有 sglang-lite 特有逻辑严格隔离在 driver 实现内部，不污染 UniGateway 核心抽象。
5. 保持 UniGateway 为可选层；sglang-lite 本身的独立服务能力不依赖该集成。

## 非目标

- 不在 UniGateway 核心（engine、protocol、pool、routing 等）引入 MoE 或 sglang-lite 特有概念。
- 不在 UniGateway 核心引入只服务于 sglang-lite 的 provider 分支；应优先抽象为 generic `LocalInferenceDriver` / capability 系统，使 vLLM、sglang-lite 等本地后端都能复用。
- 不在 UniGateway 中实现 attention backend 选择（这是引擎内部职责）。
- 不在 UniGateway 中实现 sglang-lite 的编排、KV 管理或调度逻辑。
- **明确禁止引入 PyO3 / 直接 Python 库嵌入**：
  - UniGateway 不得通过 PyO3 直接嵌入或调用 sglang-lite 的 Python 库。
  - 所有交互必须通过标准传输方式（HTTP 或 gRPC）。
  - 理由：PyO3 会改变 UniGateway 的构建形态（需要 Python 运行时）、显著提高 MSRV/CI/交叉编译复杂度，并与 UniGateway 作为通用可嵌入 SDK 的定位冲突。
- 目前不要求完整 Expert Parallelism，先实现基础本地 MoE 运行能力。
- 不要让 UniGateway 对 sglang-lite 产生硬依赖（建议使用可选 feature 或独立 crate）。

### 集成边界（重要）

**严禁在 UniGateway 中引入 PyO3 或任何形式的直接 Python 库嵌入**：

- sglang-lite 必须作为独立的进程或服务运行。
- UniGateway 只能通过 **HTTP** 或 **gRPC** 与其通信。
- 不得使用 PyO3、ctypes、subprocess + Python embedding 等方式在 UniGateway 进程内直接加载 sglang-lite 的 Python 代码。
- 此边界是为了保护 UniGateway 作为通用 Rust SDK 的可构建性、可嵌入性和跨平台特性。

此边界已作为 Non-Goal 写入本文档，后续任何需求都不会再提出 PyO3 相关集成方式。

## 功能需求

### 1. Driver 支持
- 实现 `SglangLiteDriver`（或 `LocalMoEDriver`），实现 `ProviderDriver` 接口。
- 通过 `driver_id = "sglang-lite"` 识别和配置。
- 支持 `ProxyChatRequest`（streaming + non-streaming）。
- 推荐把 `SglangLiteDriver` 设计为 `LocalInferenceDriver` 家族的一个实现；`provider_type = "sglang-lite"` 可以保留为兼容别名，但新配置优先考虑 `provider_type = "local-inference"` + `backend_kind = "sglang-lite"`。
- vLLM 可作为同家族的 `backend_kind = "vllm"` 实现，复用 OpenAI-compatible HTTP 调用、request id 透传、prefix-cache metrics 聚合等逻辑。

### 2. 执行路径
- **推荐且当前阶段唯一路径**：通过 HTTP（或 gRPC）与 sglang-lite 交互。
  - sglang-lite 以独立进程/服务形式运行，对外暴露标准的 OpenAI 兼容接口。
  - UniGateway 通过其现有的 HTTP 传输层对接。
- **明确不考虑**：PyO3 直接嵌入或同进程 Python 库调用（见非目标部分）。
- 未来如果需要更高性能的集成方式，应通过 gRPC 等标准跨语言机制，而非语言级直接绑定。

### 3. 模型与端点配置
- 支持通过 `DriverEndpointContext.metadata` 或专用配置传入模型路径、设备、batch size 等。
- 支持本地 MoE backend 不强制 api_key。

### 4. 能力声明
- 声明 sglang-lite 特有能力（如 Radix prefix caching）。
- 通过 UniGateway capability 系统暴露。
- capability 命名应保持通用，例如 `prefix_cache = true`、`prefix_cache_metric = "usage.cache_hit_tokens"`、`moe_optimized = true`，不要把 routing/metrics 逻辑绑定到 `radix` 或 `sglang-lite` 名称。
- vLLM 的 Automatic Prefix Caching 也应映射到同一个 `prefix_cache` capability。

### 5. 流式、协议与错误
- 完整支持 chat completions streaming。
- 响应格式与 `ProxySession` 等抽象兼容。
- 错误映射为 `GatewayError`。
- 支持 request id 透传，提供 hook 供 metrics（TTFT、tokens/s、cache hit rate 等）。

### 6. Tool Calling
- 支持基本占位或透传（具体执行可上移）。

## 对 sglang-lite 库的期望

- 提供清晰稳定的 library API（推荐直接暴露 `LiteEngine` 或其构建块）。
- 支持简单初始化：`LiteEngine(model_name=..., device=...)`。
- 提供生成接口（支持流式）。
- `engine/` 不包含 HTTP server；仓库的 `control/` + `serving/` 提供独立服务所需的
  最小 HTTP/SSE、配置、health/readiness 和 metrics。
- 不包含认证、多后端 routing、配置中心、业务逻辑或宽 OpenAI 表面等网关能力。

## 实现建议（保持 UniGateway 通用性）

1. 严格使用现有 `ProviderDriver` 抽象，不要在核心引入 sglang-lite 专有接口。
2. sglang-lite 相关代码放在独立模块或子 crate（如 `drivers/sglang_lite.rs` 或 `unigateway-sglang-lite`）。
3. 通过 `DriverEndpointContext.metadata` 传递特有参数。
4. Driver ID 使用 `"sglang-lite"`。
5. 通过 feature flag 控制是否包含 sglang-lite 驱动，避免强制 Python 依赖。
6. **关键**：UniGateway 核心保持完全通用，sglang-lite 逻辑仅限 driver 内部。不要在 UniGateway 中选择 attention backend 或实现 MoE 特定调度。

## UniGateway 需要做的具体修改与优化

以下是 UniGateway 侧需要进行的修改和优化点，按优先级和模块分类：

### 1. 新增/扩展 Driver 支持（P0）
- 在 `unigateway-core/src/protocol/` 下新增 `sglang_lite/` 模块（或直接在 drivers 下）。
- 实现 `SglangLiteDriver` 结构体，实现 `ProviderDriver` trait。
- 目前可复用 `OpenAiCompatibleDriver` 作为基础（因为 sglang-lite 暴露 OpenAI 兼容接口），但使用独立的 `ProviderKind::SglangLite`。
- 以 HTTP 传输为主（调用 sglang-lite 暴露的 `/v1/chat/completions`）。
- 注册机制：在 `with_builtin_drivers` 或 registry 中根据 feature "sglang-lite" 注册该 driver。
- **注意**：不得规划或实现 PyO3 / direct embed 相关逻辑。

### 2. 配置系统扩展（P0）
- 在 `unigateway-config` 中支持 `provider_type = "sglang-lite"`。
- 允许本地 backend 的 `api_key` 为空（当前已有特殊处理）。
- 在 `DriverEndpointContext.metadata` 中支持 sglang-lite 特有字段，例如：
  - `model_path`: 本地模型路径
  - `device`: "cuda" / "cpu"
  - `max_batch_size`
- 扩展配置 schema 和 core_sync 逻辑，允许 sglang-lite 作为 local provider。
- **注意**：不得包含任何与 PyO3 或进程内嵌入相关的配置字段。

### 3. 能力与协议扩展（P0/P1）
- 在 `capabilities.rs` 中为 `ProviderKind::SglangLite` 添加默认能力：
  - `LocalInferenceCapabilities::sglang_lite_default()`（突出 Radix prefix caching）。
- 在协议处理中，针对 `SglangLite` 做特殊标记（例如 responses render、targeting）。
- 支持 prefix caching 相关报告和路由优化（这是 sglang-lite 的核心优势）。

### 4. 集成与调用优化（当前阶段）

**推荐方式**：
- 以 HTTP 模式为主（当前已支持）。
- 后续可考虑引入 gRPC / 子进程模式。
- sglang-lite 以独立进程/服务形式运行，对外暴露标准的 OpenAI 兼容接口。

**明确边界（重要）**：
- **严禁使用 PyO3 或任何直接 Python 库嵌入方式**。
- UniGateway 不得在进程内直接加载或调用 sglang-lite 的 Python 代码。
- 所有交互必须通过标准网络传输（HTTP 或 gRPC）。
- 此边界已作为 Non-Goal 明确写入，请在实现时严格遵守。

### 5. 可观测性与指标优化（P1）
- 为 `SglangLite` 扩展特定 metrics（cache hit rate、expert utilization 如果暴露、TTFT 等）。
- 在 reporting 和 hooks 中区分 `SglangLite` provider。
- 允许 driver 暴露内部指标给 unigateway 聚合。

### 6. 测试与示例
- 增加 `sglang-lite` 相关的集成测试（类似当前 `build_core_pool_for_sglang_lite_provider`）。
- 在 unigateway-sdk/examples 中增加 sglang-lite 使用示例。
- 更新文档（docs/guide/Provider 示例）说明如何配置 sglang-lite backend。

### 7. 进一步优化方向（非必须，但建议）
- 设计通用的 "LocalInference" driver 基类或 trait，让 sglang-lite、vLLM-local 等本地引擎共享 OpenAI-compatible 调用、request id、健康检查、prefix-cache metrics 等逻辑，同时保持各自特殊能力。
- 当 sglang-lite 暴露更丰富的服务协议时，driver 可以使用通用 capability 或内部
  gRPC endpoint，但不能直接操作其 Python building blocks。
- 性能优化：通过 gRPC 等高效传输方式和连接复用，而非语言级直接嵌入；请求合并和
  KV/scheduler 决策必须由 sglang-lite engine 自己完成。
- 保持松耦合：即使 sglang-lite 演进，UniGateway 也能通过 driver 适配，而不影响其他 provider。

## 修改影响范围

- **unigateway-core**：新增 driver、扩展 pool、capabilities、protocol 处理。
- **unigateway-config**：支持新 provider type 和 metadata。
- **unigateway-host**：可能需要支持本地引擎的特殊 targeting 和 responses 处理。
- **unigateway-sdk**：暴露 sglang-lite feature，更新示例。
- **测试与文档**：大量新增针对 sglang-lite 的测试和指南。

这些修改应作为独立 feature（例如 feature = "sglang-lite"）开发，避免影响 UniGateway 对其他后端的通用支持。

## 优先级（已与 sglang-lite 团队协商）

**P0（已基本完成）**：HTTP 模式 chat completions（streaming + non-streaming）
- 通过 OpenAI 兼容接口对接。
- 独立 `ProviderKind::SglangLite` + 特殊处理（空 api_key、本地 backend）。

**P1**：通过 metadata / 配置支持 sglang-lite 特有参数
- 模型路径、device、batch size 等。
- 配置 `provider_type = "sglang-lite"`。

**P2**：gRPC / 子进程模式

**P3**：cache hit rate 等 sglang-lite 特定 metrics 透传、tool calling 占位。

**未来（不在当前需求范围内）**：
- 任何形式的 PyO3 直接 Python 库嵌入或同进程调用，均明确不作为需求提出（详见“非目标”部分）。

## 验收标准

- 不使用 UniGateway 时，OpenAI SDK 可直接调用 sglang-lite standalone 服务。
- 可通过 UniGateway 配置 sglang-lite 后端（HTTP 模式），用标准 OpenAI 客户端调用 MoE 模型。
- 前缀共享（Radix）效果在 UniGateway 层面可见（cache hit 相关指标）。
- UniGateway 其他后端行为不受影响。
- sglang-lite 相关逻辑未污染核心抽象。
- 代码边界清晰，未引入 PyO3 或任何直接嵌入方式。

---

**建议**：本需求应作为 UniGateway 独立 feature 开发（使用 feature = "sglang-lite"），保持与 sglang-lite 仓库的松耦合。当前及可预见阶段均以 HTTP/gRPC 模式为准，严禁引入 PyO3 或直接嵌入。

---

**与 UniGateway 团队的沟通要点（供参考）**：

我们完全认同你们对构建复杂度、API 稳定性、以及 UniGateway “通用可嵌入 SDK” 定位的担忧。

边界已明确写入“非目标”：
- 所有交互必须通过 HTTP 或 gRPC。
- 严禁引入 PyO3 / 直接 Python 嵌入。

后续我们不会再把 PyO3 作为需求提出。当前重点是把 HTTP 模式 + metadata 配置做好，同时保持 UniGateway 的通用性。

如果你们有其他顾虑或对 driver 实现方式的建议，欢迎直接讨论。
