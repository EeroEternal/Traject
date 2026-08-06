# vLLM 定位与兼容评估

本文回答一个边界问题：sglang-lite 目前明显参考了 SGLang 的 RadixAttention 路线，但是否也应该考虑 vLLM 的定位与兼容性？

结论分两层：

1. **之前从 SGLang 提炼的三个核心能力，在 vLLM 中同样成立。**
2. **需要兼容 vLLM 的后端定位和协议生态，但不追求兼容 vLLM 的完整功能面或内部实现。**

sglang-lite 应该被定义为 `local-inference backend` 家族中的一个 MoE 优化后端；vLLM 是同一层级的通用后端。UniGateway 面向两者暴露统一的 provider/driver/capability 抽象，而不是把 UniGateway 的核心抽象绑定到 SGLang 或 sglang-lite。

## 1. 三个核心能力是否同样适用于 vLLM

**是，完全适用于能力层；不等于内部实现可以直接互换。**

之前基于 SGLang 总结的核心，不应该命名为 SGLang 专属组件，而应该先定义成三类引擎通用职责：

| 通用核心能力 | sglang-lite / SGLang 路线 | vLLM 路线 | 共同语义 |
| --- | --- | --- | --- |
| KV cache management + prefix reuse | RadixKVCache / RadixAttention | KVCacheManager + APC + PagedAttention blocks | KV 分配、block/page 生命周期、引用/释放、淘汰、前缀命中 |
| Continuous scheduling | BatchingScheduler | vLLM V1 Scheduler | waiting/running 状态、token budget、prefill/decode 混合、batch formation |
| Model execution | MoEModelRunner + KernelBackend/CUDA graph | GPUModelRunner / Worker + CUDA graph/compile | model forward、sampling、KV 写入、kernel/backend 选择、MoE routing |

因此原来的三个核心组件可以保留为 **sglang-lite 的实现名称**，但架构层应该使用：

```text
KVCacheManager / PrefixCache / BlockKVCache
ContinuousScheduler
ModelExecutor
```

需要区分：

- **能力相同**：两者都必须解决 KV 生命周期、持续调度和模型执行。
- **机制不同**：SGLang/sglang-lite 偏 Radix tree；vLLM 偏 block table + hash-based APC。
- **内部接口不同**：cache key、block ownership、scheduler state、runner input 都不能假设可直接互换。
- **外部可统一**：健康检查、请求/流式协议、capability、request id、cache-hit metrics 可以由 UniGateway 统一。

所以正确的架构不是“把 vLLM 再接进一个 SGLang 模型”，而是：

```text
                       LocalInferenceBackend
                               │
            ┌──────────────────┴──────────────────┐
            │                                     │
       sglang-lite                              vLLM
  RadixKVCache / Scheduler / Runner    KVCacheManager / Scheduler / Runner
            │                                     │
            └──── shared protocol + capabilities ─┘
```

## 2. sglang-lite + FlashInfer 是否能替代完整 vLLM

**不能直接得出这个结论。能力同构不等于产品等价。**

FlashInfer 的定位是 GPU kernel/backend library，主要覆盖 attention、paged KV、sampling、MoE 等算子与执行原语。它不能单独提供：

- request/sequence scheduler 与抢占、公平性、chunked prefill 策略；
- KV block 生命周期、prefix cache policy、内存预算和 OOM 恢复；
- model registry、权重加载、量化兼容和 tokenizer/preprocessor；
- tensor/pipeline/expert parallel 与多节点 worker 管理；
- OpenAI API、streaming、错误兼容、部署和可观测性；
- multimodal、LoRA、speculative decoding、pooling/embeddings 等产品能力。

组合后的分层是：

| 层 | 组件 | 负责内容 |
| --- | --- | --- |
| Optional gateway | UniGateway | 多后端 routing、auth、全局 policy、metrics 聚合 |
| Standalone control/serving | sglang-lite control/serving | 最小 OpenAI API、真实 streaming、单引擎 request lifecycle |
| Engine core | sglang-lite | KV/cache policy、continuous scheduling、MoE model execution |
| Kernel backend | FlashInfer/Triton/CUDA | attention、paged KV、sampling、MoE kernels |

这套组合可以在明确边界内成为 vLLM 的替代方案：

- 主流 MoE 模型；
- 单机或受控的并行拓扑；
- prefix-heavy chat/agent workload；
- 固定且已验证的 dtype/quantization/kernel 组合；
- 只要求最小 OpenAI chat/streaming surface。

它不是 vLLM 的通用 drop-in replacement，除非继续补齐 vLLM 已经承担的广泛模型、硬件、分布式和高级功能矩阵。那样做会把 sglang-lite 重新建设成另一个通用大引擎，并违背 `lite` 的 scope。

因此应明确两个目标：

```text
目标 A：在选定生产 workload 中替代 vLLM     -> 是，sglang-lite 的合理目标
目标 B：覆盖 vLLM 的全部场景和功能矩阵       -> 否，不是 sglang-lite 的目标
```

## 3. vLLM 的相关定位

vLLM V1 的公开设计重点包括：

- OpenAI-compatible HTTP server：`/v1/chat/completions`、`/v1/completions`、`/v1/responses`、`/v1/embeddings` 等更宽的 OpenAI-like 表面。
- EngineCore：将 tokenization、multimodal preprocessing、detokenization、streaming 等 CPU-heavy 工作与核心 scheduler/model executor loop 解耦。
- Scheduler：统一调度 prompt tokens 和 output tokens，用 `{request_id: num_tokens}` 表示每步 token budget；支持 chunked prefill、prefix caching、speculative decoding 等。
- KVCacheManager / PagedAttention：以 block table 管理 KV cache。
- Automatic Prefix Caching：使用 block hash（parent hash + block tokens + extra hashes）缓存完整 KV blocks，面向多轮对话、重复长上下文。
- GPUModelRunner / Worker：负责 model forward、sampler、KV cache buffer、CUDA graph/compile 等执行细节。

参考：

- vLLM V1 guide: https://docs.vllm.ai/en/stable/usage/v1_guide.html
- vLLM OpenAI-compatible server: https://docs.vllm.ai/en/stable/serving/online_serving/openai_compatible_server/
- vLLM automatic prefix caching: https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/
- vLLM V1 architecture blog: https://blog.vllm.ai/2025/01/27/v1-alpha-release.html

## 4. sglang-lite 与 vLLM 的定位差异

| 维度 | vLLM | sglang-lite |
| --- | --- | --- |
| 产品定位 | 通用 LLM/VLM serving engine + OpenAI-compatible server | 独立 MoE inference backend；pure engine core + thin standalone service |
| 模型范围 | Dense、MoE、multimodal、pooling、audio 等广泛模型 | 只支持主流 MoE；dense/multimodal 不进 core |
| API 面 | 宽 OpenAI-like 表面 + vLLM extra params | 独立提供最小 chat completions / streaming / models / health；宽 API 和多后端能力可由 UniGateway 承担 |
| Prefix cache | Hash-based APC over full blocks | RadixKVCache 默认；可暴露 generic prefix-cache/block-KV capability |
| KV cache | PagedAttention / KVCacheManager | RadixKVCache + allocator；可兼容 block table / paged KV 语义 |
| Scheduler | 统一 token budget，支持多功能组合 | MoE-aware continuous batching；优先保持可理解与低复杂度 |
| 高级功能 | structured output、LoRA、spec decode、disagg、multimodal 等 | core 明确不做；上移 gateway 或独立 sidecar |

因此，sglang-lite 不应该成为 “mini vLLM”，也不应该只做 “mini SGLang”。更准确的定位是：

> 一个可独立运行、MoE-only、prefix-heavy workloads 优先的 local inference backend；
> 与 vLLM 共享 OpenAI-compatible 协议、provider capability 和观测指标抽象，
> UniGateway 可选。

## 5. 兼容层级

### P0：协议兼容

sglang-lite 应保持与 vLLM local serving 最小交集兼容：

- `POST /v1/chat/completions`
- streaming SSE chunk shape
- `GET /v1/models`
- health endpoint
- OpenAI-compatible error shape
- request id 透传（例如 `X-Request-Id` / response metadata）
- `usage.prompt_tokens`、`usage.completion_tokens`、`usage.total_tokens`
- 可选扩展：`usage.cache_hit_tokens`

对于 vLLM 的 extra parameters，sglang-lite 应采用稳定策略：

- 已明确支持的 sampling 字段正常接受。
- scope 外字段要么显式 4xx reject，要么 warning-ignore；不要无声改变语义。
- `structured_outputs`、multimodal content、LoRA、spec decode、disagg 等不进入 core。

### P1：Provider capability 兼容

UniGateway 不应只认识 `sglang-lite`，而应抽象出 generic local inference capability：

```toml
[[providers]]
name = "local-moe"
provider_type = "local-inference"
backend_kind = "sglang-lite"
base_url = "http://localhost:8000/v1"
api_key = ""

[providers.capabilities]
chat_completions = true
streaming = true
prefix_cache = true
prefix_cache_metric = "usage.cache_hit_tokens"
moe_optimized = true
multimodal = false
structured_output = false
```

vLLM 可以使用同一族抽象：

```toml
[[providers]]
name = "local-vllm"
provider_type = "local-inference"
backend_kind = "vllm"
base_url = "http://localhost:8000/v1"
api_key = "token-abc123"

[providers.capabilities]
chat_completions = true
responses_api = true
streaming = true
prefix_cache = true
vllm_extra_params = true
multimodal = true
```

`provider_type = "sglang-lite"` 可以作为兼容别名保留，但新设计不应把核心路由、pool、protocol 抽象写死为 sglang-lite 专用。

### P2：内部概念映射兼容

sglang-lite 内部仍可以使用 RadixKVCache，但对外不要暴露 RadixTree 作为 driver 的硬协议。建议对外使用通用术语：

| 通用概念 | sglang-lite 内部 | vLLM 对应 |
| --- | --- | --- |
| `PrefixCache` | RadixKVCache | Automatic Prefix Caching |
| `BlockKVCache` | KVAllocator / block table | KVCacheManager / PagedAttention blocks |
| `ContinuousScheduler` | BatchingScheduler | vLLM V1 Scheduler |
| `ModelExecutor` | MoEModelRunner | GPUModelRunner / Worker |
| `BackendCapabilities` | MoE-only + Radix metrics | broad model/API/hardware capability |

这允许 UniGateway 的 KV-affinity、prefix-cache-aware routing、metrics aggregation 同时兼容 sglang-lite 与 vLLM。

## 6. 可借鉴但不照搬的 vLLM 设计

### 值得吸收

- **统一 token budget 表达**：可以作为 `BatchFormer` 的策略输入，帮助混合 prefill/decode/chunked prefill，但不必立刻完整实现 vLLM 的 scheduler。
- **hash-based full-block 索引**：可作为 RadixKVCache 的可选二级索引，用于 block-level lookup、tenant salt、LoRA/multimodal hash 扩展；默认仍保留 Radix prefix tree。
- **persistent batch / diff update 思路**：适合未来降低 Python CPU overhead。
- **request id / metadata 透传**：便于 UniGateway 统一追踪 sglang-lite 与 vLLM。
- **capability 声明**：让 routing 不依赖 backend 名称。

### 不应吸收进 core

- vLLM 的完整 OpenAI server 表面。
- Responses/audio/embeddings/multimodal 等宽 API。
- vLLM-specific extra params 作为 sglang-lite 公共契约。
- Structured output、dynamic LoRA、spec decode、PD disaggregation。
- 直接复用 vLLM scheduler/KVCacheManager 内部实现，导致 lite 失去可理解性。

## 7. 对现有文档/实现的调整要求

1. `README.md`、`architecture.md`、`scope.md` 必须把外部抽象从 “SGLang/Radix 特化” 提升到 “local inference backend + generic prefix cache capability”。
2. UniGateway 需求中应明确：`SglangLiteDriver` 是 `LocalInferenceDriver` 家族的一个实现；vLLM 应作为同层级 backend 兼容。
3. `cache_hit_tokens` 是 generic prefix-cache metric，不应命名为 Radix-only metric。
4. sglang-lite 的 Paged/block KV 设计要兼容 vLLM-style block table 语义，但实现可继续保持 Radix-first。
5. 不要为追求 vLLM 兼容而扩大 core scope；不在 core 支持 dense/multimodal/structured output。

## 8. 选择建议

| 场景 | 优先选择 |
| --- | --- |
| 需要最广模型/硬件/API 兼容 | vLLM |
| 需要 VLM、多 API、多 LoRA、spec decode、disagg | vLLM 或 full SGLang |
| MoE-only、prefix-heavy chat/agent、希望极低运维复杂度 | sglang-lite |
| 需要 SGLang frontend language / structured generation runtime | full SGLang |
| Gateway 层统一多后端路由、auth、metrics、policy | UniGateway |

sglang-lite 的核心价值不是复制 vLLM，而是在更窄的 MoE + prefix-heavy 生产场景里
独立提供更小、更稳定、更容易调试的推理服务；UniGateway 只在需要统一多后端时使用。
