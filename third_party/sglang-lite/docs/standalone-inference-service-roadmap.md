# sglang-lite 独立流式推理服务补齐路线

## 1. 结论

sglang-lite 的目标运行方式是一个**独立推理后端**，不需要与 SGLang 或 vLLM 配合：

```text
OpenAI Client
  → sglang-lite-serving（Rust：协议、SSE、生命周期）
  → sglang-lite engine process（Python：统一 engine loop）
  → RadixKVCache + BatchingScheduler + MoEModelRunner
  → FlashInfer / Triton / CUDA
```

UniGateway 是可选上游：

```text
Client → UniGateway → (sglang-lite | vLLM | SGLang)
```

它负责多后端 routing、auth、全局 rate limit、租户策略和指标聚合，不负责
sglang-lite 内部的 scheduler、KV cache 或 model execution。FlashInfer 是 kernel
backend，也不是另一个必须配合运行的推理引擎。

## 2. “独立服务”与“完整网关”的边界

独立可用不等于复制 vLLM 或 UniGateway 的全部功能。

| 层 | sglang-lite 必须拥有 | 明确留给可选 gateway |
| --- | --- | --- |
| Engine | engine loop、KV 生命周期、continuous batching、model forward、sampling、token delta | 跨后端负载均衡 |
| Request lifecycle | engine-local queue、backpressure、cancel、timeout、OOM reject、graceful drain | 多租户配额、全局限流、重试/故障转移策略 |
| API | 最小 chat completions、真实 SSE、models、health、readiness、metrics、OpenAI error shape | Responses/audio、统一多供应商协议面 |
| Security/business | 本地部署所需的基础网络配置 | auth、billing、agent loop、tool execution、内容策略 |

因此 `engine/` 继续是纯库，`control/` 和 `serving/` 只形成一个单引擎、窄协议、
可部署的服务闭环。

## 3. 当前代码离目标的主要差距

当前仓库仍处于骨架阶段，不能把已有 endpoint 或 cache-hit 计数等同于真实可用：

1. `engine/runner.py` 仍包含 tiny stub fallback、dummy FlashInfer Q/K/V 和 greedy-only
   sampling；real model prefill 还会重跑完整 prompt。
2. `engine/kv_cache.py` 的 per-layer paged KV 写入、COW、完整 block 生命周期和 eviction
   仍未完成。
3. `engine/scheduler.py` 主要按 batch size 拼接 running/waiting 请求，尚未形成中央
   token-budget engine loop。
4. `examples/server.py` 使用全局锁串行请求，不能把多个 HTTP 请求送入同一个
   continuous batch。
5. Rust `serving/` 默认连接 `StubEngineClient`；转发到 Python 时的 streaming 是先
   blocking 生成，再按空格拆分，不是真实 token/delta stream。
6. model list 仍硬编码且包含不符合 MoE-only scope 的 dense 模型。

## 4. P0：形成第一个真实独立服务

### P0.1 一个真实 MoE golden path

- 选择一个可在 CI/开发机运行的小型 MoE 模型族。
- 模型加载失败必须显式失败，禁止静默退回 stub。
- 打通 tokenizer → prefill → decode → logits → sampler → detokenizer。
- greedy/fixed-seed 输出与 Transformers 参考路径一致。

**退出标准**：一条命令启动真实模型；短 prompt 的 token 序列通过 reference
correctness test。

### P0.2 真实 Radix/Paged KV

- 每层 K/V page 的分配、写入、读取和释放。
- block table、refcount、shared prefix ownership、fork/COW。
- 基于内存预算的 eviction 和确定性的 OOM rejection。
- `cache_hit_tokens` 只统计真正跳过模型 prefill 的 token。

**退出标准**：共享前缀请求减少实际 prefill token；取消、结束、eviction 后无 block
泄漏。

### P0.3 中央 continuous batching engine loop

- engine process 内部拥有长期运行的主循环。
- HTTP/gRPC handler 只提交请求和消费 delta，不直接调用整段 blocking generation。
- waiting/running 状态、prefill/decode 混合、token budget、max batch tokens。
- 基础 fairness、chunked prefill 或明确的长 prompt 上限。
- cancel、timeout、client disconnect 能从控制面传播到 sequence 和 KV cleanup。

**退出标准**：至少 32 个并发请求进入同一个调度循环；不存在全局生成锁；调度 trace
能证明一个 batch 包含多个 request。

### P0.4 基础 generation semantics

- `temperature`、`top_p`、`top_k`、`seed`。
- EOS、stop token/string、`max_tokens`。
- 正确 chat template、detokenization 和 Unicode-safe delta。
- unsupported 参数显式 4xx，不得静默改变语义。

**退出标准**：stream/non-stream 文本一致；stop/EOS/seed conformance tests 通过。

### P0.5 真实 Rust ↔ Python 流式协议

- 用稳定的 `GenerationRequest` / `TokenDelta` 协议替换 `StubEngineClient`。
- HTTP 或 gRPC 必须逐 token/delta 转发，禁止先生成全文再切词。
- bounded channel 提供 backpressure。
- request ID、finish reason、usage、engine error 完整透传。

**退出标准**：TTFT 在生成结束前可观测；断开客户端会触发后端 cancel；慢客户端不会
无限占用内存。

### P0.6 官方 standalone 入口

提供一个正式入口，例如：

```bash
sglang-lite serve \
  --model <supported-moe-model> \
  --device cuda \
  --port 8000
```

最小端点：

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /healthz`
- `GET /readyz`
- `GET /metrics`

服务必须具备 model load readiness、queue limit、request timeout、max concurrency、
graceful drain、结构化日志和稳定错误格式。

**退出标准**：OpenAI SDK 可直接调用，不安装或启动 SGLang、vLLM、UniGateway。

## 5. P1：达到受控生产可用

1. **模型矩阵**：分别验证 Mixtral-style、Qwen-MoE、DeepSeek-style；未验证模型不进入
   `/v1/models`。
2. **数值矩阵**：先 BF16，再按真实需求增加 FP8/AWQ/GPTQ；每条路径都有 correctness
   和显存预算测试。
3. **性能**：FlashInfer 真正进入 prefill/decode hot path；CUDA graph decode；减少
   Python CPU overhead。
4. **单机并行**：优先支持一个受控 tensor parallel 拓扑；expert parallel 和多节点
   不是首版要求。
5. **可观测性**：TTFT、ITL、tokens/s、queue wait、running/waiting requests、
   batch tokens、KV 使用量、cache-hit tokens、OOM/cancel/error count。
6. **可靠性**：模型预热、健康/就绪分离、OOM soak、取消风暴、慢客户端、优雅升级。
7. **交付**：Docker、固定版本兼容矩阵、配置说明、operator guide、benchmark 报告。

## 6. P2：保持非目标

以下能力不是“成为独立推理服务”的前置条件：

- multimodal、audio、Responses API；
- structured output / grammar、agent runtime、tool execution；
- dynamic LoRA、speculative decoding、prefill/decode disaggregation；
- 完整 expert parallel、多节点 serving；
- vLLM 全 API、全模型、全硬件兼容。

需要这些功能时，应选择上游 gateway/sidecar，或直接使用 vLLM/full SGLang。

## 7. 推荐实施顺序

```text
M1 真实单模型 correctness
  → M2 真实 per-layer KV + prefix skip
  → M3 中央 async engine loop + continuous batching
  → M4 Rust/Python 真流式协议 + cancel/backpressure
  → M5 官方 standalone CLI + readiness/metrics/errors
  → M6 CUDA graph、量化、TP、模型矩阵和 soak benchmark
```

不要先扩 API 或模型数量。每个里程碑都必须建立在前一阶段的正确性和资源生命周期
测试之上。

## 8. 独立服务 Definition of Done

首个可发布版本必须同时满足：

- 不启动 SGLang、vLLM 或 UniGateway，也能用 OpenAI SDK 完成 stream/non-stream 请求。
- 至少一个真实 MoE 模型的输出通过参考正确性测试。
- 并发请求由同一个 continuous batching loop 调度，而不是 HTTP 层串行。
- prefix hit 对应真实 prefill 计算减少，且指标可验证。
- disconnect、timeout、EOS、stop、OOM 后 sequence 和 KV block 均被回收。
- `/readyz` 只在模型加载和预热完成后成功。
- 真实 token delta 在生成过程中到达客户端，不是完成后伪切分。
- 文档明确支持的模型、dtype、GPU 和限制，不宣称 vLLM 全量替代。
