# DeepSeek V4-Flash 支持路线评审：先底座、后家族、最大化复用

状态：设计评审稿（不改变现有 scope 结论，细化执行顺序与复用边界）。
关联文档：[scope.md](scope.md)、[standalone-inference-service-roadmap.md](standalone-inference-service-roadmap.md)、[architecture.md](architecture.md)。

## 1. 结论

**不把 V4-Flash 当成"补齐模型列表"的入口，而是当成逼出真实执行底座的压力测试目标。**

顺序：先补完通用 MoE 执行底座（真 paged attention、中央 engine loop、KV layout
抽象），再以 DeepSeek-V2-Lite 作为 MLA/非标准 KV layout 的先导模型，最后才把
`deepseek_v4` 作为一个受控家族注册进 `engine/models.py`。

同时明确**自研 vs 复用**边界：sglang-lite 只自持三个核心（RadixKVCache、
BatchingScheduler、MoEModelRunner 的组合与 engine loop），其余尽可能以小模块方式
复用 FlashInfer、sgl-kernel、Triton kernel、HF/官方模型定义等成熟组件。

## 2. 现状差距（基于当前代码）

V4 难的不是在 registry 加一行 `deepseek_v4`，而是它把「权重如何切到 8 卡」
「FP4/FP8 如何计算」「CSA/HCA 的 KV 形态」「Radix 如何复用这些 page」绑在一起。
当前执行面离这个要求还有距离：

1. **执行面仍是 HF 前向包装**。`engine/runner.py` 用
   `AutoModelForCausalLM + device_map=auto` 单进程加载；paged KV 是"影子存储"：
   每步 `radix.build_cache()` 把 page 重建成 HF `DynamicCache` 再喂给模型
   （`_past_for_seq`），真相仍是 HF `past_key_values`。284B/13B-active 级别下，
   重建 cache 的显存与拷贝开销本身就不可行。
2. **FlashInfer 未进入 attention hot path**。prefill/decode wrapper 已创建但
   forward 仍走 HF SDPA；FlashInfer 只用于 paged append。
3. **KV 写入是逐 token Python 循环**（`kv_cache.py: write_kv`），且 page 布局
   硬编码为全层同构 `[layers, blocks, block_size, kv_heads, head_dim]`。
   MLA compressed KV（DeepSeek-V2/V3 起）已不满足该假设——这笔 layout 债在
   V2-Lite 阶段就会被逼出，不必等 V4。
4. **P0 golden path 未闭环**：prefill 分组长度不一致时逐条串行回退；
   roadmap 中 M2/M3（真 per-layer KV + 中央 async loop）尚未完成。

因此第一步不是"接 V4"，而是把 M2/M3 真正做完。

## 3. 自研 vs 复用边界（本文档的核心补充）

原则：**核心组合权自持，叶子实现最大化复用**。sglang-lite 的差异化在于
「Radix 前缀复用 × continuous batching × MoE 执行」这条 hot path 的组合方式，
不在于任何单个 kernel 或 loader。凡与这条 hot path 的"驱动/调度核心"无关的内容，
一律优先引用外部小模块，不自研。

| 层 | 策略 | 复用来源 | 说明 |
| --- | --- | --- | --- |
| Attention kernel（paged prefill/decode、MLA） | **直接引用** | FlashInfer（含 MLA/paged wrapper）、sgl-kernel | 收口到 KernelBackend 接口后整体替换，不自写 kernel |
| MoE kernel（grouped GEMM、fused expert、topk routing） | **直接引用** | FlashInfer / sgl-kernel / Triton 现成 kernel | 同上；expert 路由计算留在模型图内 |
| 量化（FP8/FP4 GEMM、KV 量化） | **直接引用** | sgl-kernel / DeepSeek 官方 kernel（DeepGEMM 类） | 量化布局归 loader，量化计算归 KernelBackend，不散落 |
| 权重加载与分片 | Hybrid | HF safetensors 加载器 + 自持的分片映射 | 只自写 "哪个 shard 放哪张卡" 的映射，不自写文件格式 |
| Tokenizer / chat template | **直接引用** | HF tokenizers | 已是现状 |
| 模型图定义（V2-Lite / V4） | Hybrid | transformers remote code 或 DeepSeek 官方 `inference/` | 可整体引用模型定义，只要求其 attention/MoE 调用点可被 KernelBackend 接管 |
| TP 通信原语 | **直接引用** | torch.distributed / NCCL | 只自持 "TP=8 单机" 这一种受控拓扑的初始化与 shape 约定 |
| RadixKVCache（page 生命周期、COW、eviction） | **重构（自持）** | — | 核心差异化，唯一需要扩展的是 per-layer KV layout 描述符 |
| BatchingScheduler / engine loop | **重构（自持）** | — | 核心差异化 |
| Runner 的 batch 组装与采样 | **重构（自持）** | 采样可复用 flashinfer sampling kernel | 组合逻辑自持，叶子 kernel 复用 |

**不做的自研**：不自写 attention/MoE/量化 kernel，不自写分布式框架，不复制
SGLang 的完整 ModelRunner/EPLB/spec-decode 体系。若需要从 SGLang 借用，只借
**叶子级小模块**（如 sgl-kernel 的单个算子），不引入其调度或内存管理框架——
否则就失去了 lite 的意义。

对应到 AGENTS.md 的分层纪律，执行面重构建议只拆两层，避免 Phase 0 出现四层抽象：

- **KernelBackend**：attention + MoE + 量化计算的唯一收口，内部由外部 kernel 组成；
- **ModelLoader**：分片与量化布局；
- Prefill/Decode 执行逻辑暂留在 `ModelRunner` 内（保持 2 跳可追踪）。

### 3.0 可复用叶子组件清单（GPU 实测已回填，2026-08-04 / 8×5090）

以下均为独立可 pip 安装 / vendor 的叶子件，不携带外部调度或全局状态
（接入时在 KernelBackend 内固定版本并做 capability 探测）。实测环境：
torch `2.11.0+cu130`、CUDA 13.0、GPU RTX 5090（sm_120 / capability 12.0）。
测法见 3.0.1，逐项记录见 3.0.2；复现脚本 `scripts/leaf_component_probe.py`。

| 组件 | 实测版本 | 提供能力 | 接入点 | sm_120 结论 |
| --- | --- | --- | --- | --- |
| `flashinfer-python` | **0.6.12** | paged prefill/decode、`BatchMLAPagedAttentionWrapper`、`append_paged_kv_cache`；SM100：`trtllm_batch_decode_sparse_mla_dsv4`；SM120 叶子：`B12xMoEWrapper` / `Sm120B12xBlockScaledDenseGemmKernel` | KernelBackend（paged 已接入；MLA/sparse 按 arch 路由） | **标准 MLA/paged：可用**；**SM100 sparse MLA 在 sm_120 必挂**；须改走 SM120 专用 sparse MLA（见 3.0.3） |
| `sgl-kernel` | **0.4.4** | `topk_softmax` / MoE 辅助、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_decode`、norm/rope | KernelBackend（MoE/量化，S2+） | **可安装可 import**；topk 路由 id 与 torch 一致；**无 CSA/HCA 公开符号**，有 dsv4 融合算子 |
| `deep-gemm` | **0.1.4** | DeepSeek FP8/FP4 grouped GEMM API 面齐全 | **暂不接入**（见下） | **本版本 sm_120 不支持**；需 SM120 内核版（vLLM 路线）或回退 B12x/sgl-kernel |
| DeepSeek 官方 `inference/` / transformers remote code | 本机有 `ds-v4-flash`；**无 V2-Lite 权重** | V4 模型图含 Indexer/Compressor/`sparse_attn`/HC(Sinkhorn) | ModelLoader / vendor（Hybrid） | V4 图可对照；**V2-Lite greedy/paged 验收 BLOCKED（缺权重）** |

**待实测确认（已实测 + 对照 vLLM SM120 路径后修正）**：

- flashinfer / sgl-kernel 对 V4-Flash 新 attention（官方代码表现为
  **Indexer + compressed KV + `sparse_attn`**，以及 **HC / `hc_split_sinkhorn`**；
  文档简称 CSA/HCA）：
  - 我们手测失败的是 **SM100 TRTLLM-GEN** 入口
    （`trtllm_batch_decode_sparse_mla_dsv4`）→ `Unsupported architecture`。
  - 这**不代表** SM120 不能跑 V4 sparse MLA，而是走错了架构族路径。
    vLLM 的做法是识别 `capability family 120` 后强制路由到
    FlashInfer **SM120 专用** sparse MLA（如 `FLASHINFER_MLA_SPARSE_SM120` /
    `sparse_mla_sm120`），**禁止**再走 SM100 TRTLLM。
  - 本机 flashinfer `0.6.12` 已暴露 SM120 叶子：`B12xMoEWrapper` /
    `b12x_fused_moe`、`gemm.Sm120B12xBlockScaledDenseGemmKernel`；
    标准 `BatchMLAPagedAttentionWrapper` 已在 sm_120 数值通过。
  - **SM120 sparse MLA 补测（2026-08-04）**：`0.6.12` 无
    `flashinfer.mla._sparse_mla_sm120`；隔离前缀安装
    `flashinfer-python==0.6.16.post1`（`FLASHINFER_DISABLE_VERSION_CHECK=1`，
    勿升级共享 venv——sglang 钉死 0.6.12 / cubin）后，公开 API
    `trtllm_batch_decode_sparse_mla_dsv4` **会路由到**
    `_trtllm_batch_decode_sparse_mla_dsv4_sm120`。冒烟成功条件：
    SWA/compressed KV 为 **packed uint8 last-dim 584**，并传
    `swa_topk_lens` + `extra_sparse_indices` / `extra_sparse_topk_lens`
    （bf16 512 会通过 Python 校验但内核仍要求 584）。复现：
    `FI_PREFIX=/tmp/fi1616 python scripts/try_flashinfer_016_sparse.py`。
  - sgl-kernel 提供 `dsv4_fused_*` 与 MLA 辅助，无 CSA/HCA 命名完整入口。
- deep-gemm 与 5090（sm_120）：**当前 `deep_gemm==0.1.4` 不支持**（bf16 入口即
  `Unsupported architecture`）。vLLM 侧通过更新 `support_deep_gemm` + 带
  SM120 内核的 DeepGEMM（或社区 fork）启用；我们在拿到 SM120 版之前
  **回退** `sgl-kernel` FP8 / FlashInfer B12x GEMM / cuBLASLt（本机
  `Sm120B12xBlockScaledDenseGemmKernel` 可 import/construct）。
- **B12xMoEWrapper**：小配置 construct + `run` 在 sm_120 **已通**（FP4 packed
  `[E,2I,H//2]` / `[E,H,I//2]` + 6D MMA scale）；与 bf16 SiLU MoE 的相对误差
  **尚未闭环**（`w1_alpha` / `fc2_input_scale` 约定未对齐，随机/错 scale 时
  输出有限但 rel ≫ 1）。数值对齐留待 S2 MoE 接入时做。
- **因此暂不把 `sgl-kernel` / `deep-gemm` 写入 `pyproject.toml` 可选依赖**
  （待 KernelBackend arch 路由落地与 fused_moe 数值对齐闭环后再加）。

#### 3.0.1 叶子组件实测计划（GPU 环境，逐项验收）

统一前提：8×5090（sm_120）、CUDA/torch 版本与仓库固定的依赖矩阵一致；
每项先做"能装能跑"再做"数值对齐"；任何失败都记录具体版本与报错，
不带病接入 KernelBackend。

**T1 flashinfer-python（attention 主路径，已部分接入）**

- 测法：现有 `tests/test_gpu_paged_attention.py` 为基线；补充
  MLA wrapper 冒烟（`BatchMLAPagedAttentionWrapper` 或等价接口），用
  DeepSeek-V2-Lite 的 head 配置构造随机 KV，对比 torch 朴素实现的输出。
- 验收：paged prefill/decode 数值与参考实现 atol≤1e-2（bf16）；MLA wrapper
  在 sm_120 上可编译可运行。
- 回退：MLA 不可用则暂时用"解压回标准 KV + 普通 paged attention"的慢路径，
  并记 issue 跟踪上游。

**T2 sgl-kernel（MoE 与量化算子，S2+ 接入）**

- 测法：`pip install sgl-kernel` 后跑最小算子冒烟：`fused_moe`/topk 用
  Mixtral 小配置对比 HF 参考 MoE 层输出；FP8 GEMM 对比 bf16 GEMM 的
  相对误差。
- 验收：sm_120 上可安装（有预编译 wheel 或可源码编译）；fused_moe 输出与
  HF 参考 rel-err≤1e-2；单算子调用不依赖 SGLang runtime（import 仅
  sgl_kernel）。
- 回退：装不上或算子缺失时，MoE 先走 HF 模型图内的朴素实现，仅损失性能。

**T3 deep-gemm（FP8 grouped GEMM，V4 expert 候选）**

- 测法：安装后跑其自带 benchmark/测试；重点确认 sm_120 支持（该库最初面向
  Hopper sm_90，Blackwell 支持需实测）；FP8 grouped GEMM 数值对比 bf16 参考。
- 验收：sm_120 可运行且数值达标；不达标则明确记录"5090 不支持"。
- 回退：用 sgl-kernel 的 FP8 GEMM 或 cuBLASLt 路径替代。

**T4 官方模型图（V2-Lite 先导 → V4）**

- 测法：transformers remote code 加载 DeepSeek-V2-Lite，短 prompt greedy
  输出与 HF 参考一致（现有 reference correctness 测试的 GPU 版）；确认其
  attention/MoE 调用点可被 KernelBackend monkeypatch 接管（同 PR #4 手法）。
- 验收：V2-Lite 走 paged 路径 `paged_rebuild_count==0` 且逐 token 对齐；
  MLA 的 KV 进 RadixCache 需要的 layout 描述符明确（这是 S2 的输入）。
- 回退：remote code 不可控时 vendor 模型定义进仓库（标注来源与 license）。

产出要求：每项一份简短记录（版本、命令、结果、结论），汇总回填本文档
3.0 表格的"待实测确认"栏；全部通过后才把 sgl-kernel / deep-gemm 写进
`pyproject.toml` 可选依赖。

#### 3.0.2 实测记录（2026-08-04，5090 / sm_120）

**T1 flashinfer-python — PASS（标准路径）/ FAIL（V4 sparse）**

- 版本：flashinfer `0.6.12`；torch `2.11.0+cu130`
- 命令：`pytest tests/test_gpu_paged_attention.py`；
  `PYTHONPATH=. python scripts/leaf_component_probe.py`；
  另手测 `flashinfer.mla.trtllm_batch_decode_sparse_mla_dsv4`
- 结果：
  - 既有 GPU paged 基线 **3 passed**（`paged_rebuild_count==0`）
  - `BatchMLAPagedAttentionWrapper`（fa2）在 sm_120 可跑；vs torch 参考
    `max_abs≈0.0078`（bf16，≤1e-2）
  - `is_sm12x_supported(cuda)=True`
  - `trtllm_batch_decode_sparse_mla_dsv4`（64 heads / dim 512）→
    `TllmGenFmhaRunner ... Unsupported architecture`
- 结论：S2 可接标准 MLA wrapper；V4 sparse MLA **禁止**在 sm_120 上调用
  SM100 TRTLLM-GEN 入口，必须按 3.0.3 做 arch family 路由。

**T2 sgl-kernel — PARTIAL PASS（可装可跑，MoE 数值对齐未闭环）**

- 版本：sgl-kernel `0.4.4`（import 仅 `sgl_kernel`，无 SGLang runtime）
- 命令：probe 脚本 + `topk_softmax` / `fused_add_rmsnorm` 手测
- 结果：
  - 安装/import OK；暴露 `moe.*`、`fp8_*_mm`、`dsv4_fused_*`、`cutlass_mla_*`
  - `topk_softmax`：top-k **id 集合 8/8 与 torch.topk(softmax) 一致**
    （renormalize 后权重尺度不同，属预期）
  - `fused_add_rmsnorm` 在 sm_120 正常
  - 公开 API **无 CSA/HCA 命名**；有 V4 相关 `dsv4_fused_q_norm_rope` 等
- 结论：可作为 S2+ MoE/量化叶子候选；完整 `fused_moe`↔HF Mixtral rel-err
  验收仍待补；**暂不写入 pyproject**。

**T3 deep-gemm — FAIL（5090 不支持）**

- 版本：deep-gemm / `deep_gemm` `0.1.4`；`get_cuda_arch()==12.0`
- 命令：`deep_gemm.bf16_gemm_nt(a,b,d)`（256² bf16）
- 结果：`Assertion error (.../gemm.hpp:436): Unsupported architecture`
  （FP8 API 面存在但未再测——bf16 入口已失败）
- 结论：**明确记录「deep-gemm 0.1.4 在 sm_120 不可用」**（非“永远不能用
  DeepGEMM”，而是缺 SM120 内核/能力探测）。V4 expert GEMM 近期待
  SM120 版 DeepGEMM 或回退 FlashInfer B12x / sgl-kernel FP8 / cuBLASLt；
  **不写入 pyproject**。

**T4 官方模型图 — BLOCKED（V2-Lite）/ 可对照（V4-Flash）**

- 本机：`~/models/ds-v4-flash`（含 `inference/model.py`）；
  `deepseek-ai/DeepSeek-V4-Flash`；**无 DeepSeek-V2-Lite 权重/缓存**
- V4 图要点（S2/S5 输入）：`Attention` = MLA + sliding window +
  可选 `Compressor`/`Indexer` + `sparse_attn`；Block 侧 HC
 （`hc_mult` + `hc_split_sinkhorn`）；expert `fp4` + attention 侧 FP8 量化配置
- S2 layout 描述符需求（在缺 V2-Lite 权重时仍成立）：
  - 标准：`[layers, blocks, block_size, kv_heads, head_dim]`
  - MLA compressed：`ckv [blocks, page_size, kv_lora_rank]` +
    `kpe [blocks, page_size, qk_rope_head_dim]`（常见 `page_size=1`）
  - V4 另需 SWA 窗口池 + compressed 池 + sparse index（与当前单一
    isomorphic page 布局不兼容）
- 结论：V2-Lite greedy / `paged_rebuild_count==0` **未跑**（缺权重）；
  取得 V2-Lite 后再补 T4 闭环。

#### 3.0.3 SM120 架构族与 KernelBackend 路由（对照 vLLM，禁止抄调度）

vLLM 已证明：消费级 Blackwell（RTX 5090 / sm_120）与数据中心 Blackwell
（B200 / sm_100）**不是同一二进制兼容族**。sglang-lite 只借鉴其**叶子级
capability 探测 + 内核路由**，不引入 vLLM 调度/内存管理框架。

**硬件差异（为何不能复用 SM100 路径）**

| 特性 | SM100（B200） | SM120（5090） |
| --- | --- | --- |
| Shared Memory | ~228 KB | ~99 KB |
| Tensor Memory (TMEM) | 有 | 无 |
| 主要 MMA | tcgen05 / UMMA | mma.sync（偏 Ampere 风格） |
| 集群/多播 | 支持 | 基本不支持（1×1×1） |

依赖 TMEM / tcgen05 的内核（大量 SM100 FlashInfer TRTLLM、部分 DeepGEMM）
在 SM120 上会直接 `Unsupported architecture`——这与 §3.0.2 手测一致。

**vLLM 做法（参考，非目标 scope）**

1. 设备能力：把 `is_device_capability_family(120)` 纳入支持列表（相关改动如
   vLLM `#41062` / `#41028`）。
2. Attention：新增 `FLASHINFER_MLA_SPARSE_SM120`；DeepSeek-V4 sparse MLA 在
   SM12x **强制**走 SM120 专用内核，**禁止** SM100 TRTLLM；prefill/decode
   分治并缩小 workspace（适配 32GB）。
3. MoE/GEMM：启用带 SM120 内核的 DeepGEMM；FlashInfer B12x / CUTLASS SM120
   跑 NVFP4/MXFP4 MoE（tile 限制在 ~99KB SMEM）；失败再 fallback Marlin/Triton。
4. 大 PR 参考：上游 `#43477`（DeepSeek V4 + SM120 基础）；社区更完整分支如
   jasl `#41834`（含更多 V4-Flash 细节）。**DSpark 投机解码仍为 sglang-lite
   scope 外**。

**本机 flashinfer SM120 叶子（2026-08-04 补测）**

- `0.6.12`：`B12xMoEWrapper` / `b12x_fused_moe`、
  `gemm.Sm120B12xBlockScaledDenseGemmKernel`、`mla.is_sm12x_supported==True`；
  **无** `_sparse_mla_sm120`（公开 dsv4 仍走 SM100 → 必挂）。
- `0.6.16.post1`（隔离 `/tmp/fi1616`）：存在
  `flashinfer.mla._sparse_mla_sm120`；公开
  `trtllm_batch_decode_sparse_mla_dsv4` **自动路由 SM120**；packed uint8 584
  + `swa_topk_lens` / `extra_sparse_*` 冒烟 **OK**（shape `(B,q,H,512)`，finite）。
- DeepGEMM `0.1.4`：`bf16_gemm_nt` → `Unsupported architecture`（维持回退）。

**sglang-lite KernelBackend 必须遵守的规则**

1. capability 声明包含 `arch_family ∈ {sm90, sm100, sm120, ...}`，以及
   `sparse_mla_backend` / `moe_gemm_backend` 枚举；runner **不得**写
   `if major == 10`。
2. SM120 上 V4 sparse MLA：必须确认走 **SM120 后端**（FI≥0.6.16 的公开
   dsv4 可接受，因其内部 dispatch 到 `_…_sm120`）；在仅有 0.6.12 时**禁止**
   调用会落到 SM100 TRTLLM 的 dsv4，应改官方 `sparse_attn` 或升 FI。
3. MoE：SM120 优先 FlashInfer B12x / sgl-kernel FP8；DeepGEMM 仅在
   `support_deep_gemm(sm120)` 探测通过后启用。
4. 复用边界不变：只借叶子 kernel 与路由表，不借 vLLM ModelRunner/Scheduler。

**S5 前增量验收（补测）**

- [x] SM120 sparse MLA 入口可 import + 随机 KV 冒烟（FI 0.6.16.post1 隔离前缀；atol 另定）
- [~] `B12xMoEWrapper` 小配置：`run` 已通 / vs HF MoE rel-err **未闭环**（scale 约定）
- [x] DeepGEMM SM120 版：无（0.1.4 FAIL）→ 维持 B12x/sgl-kernel 回退
- [x] KernelBackend 单测：伪造 sm100/sm120 capability 时路由表断言（`tests/test_capability_routing.py`）

### 3.1 硬件后端隔离（非 NVIDIA 卡的前置设计）

未来需要支持华为昇腾（Ascend/CANN）及其他非 NVIDIA GPU，因此硬件隔离边界必须在
KernelBackend 设计时一次划清，避免事后剥离 CUDA 假设：

- **KernelBackend 是唯一的硬件抽象点**。attention、MoE、量化、采样 kernel 与
  TP 通信原语全部收口在此；CUDA + FlashInfer/sgl-kernel 只是它的第一个实现
  （`CudaKernelBackend`），昇腾对应 `AscendKernelBackend`（torch_npu + CANN 算子/
  MindIE kernel），其他卡同理各自成模块。
- **KernelBackend 之外禁止硬件特有假设**：RadixKVCache、scheduler、loader、
  engine loop 只依赖 torch device 语义与 layout 描述符，不 import flashinfer、
  不写 `torch.cuda.*` 直调、不假定 NCCL（通信后端由 KernelBackend 声明，
  如 NCCL/HCCL）。现状中 runner 顶层 `import flashinfer` 与 CUDA-only 分支
  是要在 S1 重构中清掉的典型例子。
- **能力声明而非 if-else**：每个后端声明自己支持的 dtype（FP4/FP8 是否可用）、
  KV layout、CUDA graph 等价物等 capability；上层按声明降级（如昇腾初期
  BF16-only），不在核心路径散落设备判断。
- **优先级**：Phase 0/1 只实现和验证 CUDA 后端；昇腾等以"接口预留 + 不阻塞"的
  方式对待——即接口设计时用上述规则审查，但不提前实现，避免抽象空转。

## 4. 实施顺序

```text
S1  M2/M3 补完：paged KV 成为唯一真相（去掉 DynamicCache 重建），
    FlashInfer prefill/decode kernel 进入 hot path，中央 async engine loop。
    验证模型：fixture / Mixtral 小模型。
S2  KV layout 抽象（跳过 V2-Lite 先导，直接用 V4 形状）：
    RadixKVCache 增加 layout 描述符（标准 KV / MLA compressed / DSV4 packed），
    KernelBackend 收口 MLA + arch_family 路由（见 3.0.3）。
S3  单机 TP=8（受控拓扑，5090×8）：ModelLoader 分片 + 官方 convert 契约；
    EP 只做最小可用，不做 expert load-balancing 调度。
S4  数值路径：BF16 → FP8；量化计算全部走 KernelBackend（B12x / sgl-kernel）。
S5  注册 deepseek_v4 家族：tokenizer 直接引用；模型图 Hybrid
    （官方 inference/）；FP4 expert + FP8 attention 经 KernelBackend 门面；
    CSA/HCA 的 KV 形态以 S2 的 layout 描述符承接；
    KernelBackend 按 arch_family 路由 SM120 sparse MLA + B12x/MoE（见 3.0.3）。
    mHC、hash routing 等留在模型图内部，scheduler 不感知。
```

**决策（2026-08-05）**：跳过 DeepSeek-V2-Lite 先导（无权重且对 V4 终局价值低）。
叶子能力通过 FlashInfer / sgl-kernel / 官方 `inference/` **小包引用**接入，
禁止整段引入 vLLM/SGLang 调度。P0 脚本：`scripts/v4_official_smoke.sh`；
短生成：`scripts/v4_lite_short_gen.py`。

## 5. 明确不做或后置

与 scope.md 一致：

- DSpark/MTP 投机解码：scope 外，主模型稳定 decode 即可；
- 1M 上下文：后置，先保证常规长度正确性与吞吐；
- tool/reasoning parser：上移 gateway；
- 完整 EP、disagg、多节点：非目标。

## 6. 验收标准（窄口径）

8×5090 上：

1. 加载本地 `ds-v4-flash` 权重成功（TP=8）；
2. 短 prompt 生成正确，数值与官方参考实现 / vLLM 短输出对齐；
3. prefix 复用时 `cache_hit_tokens` 上升，且对应真实 prefill 计算减少；
4. standalone serving 能真实 SSE 流式输出。

任何一条达不到，说明底座还没好，不应继续堆 V4 特例。

### 6.1 P0 官方基线进度（2026-08-05 / 8×5090）

- **convert PASS**：`inference/convert.py --model-parallel 8` →
  `/tmp/ds-v4-mp8/model{0..7}-mp8.safetensors`。
- **generate PASS**：`torchrun --nproc-per-node=8 generate.py`，prompt
  `Hello`，greedy 输出：`Hello! How can I help you today`（max_new_tokens=8）。
- **环境要点**：
  1. `--config` 必须用 `inference/config.json`（ModelArgs 形），不是 HF config；
  2. TileLang JIT 只用系统 CUDA（`PATH=/usr/local/cuda/bin`，
     `CPATH=/usr/local/cuda/include`），禁止混入 pip `nvidia/cu13` 头；
  3. `fast_hadamard_transform`：需
     `pip install --no-build-isolation` + `TORCH_CUDA_ARCH_LIST=12.0` 从源码装。
- 脚本：`scripts/v4_official_smoke.sh`、`scripts/v4_lite_short_gen.py`。
- 引擎侧已落地：`arch_family` 路由、`KvLayout`、`deepseek_v4` 注册、
  `ModelLoader` TP=8 / Hybrid 入口（需 `WORLD_SIZE` +
  `SGLANG_LITE_DSV4_CONVERTED`）。

### 6.2 LiteEngine CB MVP（官方 forward 包装，2026-08-05）

- **范围**：`ModelRunner` 对 `_v4_hybrid` 调用官方
  `Transformer.forward(input_ids, start_pos)`；Scheduler continuous batching
  可出 token；KV 仍在官方 `Attention` 缓冲内；Radix prefix hit **禁用**。
- **加载顺序**：`ensure_tp_process_group()`（NCCL）必须在 `Transformer(...)`
  之前；入口脚本 `scripts/v4_lite_engine_gen.py`（`torchrun`，`start_loop=False`
  + `pump_until_idle`，全 rank 同步 forward）。
- **计时用例**（8×5090，`scripts/v4_lite_engine_gen.py`，prompt `Hello`，
  ignore_eos，warm tok/s）：

  | Case | LiteEngine warm | 官方基线 warm |
  | --- | --- | --- |
  | 1×128 | ~4.8–5.1 | ~5.0 |
  | 4×96 | ~18.3 | ~13.3 |
  | 1×256 | ~4.8 | ~4.9 |

  （仍走官方 `sparse_attn`；入口需在 import torch 前 remap
  `CUDA_VISIBLE_DEVICES=$LOCAL_RANK`，供 TileLang device_id=0。）
- **Prefix 复用（2026-08-05）**：`engine/v4_prefix_cache.py` 对官方
  `kv_cache` / `kv_state` / `score_state` 做 CPU 快照；admit 用最长精确前缀
  命中设 `cache_hit_tokens`/`cached_len`；exact hit 跳过 prefill forward，
  partial hit 只跑 suffix（`start_pos>0` 时按官方约束逐 token）。单测：
  `tests/test_v4_prefix_cache.py`。仍非 Radix 双池 COW。
- **Standalone SSE（2026-08-05）**：`python -m sglang_lite.process` 支持
  `torchrun` TP（rank0 HTTP NDJSON，其余 rank `broadcast_object_list` +
  `pump_until_idle`）；`sglang-lite-serving --tp 8` 或
  `scripts/v4_serve_sse.sh`；Rust SSE 输出 `chat.completion.chunk` + 最终
  `usage` + `data: [DONE]`。控制面单测：`control/tests/sse_stream_tests.rs`。
- **真机验收（2026-08-05 / 8×5090）**：
  - **Align**：`match_soft_top5=true`（`Hello`∈prefill top5；greedy 常翻成
    `你好`，logit 差≈0.09）。摘要 `/tmp/v4_align_summary.json`。
  - **SSE**：`torchrun -m sglang_lite.process` + `sglang-lite-serving
    --engine-url` → OpenAI `chat.completion.chunk` + `[DONE]`（需
    rustup stable≥1.85；系统 cargo 1.75 不够）。引擎侧需 **专用 CUDA
    线程**（asyncio executor 会触发 TileLang device mismatch）。
  - 脚本：`scripts/v4_remote_acceptance.sh`、`v4_align_tokens.py`、
    `v4_debug_first_token.py`。
- **KV 生命周期（Hybrid）**：`clear_v4_kv_slot` 在 cold prefill 前与请求
  `final` 时清官方 Attention/Compressor 行；prefix 仍靠 CPU 快照，非 Radix
  双池。
- **内核现状**：生产 decode 维持官方 `sparse_attn`；FI SM120 sparse MLA
  仍 blocked（§6.3），默认 `SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`。
- **数值稳态 / logits 门禁（2026-08-05）**：`scripts/v4_logits_compare.py`
  （`torchrun` TP=8，`SGLANG_LITE_V4_DISABLE_FI_SPARSE=1`，seed 与 align
  同源）。8×5090 / prompt `Hello`（5 tok）Lite prefill 末步：argmax=`你好`
  (27.574) vs top2 `Hello` (27.483)，**top2 Δ≈0.09**；`gate_soft=true`
  （英文 lead 仍在 top5）。摘要 `/tmp/v4_logits_compare.json`，可选
  `--official-logits` 填 `max_abs` / `argmax_match` / `top5_overlap`。
  全层逐层 hook 仍不做（官方 Transformer 未暴露 per-layer API）。CVD
  remap 导致与官方 `set_device(local_rank)` 贪心首 token 可翻牌，属已知软门槛。
- **增量 decode UTF-8（2026-08-05）**：`ModelRunner.detokenize_delta` 收紧
  （不完整多字节空 delta；发散改写不回吐整段；lone `�` 抑制）；
  `SchedulerLoop._apply_stop_and_limits` 全分支走 delta；NDJSON
  `ensure_ascii=True`。单测 `tests/test_detokenize_delta.py`。
- **下一阶段**：Radix compressed+SWA 双池 COW 真写；FI SM120 换核（等上游
  absmean≠0）。

### 6.3 SM120 sparse MLA decode 换核（2026-08-05）

- **入口**：`scripts/v4_lite_engine_gen.py` 在 import 前插入
  `SGLANG_LITE_FI_PREFIX`（默认 `/tmp/fi1616`）+
  `FLASHINFER_DISABLE_VERSION_CHECK=1`；共享 venv 仍可保留 FI **0.6.12**。
- **Pack**：`engine/dsv4_kv_pack.py` — bf16 KV → uint8 584
  （448 FP8 NoPE + 128 RoPE bf16 + 7 ue8m0 + pad）；优先用官方
  `kernel.act_quant`。
- **Hook**：`engine/v4_sparse_mla.py` `attach_v4_sparse_mla` 在 Hybrid load
  后替换 `model.sparse_attn` / `kernel.sparse_attn`；**仅 decode**
  （`q_len==1`）走 `KernelBackend.sparse_mla_decode_dsv4`；失败或 prefill
  回退官方 TileLang。
- **验收**：capability 为 `flashinfer_sparse_sm120`；hook 已 armed；同用例
  warm tok/s 对照 §6.2。
- **本机实况（2026-08-05）**：已在隔离前缀 `/tmp/fi1616_full` 对齐安装
  `flashinfer-python==0.6.16.post1` + `flashinfer-cubin==0.6.16.post1` +
  `flashinfer-jit-cache==0.6.16.post1+cu130`（含 `sparse_mla_sm120.so`）。
  probe（随机/真实 584 pack、H=8/64）仍 **absmean=0**（finite）。
  非版本错配问题；更像 SM120 dsv4 在本机 5090（sm_120）上的上游内核/
  调用约定未闭环。hook 零输出探测会回退官方 `sparse_attn`。
  远端直连 GitHub 超时，wheel 需本机下载后 `scp`。

## 7. 风险与开放问题

- V4-Flash 的 CSA/HCA、FP4 expert、mHC 细节以官方 `inference/` 参考实现为准，
  本文对其的描述属于需求输入，接入前需逐项核对；
- FlashInfer / DeepGEMM 的 **SM120 专用内核**进度是外部依赖：SM100 路径在
  5090 上不可用；KernelBackend 必须 capability 路由并允许回退官方
  `sparse_attn` / B12x / cuBLASLt（见 3.0.3）；
- 把 SM120 “当作 SM100” 是已验证的失败模式，禁止再出现；
- MLA / V4 sparse 的 Radix page 复用语义（compressed + SWA 双池、COW/fork）
  需要在 S2/S5 单独验证；
- vLLM `#43477` / 社区 `#41834` 仅作叶子路由参考，不扩大 sglang-lite 到
  DSpark/宽功能面。
