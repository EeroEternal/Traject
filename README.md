# Traject

Agent-native Runtime：以 **Trajectory** 为调度单位，把 Agent 轨迹执行与 LLM 推理合并在同一系统。

## 当前进度

Phase 0 完成；Phase 1 已接上真实 GPU 推理（DeepSeek-V4-Flash @ 8×5090）。

已具备：
- Trajectory 状态机、统一调度、前缀树、Tool pin、多轨迹
- **sglang-lite 原生引擎后端**（`:9001` `/v1/generate`）
- OpenAI 兼容桥（`:8000` 或任意 `/v1/chat/completions`）
- `LocalEngineHandle`：可由 Traject 拉起 torchrun 引擎进程
- **进程内 Kernel**：`KernelBackend` + CPU ref + FlashInfer（`--features flashinfer --flashinfer`）
- `GpuCapabilities` 探测

## 远端真实推理

```bash
# A) 完整 MoE 模型（sglang-lite 引擎进程）
bash scripts/start_engine.sh
cargo run -p traject-cli --release -- \
  --engine-url http://127.0.0.1:9001 \
  --model /home/bodesi/models/ds-v4-flash \
  --max-tokens 48 \
  "用一句话介绍你自己"

# B) 进程内 FlashInfer attention smoke（同进程 CUDA kernel）
source /home/bodesi/sglang-dflash-venv/bin/activate
export PYO3_PYTHON=/home/bodesi/sglang-dflash-venv/bin/python
cargo run -p traject-cli --release --features flashinfer -- --flashinfer --max-tokens 4 "hi"
```

## Zene agent（完整 coding agent + 本地 GPU）

```bash
# 引擎 :9001 + 控制面 :8000 已就绪时：
cargo run -p traject-cli --release -- agent \
  --backend-url http://127.0.0.1:8000/v1 \
  --model /home/bodesi/models/ds-v4-flash \
  --workdir /tmp/traject-zene-demo \
  --max-turns 6 \
  "列出当前目录下的文件，用 Glob 工具，然后简短总结"
```

Zene 通过 path 依赖引入（`../zene`），`traject-zene` 把每次 `prompt` 记为一条 Trajectory。
