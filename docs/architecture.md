# Traject Architecture

## Layers

```text
Policy / Zene Agent
        ↓
Trajectory Manager
        ↓
Unified Scheduler  ←── pin / fairness / token+tool budgets
        ↓
Inference Core / Tool Runtime
        ↓
Hierarchical Memory (PrefixTree + engine handles)
```

## Core types

- `Trajectory` — first-class agent execution process
- `Step` — Generate | Tool | Control
- `PrefixNode` — logical radix share + `engine_handle` for sglang-lite KV/radix
- `PinInfo` — TTL pin for tool gaps (protects prefix from eviction)

## Drive modes

1. **Policy-driven** — `Driver::run_until_finished` (ReAct / Plan-Execute)
2. **External (Zene)** — `Driver::run_generate_step` / `run_external_tool_step`  
   Zene owns agent semantics; every step still goes through Scheduler + MemoryManager + InferenceEngine.

## Scheduler tick

1. Collect ready steps (active decode, post-tool generate, tools, new)
2. Sort by `SchedPriority`
3. Fill `token_budget` + `tool_concurrency_budget`
4. Emit chunked generate / async tool actions
5. Update prefix refs, pins, trajectory state; record `cache_hit_tokens`

## Memory ↔ engine alignment

| Traject | Engine (sglang-lite) |
|---------|----------------------|
| `session_id` | request `session_id` |
| `engine_prefix_hint` / `PrefixNode.engine_handle` | request `prefix_id` |
| `trajectory_id` + `step_id` | request fields + logging |
| `note_cache_hit` | response `usage.cache_hit_tokens` |
| pin / ref / eviction | logical; physical free when node evicted |

Physical MoE weights + paged KV still live in the engine subprocess today.
Full in-process ownership is a later phase (see [kernels.md](kernels.md)).
