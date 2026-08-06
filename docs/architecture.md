# Traject Architecture

## Layers

Policy → Trajectory Manager → Unified Scheduler → Inference Core / Tool Runtime → Hierarchical Memory

## Core types

- `Trajectory` — first-class agent execution process
- `Step` — Generate | Tool | Control
- `PrefixNode` — logical share + physical blocks
- `PinInfo` — TTL pin for tool gaps

## Scheduler tick

1. Collect ready steps (active decode, post-tool generate, tools, new)
2. Sort by `SchedPriority`
3. Fill `token_budget` + `tool_concurrency_budget`
4. Emit chunked generate / async tool actions
5. Update prefix refs, pins, trajectory state

## Phase 0 boundaries

Inference backend is a trait (`InferenceBackend`). Memory is single-tier stub.
Pinning TTL is computed but eviction is not yet wired to real device memory.
