# Trajectory

## State machine

```
Created → Running ⇄ WaitingTool → Running → Finished
                ↘ Failed
                ↘ Suspended ⇄ Running
```

## Fields

- `id`, `tenant_id`
- `state`
- `current_prefix` → `MemoryManager` leaf (`engine_handle` aligned with engine radix)
- `memory` (scratchpad + slots; includes cache-hit counters)
- `priority` + `fairness_credit`
- `pin` (`PinInfo`)
- `history` / `active_step`

## Step kinds

- `Generate { delta, constraints, max_tokens }`
- `Tool { call, timeout }`
- `Control { Reflect | Plan | Branch | EarlyStop, payload }`

## External agent path (Zene)

1. `create_external_trajectory` — policy does not auto-advance
2. Each Zene LLM turn → `Driver::run_generate_step` (Scheduler + engine)
3. Each tool result → `Driver::run_external_tool_step` (pin → record → unpin)
4. `finish_trajectory` when the agent loop ends
