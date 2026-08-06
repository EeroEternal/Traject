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
- `current_prefix`
- `memory` (scratchpad + slots)
- `priority` + `fairness_credit`
- `pin` (`PinInfo`)
- `history` / `active_step`

## Step kinds

- `Generate { delta, constraints, max_tokens }`
- `Tool { call, timeout }`
- `Control { Reflect | Plan | Branch | EarlyStop, payload }`
