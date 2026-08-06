//! Pin decision helpers used by the scheduler tick.

use traject_core::{PinInfo, PinReason, Trajectory};

use crate::priority::{PinAction, PinDecision, PinPolicy};

/// Decide pin when a Generate finishes as a tool call.
pub fn pin_for_tool_wait(
    traj: &Trajectory,
    now_ms: u64,
    policy: &PinPolicy,
    tool_p95_ms: Option<u64>,
) -> PinDecision {
    let ttl = policy.ttl_for_tool_latency(tool_p95_ms);
    PinDecision {
        trajectory_id: traj.id,
        action: PinAction::Pin {
            until_ms: now_ms.saturating_add(ttl),
            reason: PinReason::WaitingTool,
            strength: 2,
        },
    }
}

/// After a tool returns, keep prefix warm briefly for the next Generate (prefetch).
pub fn pin_for_prefetch(
    traj: &Trajectory,
    now_ms: u64,
    policy: &PinPolicy,
) -> PinDecision {
    // Prefetch window is a fraction of the base tool TTL (not full tool wait).
    let ttl = (policy.base_ttl_ms / 2).clamp(policy.min_ttl_ms, policy.max_ttl_ms);
    PinDecision {
        trajectory_id: traj.id,
        action: PinAction::Pin {
            until_ms: now_ms.saturating_add(ttl),
            reason: PinReason::Prefetch,
            strength: 1,
        },
    }
}

/// Under memory pressure, prefer unpinning late + low-share pins.
pub fn should_force_unpin(pin: &PinInfo, now_ms: u64, pressure_high: bool) -> bool {
    if !pressure_high {
        return false;
    }
    if !pin.is_pinned(now_ms) {
        return true;
    }
    pin.strength < 2
}

pub fn apply_pin(traj: &mut Trajectory, decision: &PinDecision) {
    if decision.trajectory_id != traj.id {
        return;
    }
    match &decision.action {
        PinAction::Pin {
            until_ms,
            reason,
            strength,
        } => {
            traj.pin = PinInfo::pin_until(*until_ms, *reason, *strength);
        }
        PinAction::Unpin => traj.pin.unpin(),
        PinAction::Keep(info) => traj.pin = info.clone(),
    }
}
