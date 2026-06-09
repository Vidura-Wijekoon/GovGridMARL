"""Safety constraint definitions and cost signals (Phase 2, m5-t1/m5-t2).

Constraints: bus-voltage limits, no-blackout, household comfort bounds.
Each constraint emits a non-negative cost consumed by the Lagrangian agent.
Phase-2 milestone: zero safety violations on IEEE-14.
"""

from __future__ import annotations


def voltage_violation_cost(voltages, v_min: float = 0.95, v_max: float = 1.05):
    """Return per-bus cost for voltages outside the [v_min, v_max] p.u. band."""
    raise NotImplementedError("voltage_violation_cost — scheduled for Month 5")
