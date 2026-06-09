"""Constrained RL via Lagrangian and PID-Lagrangian methods (Phase 2, m5).

Wraps a base policy with dual variables (Lagrange multipliers) that price
constraint violations (voltage limits, comfort bounds, no-blackout). The
PID variant stabilises multiplier dynamics (task m5-t6).
"""

from __future__ import annotations


class LagrangianController:  # pragma: no cover - stub
    """Updates Lagrange multipliers from observed constraint costs. TODO(m5-t3)."""

    def __init__(self, n_constraints: int, lr: float = 1e-3) -> None:
        self.multipliers = [0.0] * n_constraints
        self.lr = lr

    def update(self, costs):
        raise NotImplementedError("LagrangianController.update — Month 5")
