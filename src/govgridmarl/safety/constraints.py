"""Safety constraint definitions and cost signals (Phase 2, tasks m5-t1 / m5-t2).

Formal statement: ``docs/design/safety_constraints.md``. Every constraint is a
non-negative *cost* ``c_k(s_t) >= 0`` with ``c_k = 0`` exactly when the
constraint is satisfied, so that a constrained learner can require
``E[sum_t c_k] <= d_k`` (``d_k = 0`` for the Phase-2 "zero violations" gate).

=========  =====================================================  ========
name       cost                                                   scope
=========  =====================================================  ========
voltage    band violation of the extreme bus voltages (pu)        shared
loading    ``max(0, max branch loading - 1)``                     shared
comfort    ``max(0, B_i / F_i - theta_g)`` (own daily burden)     per agent
blackout   power-flow divergence -> voltage 1.0, loading 2.0      shared
fairness   ``max(0, tau_f - Jain(B_t)) / T``  (Jain >= 0.85)      shared
=========  =====================================================  ========

The environment (``GridMARLEnv.step``) calls these functions and exposes the
vector in ``info[agent]["cost_vec"]``; the Lagrangian agents
(``govgridmarl.agents.lagrangian``) pick the constraints they enforce by
name. The fairness entry is an *objective* constraint, not a safety one: it
is never counted in ``violation`` / ``violation_rate``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

SAFETY_CONSTRAINTS: tuple[str, ...] = ("voltage", "loading", "comfort")
ALL_CONSTRAINTS: tuple[str, ...] = SAFETY_CONSTRAINTS + ("fairness",)

# Blackout proxy (power-flow divergence), as in docs/design/reward_and_constraints.md
BLACKOUT_VOLTAGE_COST = 1.0
BLACKOUT_LOADING_COST = 2.0


@dataclass(frozen=True)
class ConstraintSpec:
    """One constraint: its name, episode budget ``limit`` (0 = hard) and bounds."""

    name: str
    limit: float = 0.0
    description: str = ""


DEFAULT_SPECS: dict[str, ConstraintSpec] = {
    "voltage": ConstraintSpec("voltage", 0.0, "all bus voltages within [v_min, v_max] pu"),
    "loading": ConstraintSpec("loading", 0.0, "no branch above its thermal rating"),
    "comfort": ConstraintSpec("comfort", 0.0, "daily curtailment within the group comfort bound"),
    "fairness": ConstraintSpec("fairness", 0.0, "Jain index of kWh burden >= 0.85"),
}


def voltage_violation_cost(
    voltages: Sequence[float], v_min: float = 0.95, v_max: float = 1.10, per_bus: bool = False
):
    """Band-violation cost of bus voltages (pu).

    Default (``per_bus=False``): ``max(0, v_min - min V) + max(0, max V - v_max)`` -
    the scalar used by the environment. ``per_bus=True`` returns the vector of
    per-bus violations instead (for diagnostics / IEEE-118).
    """
    v = np.asarray(voltages, dtype=float)
    if v.size == 0:
        return np.zeros(0) if per_bus else 0.0
    if per_bus:
        return np.maximum(0.0, v_min - v) + np.maximum(0.0, v - v_max)
    return float(max(0.0, v_min - v.min()) + max(0.0, v.max() - v_max))


def loading_cost(loadings: Sequence[float], max_loading: float = 1.0) -> float:
    """``max(0, max branch loading - max_loading)`` (loading as a fraction of rating)."""
    ld = np.asarray(loadings, dtype=float)
    if ld.size == 0:
        return 0.0
    return float(max(0.0, ld.max() - max_loading))


def comfort_cost(
    burden: Sequence[float], flex_energy: Sequence[float], tolerance: Sequence[float] | float
) -> np.ndarray:
    """Per-household ``max(0, B_i / F_i - theta_i)``."""
    b = np.asarray(burden, dtype=float)
    f = np.asarray(flex_energy, dtype=float)
    frac = b / np.maximum(f, 1e-9)
    return np.maximum(0.0, frac - np.asarray(tolerance, dtype=float))


def blackout_cost(converged: bool) -> tuple[float, float]:
    """(voltage_cost, loading_cost) to apply when the power flow diverged."""
    return (0.0, 0.0) if converged else (BLACKOUT_VOLTAGE_COST, BLACKOUT_LOADING_COST)


def fairness_cost(jain: float, target: float = 0.85, horizon: int = 24) -> float:
    """Per-step shortfall of the Jain index against the target, spread over the horizon."""
    return float(max(0.0, target - jain) / max(1, horizon))


class ConstraintSet:
    """Named subset of constraints with episode budgets (used by the Lagrangian agents)."""

    def __init__(self, names: Sequence[str] = SAFETY_CONSTRAINTS, limits=None):
        self.names = tuple(str(n) for n in names)
        unknown = set(self.names) - set(ALL_CONSTRAINTS)
        if unknown:
            raise ValueError(
                f"unknown constraints {sorted(unknown)}; choose from {ALL_CONSTRAINTS}"
            )
        lim = dict(limits or {})
        self.limits = np.asarray(
            [float(lim.get(n, DEFAULT_SPECS[n].limit)) for n in self.names], dtype=float
        )

    def __len__(self) -> int:
        return len(self.names)

    def vector(self, cost_vec: dict) -> np.ndarray:
        """Pick this set's costs out of an ``info["cost_vec"]`` dictionary."""
        return np.asarray([float(cost_vec[n]) for n in self.names], dtype=np.float32)

    @property
    def is_safety_only(self) -> bool:
        return all(n in SAFETY_CONSTRAINTS for n in self.names)


__all__ = [
    "SAFETY_CONSTRAINTS",
    "ALL_CONSTRAINTS",
    "ConstraintSpec",
    "DEFAULT_SPECS",
    "voltage_violation_cost",
    "loading_cost",
    "comfort_cost",
    "blackout_cost",
    "fairness_cost",
    "ConstraintSet",
]
