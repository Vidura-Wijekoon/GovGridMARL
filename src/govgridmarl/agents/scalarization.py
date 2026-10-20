"""Multi-objective scalarisation (task m4-t4).

The environment returns, for every agent and step, the objective vector
``info["objectives"] = {efficiency, fairness, comfort, safety}`` whose
weighted sum is the scalar reward the learner sees. The *weighted-sum*
scalarisation baseline is therefore a choice of reward weights; this module
makes that choice explicit and reusable by the Pareto sweep (m4-t5), the
MOPPO agent (m4-t8, which samples the weights per episode) and the
comparison (m4-t9).

Weighted-sum is the standard linear scalarisation; it can only recover the
convex part of the Pareto front (Das & Dennis, 1997), which is one reason to
compare it with the preference-conditioned MOPPO learner. A Chebyshev
(weighted min-max) scalarisation is included for the ablation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

OBJECTIVES: tuple[str, ...] = ("efficiency", "fairness")
"""Objectives traced on the Phase-2 Pareto front (comfort stays a fixed-weight term)."""

LAMBDA_GRID: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
"""Fairness weights of the weighted-sum sweep (m4-t5) and the MOPPO evaluation grid (m4-t9)."""


@dataclass
class WeightedSum:
    """Linear scalarisation ``sum_k w_k J_k``."""

    weights: dict[str, float] = field(default_factory=lambda: {"efficiency": 1.0, "fairness": 0.0})

    def __call__(self, objectives: Mapping[str, float]) -> float:
        return float(sum(w * float(objectives.get(k, 0.0)) for k, w in self.weights.items()))

    def vector(self, order: Sequence[str] = OBJECTIVES) -> np.ndarray:
        return np.asarray([self.weights.get(k, 0.0) for k in order], dtype=np.float32)


@dataclass
class Chebyshev:
    """Weighted Chebyshev scalarisation ``-max_k w_k (z*_k - J_k)`` (utopia point ``z*``)."""

    weights: dict[str, float]
    utopia: dict[str, float]

    def __call__(self, objectives: Mapping[str, float]) -> float:
        return -max(
            w * (self.utopia.get(k, 0.0) - float(objectives.get(k, 0.0)))
            for k, w in self.weights.items()
        )


def reward_weights_for_lambda(lam: float, base: Mapping[str, float] | None = None) -> dict:
    """Reward-weight dict for a point of the fairness sweep: ``fairness = lam``.

    ``base`` defaults to the Phase-1 efficiency weights (peak 1, cost 0.5, comfort 0.1).
    """
    w = {"peak": 1.0, "cost": 0.5, "comfort": 0.1, "fairness": 0.0, "safety": 0.0}
    w.update(dict(base or {}))
    w["fairness"] = float(lam)
    return w


def scalarise_vector(objectives: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """``objectives (..., K) . weights (K,) | (..., K)`` -> ``(...)``."""
    return np.sum(np.asarray(objectives) * np.asarray(weights), axis=-1)


__all__ = [
    "OBJECTIVES",
    "LAMBDA_GRID",
    "WeightedSum",
    "Chebyshev",
    "reward_weights_for_lambda",
    "scalarise_vector",
]
