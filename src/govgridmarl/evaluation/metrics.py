"""Core evaluation metrics.

Phase-1 evaluation protocol fixes: peak reduction %, cost, episode return.
Phase-2 milestone targets: peak cut >= 15 %, Jain >= 0.85, zero violations.
"""

from __future__ import annotations

from collections.abc import Sequence


def peak_reduction(baseline_peak: float, controlled_peak: float) -> float:
    """Percentage reduction in peak demand achieved by the controller.

    Parameters
    ----------
    baseline_peak:
        Peak demand under the no-control / baseline policy (must be > 0).
    controlled_peak:
        Peak demand under the learned policy.

    Returns
    -------
    float
        Peak reduction as a percentage (positive = improvement).
    """
    if baseline_peak <= 0:
        raise ValueError("baseline_peak must be positive")
    return (baseline_peak - controlled_peak) / baseline_peak * 100.0


def episode_return(rewards: Sequence[float], gamma: float = 1.0) -> float:
    """Discounted (or undiscounted, gamma=1) return of a reward sequence."""
    g = 0.0
    for r in reversed(list(rewards)):
        g = r + gamma * g
    return g
