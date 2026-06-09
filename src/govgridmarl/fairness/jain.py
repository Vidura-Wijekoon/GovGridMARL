"""Jain's fairness index.

Used to quantify how evenly demand-response curtailment burden is shared
across households. Target for the thesis (Phase 2 milestone): Jain >= 0.85.

Jain's index for allocations x_1..x_n is::

        (sum_i x_i)^2
    J = ---------------
        n * sum_i x_i^2

J lies in [1/n, 1]; J = 1 is perfectly equal allocation.

Reference: R. Jain, D. Chiu, W. Hawe, "A Quantitative Measure of Fairness
and Discrimination for Resource Allocation in Shared Computer Systems",
DEC Research Report TR-301, 1984.
"""

from __future__ import annotations

from collections.abc import Sequence


def jain_index(allocations: Sequence[float]) -> float:
    """Compute Jain's fairness index for a sequence of non-negative values.

    Parameters
    ----------
    allocations:
        Per-agent resource shares or burdens (e.g., curtailment per household).
        Must be non-empty and non-negative.

    Returns
    -------
    float
        Fairness index in [1/n, 1]. Returns 1.0 for an all-zero allocation
        (trivially equal).

    Raises
    ------
    ValueError
        If ``allocations`` is empty or contains a negative value.
    """
    x = list(allocations)
    if not x:
        raise ValueError("allocations must be non-empty")
    if any(v < 0 for v in x):
        raise ValueError("allocations must be non-negative")

    total = sum(x)
    if total == 0:
        return 1.0  # everyone gets zero -> perfectly equal

    sum_sq = sum(v * v for v in x)
    n = len(x)
    return (total * total) / (n * sum_sq)
