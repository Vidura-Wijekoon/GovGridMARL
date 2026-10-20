"""Pareto-front utilities for the efficiency-fairness trade-off (tasks m4-t5 / m4-t9).

Points are (peak_reduction_pct, jain_index) pairs, both to be maximised.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def non_dominated(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Boolean mask of the non-dominated points (all objectives maximised).

    A point is dominated if another point is >= on every objective and > on
    at least one.
    """
    p = np.asarray(points, dtype=float)
    n = p.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        ge = (p >= p[i]).all(axis=1)
        gt = (p > p[i]).any(axis=1)
        dom = ge & gt
        dom[i] = False
        if dom.any():
            mask[i] = False
    return mask


def pareto_front(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Non-dominated points sorted by the first objective."""
    p = np.asarray(points, dtype=float)
    if p.size == 0:
        return p.reshape(0, 2)
    f = p[non_dominated(p)]
    return f[np.argsort(f[:, 0])]


def hypervolume_2d(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """Dominated hypervolume (area) of a 2-D front w.r.t. ``ref`` (maximisation).

    Points below the reference in either objective contribute nothing.
    """
    p = np.asarray(points, dtype=float)
    r = np.asarray(ref, dtype=float)
    if p.size == 0:
        return 0.0
    p = p[(p > r).all(axis=1)]
    if p.size == 0:
        return 0.0
    f = pareto_front(p)[::-1]  # descending in objective 0
    hv, prev_y = 0.0, r[1]
    for x, y in f:
        hv += (x - r[0]) * max(0.0, y - prev_y)
        prev_y = max(prev_y, y)
    return float(hv)


def coverage(front_a, front_b) -> float:
    """C-metric (Zitzler & Thiele, 1999): share of ``front_b`` dominated by ``front_a``."""
    a = np.asarray(front_a, dtype=float)
    b = np.asarray(front_b, dtype=float)
    if b.size == 0 or a.size == 0:
        return 0.0
    dominated = 0
    for q in b:
        if (((a >= q).all(axis=1)) & ((a > q).any(axis=1))).any():
            dominated += 1
    return dominated / b.shape[0]


def in_acceptable_region(peak_pct: float, jain: float, peak_min=15.0, jain_min=0.85) -> bool:
    """Phase-2 operating region: peak cut >= 15 % and Jain >= 0.85 (m4-t10)."""
    return bool(peak_pct >= peak_min and jain >= jain_min)


__all__ = ["non_dominated", "pareto_front", "hypervolume_2d", "coverage", "in_acceptable_region"]
