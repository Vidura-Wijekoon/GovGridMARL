"""Fairness metrics over demand-response burden (task m4-t1).

The thesis measures *who carries the curtailment*. Per household cluster
``i`` the environment accumulates the curtailed energy ``B_i`` (kWh/home)
and the flexible energy ``F_i`` that was available (kWh/home). Two bases are
reported throughout Phase 2:

* **kWh basis** - Jain over ``B_i``: equal *absolute* burden. This is the
  definition fixed in the evaluation protocol (m3-t7) and the one the
  Phase-2 gate (Jain >= 0.85) is checked against.
* **fraction basis** - Jain over ``B_i / F_i``: equal *relative* burden, each
  household giving the same share of what it can give ("ability-to-pay"
  proportionality, the energy-justice reading of Area 3 in the review).

Both are exposed because they can disagree: low-income clusters have ~5x
less flexible energy than high-income ones, so a controller that is fair in
kWh must under-use high-income flexibility, while one that is fair in
fractions asks the vulnerable for the same *share*. The trade-off section of
the thesis reports both and the gate uses the kWh basis.

Group metrics (m4-t2) take the per-household group / vulnerability tags and
summarise burden by group, the max/min group ratio, and the burden carried
by vulnerable households relative to the rest.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from govgridmarl.fairness.jain import jain_index

GAP_CAP = 100.0  # finite cap for ratios whose denominator is a zero-burden group


def burden_fraction(burden: Sequence[float], capacity: Sequence[float]) -> np.ndarray:
    """Relative burden ``B_i / F_i`` (0 when nothing was available)."""
    b = np.asarray(burden, dtype=float)
    f = np.asarray(capacity, dtype=float)
    return np.where(f > 1e-12, b / np.maximum(f, 1e-12), 0.0)


def gini(values: Sequence[float]) -> float:
    """Gini coefficient of non-negative values (0 = equal, 1 = one carries all)."""
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if n == 0:
        raise ValueError("values must be non-empty")
    if (x < 0).any():
        raise ValueError("values must be non-negative")
    s = x.sum()
    if s <= 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2.0 * np.sum(idx * x)) / (n * s) - (n + 1.0) / n)


def fair_share_deviation(burden: Sequence[float], capacity: Sequence[float] | None = None):
    """Per-agent deviation from the population mean burden (the m4-t3 "gap" signal).

    With ``capacity`` the deviation is in relative burden (fraction basis),
    otherwise in kWh. Positive = carries more than the average.
    """
    s = (
        burden_fraction(burden, capacity)
        if capacity is not None
        else np.asarray(burden, dtype=float)
    )
    return s - s.mean()


def group_burden(burden: Sequence[float], groups: Sequence[str]) -> dict[str, float]:
    """Mean burden per group label (insertion order of first appearance)."""
    b = np.asarray(burden, dtype=float)
    g = np.asarray(groups)
    return {str(k): float(b[g == k].mean()) for k in dict.fromkeys(groups)}


def group_gap_ratio(by_group: dict[str, float]) -> float:
    """max / min of the group means (1 = groups carry the same mean burden)."""
    vals = np.asarray(list(by_group.values()), dtype=float)
    if vals.size == 0:
        return 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if hi <= 1e-12:
        return 1.0
    return float(min(hi / max(lo, 1e-12), GAP_CAP))


def vulnerable_burden_ratio(burden: Sequence[float], vulnerable: Sequence[bool]) -> float:
    """Mean burden of vulnerable households over that of the others (< 1 = protected).

    Returns 1.0 when either set is empty or nobody carries any burden.
    """
    b = np.asarray(burden, dtype=float)
    v = np.asarray(vulnerable, dtype=bool)
    if v.all() or (~v).all() or b.sum() <= 0:
        return 1.0
    other = float(b[~v].mean())
    return float(min(b[v].mean() / max(other, 1e-12), GAP_CAP))


def fairness_report(
    burden: Sequence[float],
    capacity: Sequence[float] | None = None,
    groups: Sequence[str] | None = None,
    vulnerable: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """All Phase-2 fairness numbers for one episode in one dictionary.

    Keys: ``jain_kwh``, ``gini_kwh`` and - when ``capacity`` is given -
    ``jain_frac``, ``gini_frac``, ``burden_frac_mean``; with ``groups``:
    ``burden_by_group``, ``group_gap_ratio``; with ``vulnerable``:
    ``vulnerable_burden_ratio``, ``vulnerable_share_of_burden``.
    """
    b = np.asarray(burden, dtype=float)
    out: dict[str, Any] = {
        "jain_kwh": float(jain_index(b.tolist())) if b.sum() > 0 else 1.0,
        "gini_kwh": gini(b),
    }
    if capacity is not None:
        fr = burden_fraction(b, capacity)
        out["jain_frac"] = float(jain_index(fr.tolist())) if fr.sum() > 0 else 1.0
        out["gini_frac"] = gini(fr)
        out["burden_frac_mean"] = float(fr.mean())
    if groups is not None:
        bg = group_burden(b, groups)
        out["burden_by_group"] = bg
        out["group_gap_ratio"] = group_gap_ratio(bg)
    if vulnerable is not None:
        v = np.asarray(vulnerable, dtype=bool)
        out["vulnerable_burden_ratio"] = vulnerable_burden_ratio(b, v)
        out["vulnerable_share_of_burden"] = (
            float(b[v].sum() / b.sum()) if b.sum() > 0 and v.any() else 0.0
        )
    return out


__all__ = [
    "burden_fraction",
    "gini",
    "fair_share_deviation",
    "group_burden",
    "group_gap_ratio",
    "vulnerable_burden_ratio",
    "fairness_report",
]
