"""Fairness and energy-justice metrics (Phase 2, m4-t1 / m4-t2)."""

from govgridmarl.fairness.jain import jain_index
from govgridmarl.fairness.metrics import (
    burden_fraction,
    fair_share_deviation,
    fairness_report,
    gini,
    group_burden,
    group_gap_ratio,
    vulnerable_burden_ratio,
)

__all__ = [
    "jain_index",
    "burden_fraction",
    "gini",
    "fair_share_deviation",
    "group_burden",
    "group_gap_ratio",
    "vulnerable_burden_ratio",
    "fairness_report",
]
