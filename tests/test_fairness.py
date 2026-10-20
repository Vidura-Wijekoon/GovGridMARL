"""Tests for the fairness metrics (m4-t1 / m4-t2) and the Pareto tools (m4-t5)."""

import math

import numpy as np
import pytest

from govgridmarl.evaluation import (
    check_targets,
    coverage,
    hypervolume_2d,
    in_acceptable_region,
    non_dominated,
    pareto_front,
)
from govgridmarl.fairness import (
    burden_fraction,
    fair_share_deviation,
    fairness_report,
    gini,
    group_burden,
    group_gap_ratio,
    jain_index,
    vulnerable_burden_ratio,
)


def test_perfectly_equal_is_one():
    assert math.isclose(jain_index([5, 5, 5, 5]), 1.0)


def test_all_zero_is_one():
    assert jain_index([0, 0, 0]) == 1.0


def test_single_user_concentration():
    # one user gets everything -> J = 1/n
    assert math.isclose(jain_index([10, 0, 0, 0]), 0.25)


def test_bounds():
    j = jain_index([1, 2, 3, 4, 5])
    assert 0.0 < j <= 1.0


def test_rejects_empty():
    with pytest.raises(ValueError):
        jain_index([])


def test_rejects_negative():
    with pytest.raises(ValueError):
        jain_index([1, -1, 2])


# --- m4-t1: metrics -------------------------------------------------------------
def test_gini_extremes():
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([0, 0, 0, 8]) == pytest.approx(0.75)  # (n-1)/n for one carrier
    assert gini([0, 0, 0]) == 0.0


def test_burden_fraction_and_deviation():
    fr = burden_fraction([1.0, 2.0, 0.0], [2.0, 2.0, 0.0])
    assert np.allclose(fr, [0.5, 1.0, 0.0])
    dev = fair_share_deviation([1.0, 2.0, 0.0], [2.0, 2.0, 0.0])
    assert dev.sum() == pytest.approx(0.0)
    assert dev[1] > 0 > dev[2]


def test_kwh_and_fraction_bases_can_disagree():
    burden = [0.5, 1.0, 1.5]  # proportional to capacity
    capacity = [1.0, 2.0, 3.0]
    rep = fairness_report(burden, capacity)
    assert rep["jain_frac"] == pytest.approx(1.0)
    assert rep["jain_kwh"] < 1.0


# --- m4-t2: group / vulnerability -----------------------------------------------
def test_group_metrics():
    b = [1.0, 3.0, 2.0, 2.0]
    g = ["low", "high", "mid", "mid"]
    bg = group_burden(b, g)
    assert bg == {"low": 1.0, "high": 3.0, "mid": 2.0}
    assert group_gap_ratio(bg) == pytest.approx(3.0)
    assert vulnerable_burden_ratio(b, [True, False, False, False]) == pytest.approx(1.0 / (7 / 3))
    assert vulnerable_burden_ratio(b, [True] * 4) == 1.0


def test_synthetic_vulnerability_tags_are_deterministic():
    from govgridmarl.data import generate_profiles

    a, b = generate_profiles(20, seed=3), generate_profiles(20, seed=3)
    assert a.vulnerable == b.vulnerable
    assert all(v for v, g in zip(a.vulnerable, a.groups) if g == "low")
    assert not any(v for v, g in zip(a.vulnerable, a.groups) if g == "high")
    assert a.summary()["n_vulnerable"] == sum(a.vulnerable)


# --- m4-t5 / m4-t10: Pareto and targets ---------------------------------------
def test_pareto_front_and_hypervolume():
    pts = [(15.0, 0.6), (12.0, 0.9), (10.0, 0.7), (14.0, 0.8), (16.0, 0.5)]
    mask = non_dominated(pts)
    assert mask.tolist() == [True, True, False, True, True]
    front = pareto_front(pts)
    assert front[0].tolist() == [12.0, 0.9] and front[-1].tolist() == [16.0, 0.5]
    hv = hypervolume_2d(pts, ref=(0.0, 0.0))
    # exact area of the staircase
    assert hv == pytest.approx(12 * 0.9 + 2 * 0.8 + 1 * 0.6 + 1 * 0.5)
    assert coverage([(20.0, 1.0)], pts) == 1.0
    assert coverage(pts, [(20.0, 1.0)]) == 0.0


def test_acceptance_region_and_targets():
    assert in_acceptable_region(15.0, 0.85)
    assert not in_acceptable_region(14.9, 0.9)
    chk = check_targets({"peak_reduction_pct": 15.5, "jain_index": 0.86, "violation_rate": 0.0})
    assert chk.passed
    chk = check_targets({"peak_reduction_pct": 15.5, "jain_index": 0.80, "violation_rate": 0.0})
    assert not chk.passed and not chk.results["jain_index"]["ok"]
    assert "MISS" in chk.summary()
