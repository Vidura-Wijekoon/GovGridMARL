"""Tests for evaluation metrics."""

import math

import pytest

from govgridmarl.evaluation import episode_return, peak_reduction


def test_peak_reduction_pct():
    assert math.isclose(peak_reduction(100.0, 85.0), 15.0)


def test_peak_reduction_requires_positive_baseline():
    with pytest.raises(ValueError):
        peak_reduction(0.0, 1.0)


def test_episode_return_undiscounted():
    assert episode_return([1, 1, 1, 1]) == 4.0


def test_episode_return_discounted():
    # 1 + 0.5*1 = 1.5
    assert math.isclose(episode_return([1, 1], gamma=0.5), 1.5)
