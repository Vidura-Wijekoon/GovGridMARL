"""Tests for Jain's fairness index."""

import math

import pytest

from govgridmarl.fairness import jain_index


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
