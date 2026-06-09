"""Differential privacy: DP-SGD and privacy accounting (Phase 3, m7-t4/m7-t5).

Phase-3 milestone: epsilon <= 1.0 with performance retained.
"""

from __future__ import annotations


class PrivacyAccountant:  # pragma: no cover - stub
    """Tracks the (epsilon, delta) budget across federated rounds. TODO(m7-t5)."""

    def __init__(self, target_epsilon: float = 1.0, delta: float = 1e-5) -> None:
        self.target_epsilon = target_epsilon
        self.delta = delta

    def step(self, *args, **kwargs):
        raise NotImplementedError("PrivacyAccountant.step — Month 7")
