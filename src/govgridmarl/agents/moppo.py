"""MOPPO — multi-objective PPO for Pareto-optimal policies (Phase 2, m4-t8)."""

from __future__ import annotations

from govgridmarl.agents.base import BaseAgent


class MOPPOAgent(BaseAgent):  # pragma: no cover - stub
    """Multi-objective PPO (efficiency vs fairness). TODO(m4-t8)."""

    def act(self, observation):
        raise NotImplementedError("MOPPOAgent.act — scheduled for Month 4")

    def update(self, batch):
        raise NotImplementedError("MOPPOAgent.update — scheduled for Month 4")
