"""MAPPO — centralised-critic multi-agent PPO (Phase 1, task m3-t3).

The primary learner of the thesis. Phase-1 milestone requires reliable
convergence on the IEEE-14 environment.
"""

from __future__ import annotations

from govgridmarl.agents.base import BaseAgent


class MAPPOAgent(BaseAgent):  # pragma: no cover - stub
    """Multi-Agent PPO with a centralised critic. TODO(m3-t3)."""

    def act(self, observation):
        raise NotImplementedError("MAPPOAgent.act — scheduled for Month 3")

    def update(self, batch):
        raise NotImplementedError("MAPPOAgent.update — scheduled for Month 3")
