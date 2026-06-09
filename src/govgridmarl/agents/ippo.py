"""Independent PPO multi-agent baseline (Phase 1, task m3-t2)."""

from __future__ import annotations

from govgridmarl.agents.base import BaseAgent


class IPPOAgent(BaseAgent):  # pragma: no cover - stub
    """Independent learners — one PPO per agent. TODO(m3-t2)."""

    def act(self, observation):
        raise NotImplementedError("IPPOAgent.act — scheduled for Month 3")

    def update(self, batch):
        raise NotImplementedError("IPPOAgent.update — scheduled for Month 3")
