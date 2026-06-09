"""Single-agent PPO baseline (Phase 1, task m3-t1).

Reference implementation against which the multi-agent learners are compared.
Planned dependency: PyTorch. Implemented in Month 3.
"""

from __future__ import annotations

from govgridmarl.agents.base import BaseAgent


class PPOAgent(BaseAgent):  # pragma: no cover - stub
    """Proximal Policy Optimisation. TODO(m3-t1): implement."""

    def act(self, observation):
        raise NotImplementedError("PPOAgent.act — scheduled for Month 3")

    def update(self, batch):
        raise NotImplementedError("PPOAgent.update — scheduled for Month 3")
