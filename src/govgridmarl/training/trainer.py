"""Generic MARL training loop driven by Hydra configs (m3-t5).

Logs reward curves to the experiment tracker (W&B / MLflow, configured in
m1-t4). Kept framework-agnostic so PPO/IPPO/MAPPO/MOPPO share one loop.
"""

from __future__ import annotations


class Trainer:  # pragma: no cover - stub
    """Runs episodes, collects rollouts, calls agent.update(). TODO(m3-t5)."""

    def __init__(self, env, agent, cfg) -> None:
        self.env = env
        self.agent = agent
        self.cfg = cfg

    def fit(self):
        raise NotImplementedError("Trainer.fit — scheduled for Month 3")
