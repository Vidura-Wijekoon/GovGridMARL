"""Independent PPO multi-agent baseline (Phase 1, task m3-t2).

Each household runs PPO on its *local* observation with a local critic
(de Witt et al., 2020, "Is Independent Learning All You Need in the
StarCraft Multi-Agent Challenge?"). Parameters are shared across agents
with an agent-id one-hot by default; ``share_params: false`` trains one
network per household.
"""

from __future__ import annotations

from govgridmarl.agents.ppo_core import PPOFamilyAgent


class IPPOAgent(PPOFamilyAgent):
    """Independent learners — decentralised actor *and* critic (m3-t2)."""

    mode = "ippo"
    team_reward = False


__all__ = ["IPPOAgent"]
