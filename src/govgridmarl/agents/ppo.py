"""Single-agent PPO baseline (Phase 1, task m3-t1).

A *centralised controller*: one PPO policy observes the global grid state
and emits the joint curtailment action for all households (factorised as
one categorical head per household, all conditioned on the full state), and
is trained on the team-mean reward. This is the classic "treat the
multi-agent problem as one big MDP" reference point; it has full
information but no decentralised execution, so it upper-bounds what
information sharing alone can buy.
"""

from __future__ import annotations

from govgridmarl.agents.ppo_core import PPOFamilyAgent


class PPOAgent(PPOFamilyAgent):
    """Proximal Policy Optimisation, centralised single-agent baseline (m3-t1)."""

    mode = "ppo"
    team_reward = True


__all__ = ["PPOAgent"]
