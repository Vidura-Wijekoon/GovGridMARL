"""MAPPO — centralised-critic multi-agent PPO (Phase 1, task m3-t3).

Centralised Training with Decentralised Execution (Yu et al., 2022, "The
Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"): the
actor of every household sees only its local observation (so it can run on
an edge device, which matters for the privacy layer in Phase 3), while the
critic is trained on the global state. This is the primary learner of the
thesis; the Phase-1 milestone is its reliable convergence on IEEE-14, and
Phase 2 extends it with Lagrangian fairness / safety constraints.

Implemented natively (see :mod:`govgridmarl.agents.ppo_core`) rather than
via an external MARL library: the pinned stack has no maintained MAPPO for
PettingZoo Parallel envs with discrete actions on PyTorch 2.12, and a
~300-line auditable implementation is preferable for a governance thesis.
"""

from __future__ import annotations

from govgridmarl.agents.ppo_core import PPOFamilyAgent


class MAPPOAgent(PPOFamilyAgent):
    """Multi-Agent PPO with a centralised critic (m3-t3)."""

    mode = "mappo"
    team_reward = False


__all__ = ["MAPPOAgent"]
