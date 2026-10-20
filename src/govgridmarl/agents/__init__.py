"""Reinforcement-learning agents.

Phase 1: PPO (centralised single-agent), IPPO (independent), MAPPO
(centralised critic) — all built on :mod:`govgridmarl.agents.ppo_core`.
Phase 2: MOPPO (multi-objective), Lagrangian / PID-Lagrangian constrained.
"""

from __future__ import annotations

from govgridmarl.agents.base import BaseAgent
from govgridmarl.agents.ippo import IPPOAgent
from govgridmarl.agents.lagrangian import (
    LagrangianController,
    MAPPOLagrangianAgent,
    PPOLagrangianAgent,
)
from govgridmarl.agents.mappo import MAPPOAgent
from govgridmarl.agents.moppo import MOPPOAgent
from govgridmarl.agents.ppo import PPOAgent
from govgridmarl.agents.ppo_core import PPOConfig, PPOFamilyAgent, compute_gae

REGISTRY: dict[str, type[PPOFamilyAgent]] = {
    "ppo": PPOAgent,
    "ippo": IPPOAgent,
    "mappo": MAPPOAgent,
    "moppo": MOPPOAgent,
    "mappo_lagrangian": MAPPOLagrangianAgent,
    "ppo_lagrangian": PPOLagrangianAgent,
}


def make_agent(
    name: str,
    obs_dim: int,
    state_dim: int,
    n_agents: int,
    n_actions: int,
    cfg=None,
    device: str = "cpu",
) -> PPOFamilyAgent:
    """Instantiate a learner by name (``ppo`` | ``ippo`` | ``mappo`` | ``moppo`` | Lagrangian variants)."""
    try:
        cls = REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown agent {name!r}; choose from {sorted(REGISTRY)}") from exc
    return cls(obs_dim, state_dim, n_agents, n_actions, cfg=cfg, device=device)


__all__ = [
    "BaseAgent",
    "PPOAgent",
    "IPPOAgent",
    "MAPPOAgent",
    "MOPPOAgent",
    "MAPPOLagrangianAgent",
    "PPOLagrangianAgent",
    "LagrangianController",
    "PPOConfig",
    "PPOFamilyAgent",
    "compute_gae",
    "make_agent",
    "REGISTRY",
]
