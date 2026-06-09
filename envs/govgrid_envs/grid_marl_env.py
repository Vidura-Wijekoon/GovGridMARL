"""PettingZoo ParallelEnv wrapping a power grid for multi-agent demand response.

Agents (Phase 2 spec, m2-t7): household / aggregator / DSO. Each household
agent observes local load + price + grid signals and chooses a curtailment
action; the environment steps a pandapower power-flow and returns rewards
combining efficiency (peak/cost), fairness, and safety-constraint costs.

This is a skeleton: the API surface is fixed here so downstream agents,
configs, and tests can be written against it, with the dynamics implemented
in Month 2 (tasks m2-t12 / m2-t13).
"""

from __future__ import annotations

from typing import Any


class GridMARLEnv:
    """Parallel multi-agent grid environment (PettingZoo-compatible API).

    Parameters
    ----------
    n_households:
        Number of household agents attached to the grid.
    case:
        Grid test case identifier, e.g. ``"ieee14"`` or ``"ieee118"``.
    horizon:
        Episode length in timesteps.
    seed:
        RNG seed for reproducibility.
    """

    metadata = {"name": "govgrid_marl_v0"}

    def __init__(
        self,
        n_households: int = 14,
        case: str = "ieee14",
        horizon: int = 24,
        seed: int = 42,
    ) -> None:
        self.n_households = n_households
        self.case = case
        self.horizon = horizon
        self.seed = seed
        self.possible_agents = [f"household_{i}" for i in range(n_households)]
        self.agents: list[str] = []
        self._t = 0

    # -- PettingZoo ParallelEnv API (to implement in m2-t13) -----------------
    def reset(self, seed: int | None = None, options: dict | None = None):
        """Reset the environment; return (observations, infos)."""
        raise NotImplementedError("GridMARLEnv.reset — scheduled for Month 2 (m2-t13)")

    def step(self, actions: dict[str, Any]):
        """Apply joint actions; return (obs, rewards, terminations, truncations, infos)."""
        raise NotImplementedError("GridMARLEnv.step — scheduled for Month 2 (m2-t13)")

    def observation_space(self, agent: str):
        raise NotImplementedError("observation_space — scheduled for Month 2 (m2-t7)")

    def action_space(self, agent: str):
        raise NotImplementedError("action_space — scheduled for Month 2 (m2-t7)")

    def render(self):  # pragma: no cover - stub
        raise NotImplementedError

    def close(self):  # pragma: no cover - stub
        pass
