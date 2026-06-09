"""GovGrid environments.

Multi-agent demand-response environments built over electrical power-grid
test cases. Kept in a top-level ``envs/`` package (as per the proposal
layout) and installed alongside ``govgridmarl``.

  grid_marl_env   PettingZoo ParallelEnv over an IEEE bus system.
  ieee14          pandapower IEEE-14 case wrapper + Newton-Raphson power flow.
  grid2op_sandbox L2RPN / Grid2Op smoke environment (task m1-t13).
"""

from envs.govgrid_envs.grid_marl_env import GridMARLEnv

__all__ = ["GridMARLEnv"]
