"""GovGrid environments.

Multi-agent demand-response environments built over electrical power-grid
test cases. Kept in a top-level ``envs/`` package (as per the proposal
layout) and installed alongside ``govgridmarl``.

  grid_marl_env   PettingZoo ParallelEnv over an IEEE bus system (m2-t12/13).
  ieee14          pandapower IEEE-14 case wrapper, household mapping, fast PF.
  grid2op_sandbox L2RPN / Grid2Op smoke environment (task m1-t13).
"""

from .grid_marl_env import GridMARLEnv, make_env

__all__ = ["GridMARLEnv", "make_env"]
