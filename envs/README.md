# `envs/` — Multi-agent grid environments

Top-level environments package (`govgrid_envs`), installed alongside the
core `govgridmarl` library.

| Module | Purpose | Roadmap task |
|---|---|---|
| `grid_marl_env.py` | PettingZoo `ParallelEnv` over an IEEE bus system | m2-t12 / m2-t13 |
| `ieee14.py` | pandapower IEEE-14 case + Newton-Raphson power flow | m1-t14 / m2-t11 |
| `grid2op_sandbox.py` | Grid2Op / L2RPN sandbox smoke test | m1-t13 |

```python
from govgrid_envs import GridMARLEnv
env = GridMARLEnv(n_households=14, case="ieee14", horizon=24, seed=42)
```
