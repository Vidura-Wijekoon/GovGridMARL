# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer.

## [0.1.1] - 2026-06-10
### Added
- Reproducible Python environment (task m1-t2): `requirements.in` (direct deps),
  `requirements.lock` (179-package pinned transitive closure for Python 3.11 /
  Linux, resolved with `uv pip compile`), and a pinned `environment.yml`.
- Core stack pinned: torch 2.12.0, pettingzoo 1.26.1, supersuit 3.11.0,
  gymnasium 1.3.0, grid2op 1.12.4, pandapower 3.4.0, lightsim2grid 0.13.1,
  shap 0.49.1, opacus 1.6.0, hydra-core 1.3.2, wandb 0.27.2, mlflow 3.13.0,
  numpy 2.3.5, numba 0.65.1.
### Changed
- `scripts/setup_env.sh` now installs from the lock (conda / uv / pip).
- `requirements.txt` repurposed as a pointer to `requirements.lock`.
### Validated
- IEEE-14 Newton-Raphson power flow converges; numba 0.65.1 JIT works under numpy 2.3.5.

## [0.1.0] - 2026-06-09
### Added
- Initial version-controlled repository skeleton (roadmap task m1-t1).
- Package layout: `src/govgridmarl` (agents, fairness, safety, privacy,
  explainability, governance, data, training, evaluation, utils) and
  top-level `envs/govgrid_envs`.
- Working reference implementations: Jain fairness index, evaluation metrics,
  reproducible seeding, hash-chained audit log.
- Hydra config tree, experiment entrypoints, pytest suite, CI workflow,
  MIT license, CITATION.cff, and documentation (architecture, roadmap).
