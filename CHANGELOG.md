# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer.

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
