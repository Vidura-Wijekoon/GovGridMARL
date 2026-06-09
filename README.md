# GovGridMARL

**Governance-Aware Multi-Agent Reinforcement Learning for Residential Smart-Grid Demand Response**

[![CI](https://github.com/Vidura-Wijekoon/GovGridMARL/actions/workflows/ci.yml/badge.svg)](https://github.com/Vidura-Wijekoon/GovGridMARL/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> MSc thesis codebase. A constrained multi-agent RL framework for residential
> demand response that is **safe, fair, private, explainable, and auditable** —
> with a runtime governance layer aligned to the EU AI Act.

---

## Overview

Most multi-agent RL (MARL) work on smart-grid demand response optimises for
**efficiency only** (peak shaving, cost). GovGridMARL closes the resulting
**governance gap** by training MARL agents under explicit fairness and safety
constraints, protecting household privacy with federated learning and
differential privacy, explaining every curtailment decision with SHAP, and
recording decisions in a tamper-evident audit log mapped to EU AI Act Annex IV.

Agents (household / aggregator / DSO) act over an IEEE-14 (and, scale
permitting, IEEE-118) network simulated with **pandapower**, wrapped as a
**PettingZoo** parallel environment. The baseline learner is **MAPPO**
(centralised critic), compared against PPO, IPPO, and rule-based heuristics.

## Research questions

1. Can constrained MARL meet hard **safety** limits (voltage, no-blackout)
   while delivering meaningful peak reduction?
2. What is the **efficiency–fairness** trade-off, and can we reach
   Jain ≥ 0.85 with ≥ 15 % peak reduction?
3. Can **federated learning + differential privacy** retain performance at
   ε ≤ 1.0 while resisting membership/occupancy inference?
4. Can decisions be **explained** (SHAP fidelity ≥ 90 %) and **audited**
   (100 % coverage) to satisfy emerging AI-governance regulation?

## Milestones (phase gates)

| Phase | Milestone |
|---|---|
| P1 — Foundation & Baseline | Review ≥ 50 papers; baseline MARL converges on IEEE-14 |
| P2 — Fairness & Safety | Jain ≥ 0.85; ≥ 15 % peak cut; zero safety violations |
| P3 — Privacy, XAI & Audit | DP ε ≤ 1.0; SHAP fidelity ≥ 90 %; audit log 100 % coverage |
| P4 — Integration & Compliance | End-to-end pipeline passes; EU AI Act Annex IV checklist met |
| P5 — Submission | Thesis submitted; repo public with CI passing |

See [`docs/roadmap.md`](docs/roadmap.md) for the full 12-month, 180-task plan.

## Repository structure

```text
GovGridMARL/
├── src/govgridmarl/        # core library (installable package)
│   ├── agents/             # PPO, IPPO, MAPPO, MOPPO, Lagrangian-constrained
│   ├── fairness/           # Jain index + fairness reward terms        [impl]
│   ├── safety/             # constraint costs + runtime guardrails
│   ├── privacy/            # FedAvg + DP-SGD + privacy accountant
│   ├── explainability/     # SHAP / LIME attribution
│   ├── governance/         # tamper-evident audit log, Annex IV map     [impl]
│   ├── data/               # REDD / UK-DALE loaders + synthetic profiles
│   ├── training/           # Hydra-driven training loop
│   ├── evaluation/         # metrics + eval protocol                    [impl]
│   └── utils/              # seeding, logging                           [impl]
├── envs/govgrid_envs/      # PettingZoo grid env, IEEE-14, Grid2Op sandbox
├── configs/                # Hydra config tree (env / agent / reward / ...)
├── experiments/            # train.py, evaluate.py, sweeps/
├── tests/                  # pytest suite (passing on the implemented parts)
├── scripts/                # setup_env.sh, solve_ieee14_powerflow.py
├── docs/                   # architecture.md, roadmap.md
├── data/                   # raw / interim / processed  (git-ignored)
├── results/                # run outputs                (git-ignored)
├── .github/workflows/      # CI (lint + tests)
├── environment.yml         # conda env (pinned in task m1-t2)
├── pyproject.toml          # packaging + tooling (ruff/black/pytest/mypy)
├── Makefile · CITATION.cff · CONTRIBUTING.md · CHANGELOG.md · LICENSE
```

`[impl]` = working reference implementation today; other modules are
documented stubs scheduled to their roadmap month.

## Installation

```bash
git clone https://github.com/Vidura-Wijekoon/GovGridMARL.git
cd GovGridMARL

# Option A — conda (recommended; full research stack)
conda env create -f environment.yml
conda activate govgridmarl

# Option B — uv (fast) or pip, from the pinned lock
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.lock && uv pip install -e .

# One-shot helper (auto-detects conda/uv/pip)
bash scripts/setup_env.sh
```

Dependencies are fully pinned (task **m1-t2**): `environment.yml` (conda) and
`requirements.lock` (179-package transitive closure, resolved for Python 3.11
/ Linux) — both validated by an IEEE-14 power-flow run. Edit `requirements.in`
and recompile with
`uv pip compile requirements.in -o requirements.lock --python-version 3.11 --python-platform linux`.
`torch==2.12.0` ships CUDA 13 wheels; for CPU-only or another CUDA, install
torch from the matching [PyTorch index](https://pytorch.org/get-started/locally/).

## Quickstart

```bash
# Run the test suite (works at the skeleton stage)
pytest

# Compose and inspect an experiment config (needs hydra-core)
python experiments/train.py +experiment=baseline_ieee14

# Once pandapower is installed: confirm IEEE-14 power flow converges (m1-t14)
python scripts/solve_ieee14_powerflow.py
```

```python
from govgridmarl.fairness import jain_index
from govgridmarl.evaluation import peak_reduction

jain_index([5, 5, 5, 5])      # -> 1.0  (perfectly equal burden)
peak_reduction(100.0, 85.0)   # -> 15.0 (% peak reduction)
```

## Tech stack

PyTorch · PettingZoo · Grid2Op · pandapower · SHAP / LIME · Opacus (DP) ·
Hydra · Weights & Biases / MLflow · pytest · ruff · black.

## Datasets

REDD and UK-DALE (residential load), plus reproducible synthetic profiles.
Data is **not** committed — see [`data/README.md`](data/README.md) for access.

## Development

```bash
make lint     # ruff + black --check + mypy
make test     # pytest
make format   # auto-fix
pre-commit install
```

CI runs lint + tests on every push and PR (Python 3.10 / 3.11).

## Citation

If you use this work, please cite it via [`CITATION.cff`](CITATION.cff).

## License

Released under the [MIT License](LICENSE). © 2026 Vidura.

## Acknowledgements

MSc thesis project (Jun 2026 – May 2027), conducted under academic supervision.
Built on the open-source RL, power-systems, and AI-governance ecosystems.
