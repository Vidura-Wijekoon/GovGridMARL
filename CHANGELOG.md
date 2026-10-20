# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/); versioning is SemVer.

## [0.5.0] - 2026-10-20 (Month 5 - safe RL & constraint satisfaction)
### Added
- Safety constraints (m5-t1/t2): `safety/constraints.py` (voltage band, thermal loading,
  comfort, no-blackout proxy, fairness constraint; `ConstraintSet`), `docs/design/safety_constraints.md`;
  env exposes `info["cost_vec"]` (magnitudes) and `info["cost_ind"]` (violating-step indicators),
  per-constraint violation counts, `fairness_target(_basis)`, N-1 line outages
  (`GridMARLEnv(contingency=k)`, `Ieee14PowerFlow.set_contingency`).
- Constrained learners (m5-t3/t4/t6): `agents/lagrangian.py` - `LagrangianController`
  (dual ascent / PID), `MAPPOLagrangianAgent`, `PPOLagrangianAgent`; cost critics and
  hooks in `ppo_core.py`; cost GAE, episode-cost tracking and multiplier logging in the
  trainer; configs `mappo_lagrangian`, `ppo_lagrangian`, experiment `fairness_safety`.
- Evaluation (m5-t9/t11/t12/t13): `experiments/evaluate_phase2.py` - ablation, joint
  target check, stress set, N-1 screening and tests, transfer-population scenario,
  stochastic-policy columns, multiplier figures; `scripts/safety_feasibility.py`;
  run scripts `run_phase2_month5.sh`, `run_phase2_month5b.sh`; 10 new tests (55 total).
- Docs: `docs/reports/safety_results.md`, `docs/thesis/safety_section.md` (m5-t14),
  `docs/logs/month5_log.md` with the meeting #4 agenda (m5-t15).
### Changed
- Lagrangian agents use the **indicator cost signal** by default (`cost_signal: indicator`):
  magnitude episode costs are O(0.03) on IEEE-14 and leave dual ascent inert (m5-t5).
- **Evaluation protocol section 7 (population rule):** policies are evaluated on the
  population they were trained on (dataset seed = training seed); population 42 is a
  transfer scenario. `pareto.py` / `evaluate_phase2.py` follow the rule; the Phase-1
  `evaluate.py` numbers (15.5 %) are population-42 numbers - regenerated in m6-t6.
- Version 0.5.0.
### Validated (3 populations x 120k steps, 20 protocol days each)
- All indicator-cost constrained learners: 0.000 violating hours (deterministic and
  stochastic) vs 0.250 unconstrained (p < 1e-4), peak reduction unchanged (14.6 vs 14.7 %,
  p >= 0.89), Jain relative 0.94-0.95. Joint agent: fairness and safety targets met on every
  population, peak >= 15 % on 2 of 3 (13.0 / 15.4 / 15.3 %; the shortfall on population 0 is
  shared by every learner incl. the Phase-1 baseline).
- Multiplier sweep: no oscillation at any rate; eta = 1.0 overshoots (lambda 15.6, -1.9 pp peak).
- N-1 / peak-demand stress: thermal overloads identical for every policy (unrelievable by
  DR, planning case); comfort violations 0.23-0.28 -> 0.00-0.09 at equal peak reduction.
- Transfer to unseen households: 0.21 violating hours (vs 0.30) - constraints are learned
  per population; motivates the Month-9 runtime guardrail.

## [0.4.0] - 2026-10-20 (Month 4 - multi-objective & fairness formulation)
### Added
- Fairness metrics (m4-t1): `fairness/metrics.py` - Jain on kWh and on relative burden
  B_i/F_i, Gini, income-group gap ratio, vulnerable/other burden ratio, `fairness_report`;
  vulnerability tags (m4-t2) in the synthetic generator and the UK-DALE/REDD loaders.
- Fairness reward (m4-t3): `GridMARLEnv(fairness_form=jain|gap, fairness_basis=kwh|fraction)`,
  per-step term F/T, unweighted objective vector `info["objectives"]`, new episode KPIs
  (`jain_frac`, `gini_kwh`, `group_gap_ratio`, `vulnerable_burden_ratio`), trainer logs.
- Scalarisation (m4-t4): `agents/scalarization.py` (WeightedSum, Chebyshev, LAMBDA_GRID);
  MOPPO (m4-t8): `agents/moppo.py` preference-conditioned multi-objective MAPPO with a
  vector critic; trainer multi-objective rollout path; `compute_gae` over trailing dims.
- Pareto tooling (m4-t5/t9): `evaluation/pareto.py` (non-dominated set, 2-D hypervolume,
  C-metric), `evaluation/targets.py` (`check_targets`), `experiments/pareto.py`
  (sweep tables, significance, operating point, figures), `scripts/run_phase2_month4.sh`,
  `scripts/fairness_feasibility.py`, Hydra experiments `fairness_sweep`, `moppo_ieee14`.
- Docs: `docs/design/acceptance_thresholds.md` (m4-t10), `docs/reports/fairness_tradeoff.md`
  (m4-t7/t12), `docs/thesis/fairness_section.md` (m4-t14), `docs/logs/month4_log.md` (m4-t15);
  thesis build pipeline `docs/thesis/thesis.yaml` + `scripts/build_thesis.py`, `make thesis`
  (closes m1-t12); evaluation protocol section 6; CI smoke for MOPPO and the constrained agent.
### Changed
- **Evaluation protocol amendment (rule 5 of the thresholds note):** the Phase-2 Jain gate
  (>= 0.85) is evaluated on relative burden B_i/F_i instead of kWh burden. Reason: the
  kWh region is empty for every policy on this population (flexible energy spans 0.81-5.91
  kWh/home/day, ratio 7.3): scripted equal-fraction/equal-kWh blends reach at most 9.0 %
  peak reduction at Jain_kWh >= 0.85, and 15 % requires every home at the 60 % cap
  (Jain_kWh 0.735, 25 % comfort violations). Jain_kWh, Gini, group gap and vulnerable ratio
  stay in every table. Evidence: `results/tables/fairness_feasibility.csv`.
- `configs/reward/fairness.yaml` fairness weight 2.0; `make phase2-m4 pareto` targets.
### Validated (3 populations x 120k steps, 20 held-out days; own-population numbers after the
### protocol clarification of 0.5.0 - the population-42 numbers first reported on 2026-10-20
### were 15.6-15.7 % peak / Jain_kWh 0.648-0.694)
- Weighted-sum sweep lambda in {0 .. 8}: peak reduction flat at 14.7 % (every lambda; population 0
  gives 13 % for every learner), Jain_kWh 0.589 -> 0.652, Jain_frac 0.949 -> 0.980, violation
  rate 0.23 -> 0.29, income-group gap 9.4 -> 7.3, vulnerable/other burden 0.15 -> 0.18
  (kWh-equality asks the vulnerable for more). Rule-based DR on the same populations: 11.2 %.
- MOPPO: one model reproduces the six-model front (hypervolume 6.97 vs 7.09) but is
  insensitive to its preference input (recorded limitation).
- Gap/fraction form: best peak reduction (14.8 %) and lowest violation rate (0.22).
- Month-4 gate on the amended basis: Jain met at every lambda; peak >= 15 % met on populations
  1-2 only (the population-0 shortfall is a Phase-1 baseline property); lambda = 2 carried to
  Month 5 as shaping.

## [0.3.2] - 2026-09-07
### Changed
- Phase-1 dashboard regenerated around the supervisor's narrative: section nav; (1) grid &
  power flow — one-line diagram beside the Newton-Raphson / load-balance panel; (2) bus
  selection & dimensions — sensitivity table, per-unit conversions, constraint formulae;
  (3) energy share — share sweep 0-80 % plus wind-fraction sweep 20/70/80/90 % at 80 % share,
  "grid as a whole" vs "energy-share view" KPIs and a cross-scenario sweep chart;
  (4) demand-response results incl. the 20-day protocol table; (5) gate verdict and
  publication plan.
- `scripts/showcase.py`: `ENERGY_SCENARIOS` carry a group tag; `_protocol_results()` reads
  `results/tables/benchmark.csv` + `compute_profile.json` for the gate pill.

## [0.3.1] - 2026-09-06
### Added
- Generation mix / energy share (supervisor request): `attach_renewables` (PV static
  generators at the 11 load buses, wind farm at B2), `Ieee14PowerFlow.solve(pv_mw, wind_mw)`,
  `GridMARLEnv(renewable_share, wind_fraction, wind_variability)` with per-step `info["mix"]`
  and episode KPIs (renewables consumed, surplus exported, net-import peak, system losses);
  `PFResult` now carries renewable/slack/per-generator MW and the NR iteration count.
- `scripts/grid_analysis.py` (PASS log + `results/tables/grid_analysis.json`): bus types & kV,
  Newton-Raphson iterations and P/Q balance, cluster MW scaling, rating derivation, +5 MW
  bus-sensitivity test, energy-share scenario table; `docs/design/grid_electrical_basis.md`.
- Dashboard panels: Power-flow analysis (per hour), Why the clusters sit on these buses,
  Energy share scenarios (0–80 %, wind/PV splits) with stacked generation-mix chart and
  "grid as a whole" vs "energy-share view" KPIs, Dimensions and calculations.
- `experiments/evaluate.py --energy-share`: sweep for all trained seeds ->
  `energy_share_{sweep,summary}.csv`, `figures/energy_share_sweep.png`.
- Methods draft §3.2.6–3.2.7; milestone report §4b; 3 new env tests (37 total).
### Validated
- Backends agree with renewables (1e-15 pu); energy balance closes per step; renewables
  displace slack generation monotonically; PV-heavy 80 % share -> 1,800 MWh surplus and
  max loading 1.32 (constraint case for Phase 2). Phase-1 policies are invariant to the mix.

## [0.3.0] - 2026-09-06
### Added
- Phase-1 baseline learners (m3-t1..t3): `agents/ppo_core.py` — one clipped-PPO/GAE
  core parameterised by actor/critic inputs — with `PPOAgent` (centralised
  controller), `IPPOAgent` (independent) and `MAPPOAgent` (CTDE, the thesis learner);
  `agents/heuristic.py` rule-based DR and no-control benchmarks; `make_agent` registry.
- Trainer (m3-t5/t6): `training/trainer.py` — lock-step vector envs, GAE, PPO updates,
  deterministic evaluation on held-out seeds, `metrics.csv`/`eval.csv`/`summary.json`,
  checkpoints, MLflow logging, and the written convergence test (`convergence_test`).
- Hydra wiring (m3-t4): `experiments/train.py` builds env/agent/trainer from `configs/`;
  `run_root`/`run_tag`; `experiment/baseline_cloud.yaml`; sweep spec in `experiments/sweeps/`.
- Evaluation (m3-t7/t8/t10): `docs/evaluation_protocol.md`; `experiments/evaluate.py`
  produces `results/tables/{benchmark,per_seed,per_seed_stress,significance}.csv`,
  `compute_profile.json` and `results/figures/{training_curves,benchmark,daily_profile,
  burden_by_group}.png`, and prints the Phase-1 gate verdict.
- Cloud trend runs (m3-t5/t9): 3 learners x 3 seeds x 120k steps under `results/runs/`;
  laptop script `scripts/run_baselines.ps1` (5 seeds x 500k) and `run_baselines_cloud.sh`.
- Docs: `docs/thesis/methods_draft.md` (m3-t12), `docs/reports/phase1_milestone_report.md`
  (m3-t14/t15), `docs/logs/month1_log.md` (m1-t15).
- Tests (m3-t13): `tests/test_agents.py` (act/update/checkpoint round-trip, GAE, heuristics);
  CI installs the CPU stack, runs ruff, pytest, `validate_env.py` and a 2-update training smoke.
- Showcase: `scripts/showcase.py` + `Start_Show.bat` regenerate and open the interactive
  IEEE-14 status page (`docs/status/govgridmarl_status.html`).
### Changed
- `GridMARLEnv` reward: shared terms in difference form relative to the no-control load
  (do-nothing = 0) and per-agent *difference rewards* by default (`reward_mode`);
  pilot runs with a plain team reward collapsed to "never curtail" (see Methods 3.2.4).
- Convergence tolerance 2 % -> 5 % (day-to-day demand noise alone gives ~3 %).
- Ruff config ignores E501/E731/B019/B023 (black owns line length); code black-formatted.
### Validated
- Phase-1 gate PASS: MAPPO 15.5 +- 0.3 % peak reduction on 20 held-out days vs 15.0 %
  rule-based DR; Jain 0.66, violation rate 0.29 (the Phase-2 gap); 6/9 seeds meet the
  convergence rule; ~550 env-steps/s per core including learning.

## [0.2.0] - 2026-09-06
### Added
- Design docs (m2-t5..t8): `docs/design/{governance_gap,cmomg_formulation,
  agents_and_spaces,reward_and_constraints}.md`.
- Data (m2-t9/t10): `data/synthetic.py` calibrated household generator (income groups,
  TDCV/UK-DALE anchors), `data/loaders.py` for UK-DALE/REDD with the same `LoadDataset`,
  `docs/data/datasets.md` (access, licences, calibration table).
- IEEE-14 mapping and fast power flow (m2-t11): `household_bus_map`, `attach_households`,
  `set_thermal_limits` (1.5 x nominal; case14 ships without ratings), `Ieee14PowerFlow`
  on LightSim2Grid (~0.1 ms/solve) with pandapower fallback.
- Environment (m2-t12/t13): `GridMARLEnv` PettingZoo `ParallelEnv` — 14 cluster agents,
  17-dim observations, `Discrete(5)` curtailment with rebound, peak/cost/comfort reward,
  measured voltage/thermal/comfort constraint costs, `state()` for centralised critics,
  `episode_summary()` KPIs; `make_env` factory; `configs/env/ieee14.yaml` filled in.
- Validation (m2-t14): `scripts/validate_env.py` (14 checks, PASS log) and
  `tests/test_env.py` (13 tests incl. PettingZoo API and backend agreement).
### Fixed
- Transformer ratings are kept outside `net.trafo.sn_mva` (pandapower defines
  `vk_percent` relative to it — writing ratings there changed the impedance).
- Upper voltage limit 1.10 pu (case14 generator set-points hold bus 8 at 1.09 pu).

## [0.1.3] - 2026-07-30
### Added
- IEEE-14 Newton-Raphson power flow (task m1-t14): `scripts/solve_ieee14_powerflow.py`
  loads pandapower `case14`, solves with NR, prints per-bus voltages and branch
  loadings, and asserts voltages in [0.94, 1.10] pu, no branch overload, active
  AND reactive power balance closing to <1e-3, plus a determinism re-solve check.
- Deliverable log: `results/logs/ieee14_powerflow_20260730_004228.log` (PASS).
- `envs/govgrid_envs/ieee14.py`: `load_ieee14`, `run_power_flow`, and a new
  `power_balance` helper (previously all `NotImplementedError` stubs).
### Fixed
- Reactive power balance now includes the case14 shunt capacitor (-21.18 Mvar);
  without it the balance appeared off by ~21 Mvar. Voltage band widened to 1.10
  pu because generator setpoints legitimately reach 1.09 pu (bus 8).
### Validated
- NR converges in ~50 ms on CPU (no numba); slack+gen 272.4 MW = load 259.0 MW
  + losses 13.4 MW; reactive mismatch 1.6e-08 Mvar.

## [0.1.2] - 2026-07-29
### Added
- Grid2Op / L2RPN sandbox smoke test (task m1-t13): `scripts/smoke_grid2op.py`
  verifies the install end-to-end — env creation (LightSim2Grid backend),
  IEEE-14 structure checks, seeded reset, full DoNothing episode, topology +
  redispatch action round-trip, Runner replay, and seeded reproducibility
  (requires `env.set_id()` — chronics rotate on reset independently of seed).
- Deliverable log: `results/logs/grid2op_smoke_20260729_124637.log` (PASS).
### Changed
- `envs/govgrid_envs/grid2op_sandbox.py`: `make_sandbox` implemented (was a
  stub) — prefers LightSim2Grid backend, falls back to PandaPower.
### Validated
- `l2rpn_case14_sandbox` full dataset (1,004 chronics) runs at ~1,000 env
  steps/s on CPU with lightsim2grid 0.13.1 under grid2op 1.12.4.

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
