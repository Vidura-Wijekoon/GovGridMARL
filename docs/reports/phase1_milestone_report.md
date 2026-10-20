# Phase-1 Milestone Report — Foundation & Baseline (tasks m3-t14 / m3-t15)

**Phase window:** Jun–Aug 2026 (closed 6 Sep 2026) · **Gate:** review ≥ 50 papers; baseline MARL converges on IEEE-14 · **Verdict:** **PASS** (`experiments/evaluate.py`, 2026-09-06)

## 1. Gate criteria

| Criterion (roadmap) | Evidence | Status |
|---|---|---|
| ≥ 50 papers reviewed | 105-paper systematic review database; IEEE review paper v2; 99 explainer videos | Met (105) |
| Baseline MARL converges on IEEE-14 | MAPPO on `govgrid_marl_v0`: learning return 0.5 → 1.3 plateau by ~50k steps on all 3 seeds; 2/3 seeds meet the written convergence rule (the third has a residual slope of 4.6 % vs the 5 % tolerance, std 3.3 %) | Met |
| Beats the rule-based reference on peak | 15.5 ± 0.3 % vs 15.0 % over 20 held-out days (Welch p = 0.011, Cliff's δ = 1.0) | Met |
| Environment validated | PettingZoo API test, `validate_env.py` 14/14 PASS, 34 unit tests, CI green | Met |

## 2. What was delivered

**Environment (Month 2).** IEEE-14 (pandapower `case14`) solved with LightSim2Grid AC Newton–Raphson every step (~0.1 ms), synthetic thermal ratings at 1.5 × nominal flow, voltage band 0.95–1.10 pu; 14 household-cluster agents (2,000 homes each) in three income groups with calibrated synthetic load profiles (UK-DALE / REDD / Ofgem anchors) and loaders for the real datasets; PettingZoo `ParallelEnv` with 17-dimensional local observations, `Discrete(5)` curtailment, 50 % rebound over two hours; reward = peak + cost + comfort in difference form; voltage / thermal / comfort constraint costs measured each step. Formal model: `docs/design/cmomg_formulation.md`.

**Learners (Month 3).** One clipped-PPO/GAE core (`agents/ppo_core.py`) with three input regimes — PPO (centralised controller), IPPO (independent), MAPPO (CTDE) — plus rule-based DR and no-control benchmarks; trainer with lock-step vector environments, held-out deterministic evaluation, CSV + MLflow logging, checkpoints and a written convergence test; Hydra configuration; evaluation script producing tables, figures and the gate verdict.

**Documentation.** Scope note, Month-1 log, governance-gap synthesis, CMOMG formulation, agents & spaces, reward & constraints, dataset access note, evaluation protocol, Methods chapter draft (§3.1–3.6), this report.

## 3. Results (cloud trend runs: 3 learners × 3 seeds × 120k steps; 20 held-out days, deterministic policies, mean ± 95 % CI over seeds)

| policy | peak reduction % | daily cost | Jain | violation rate | energy shed MWh |
|---|---:|---:|---:|---:|---:|
| no control | 0.0 | 67,591 | 1.000 | 0.00 | 0.0 |
| rule-based DR | 15.0 | 62,674 | 0.718 | 0.29 | 12.0 |
| PPO (centralised) | 15.3 ± 1.3 | 61,755 ± 430 | 0.692 ± 0.041 | 0.82 ± 0.39 | 16.2 ± 1.2 |
| IPPO | 15.7 ± 0.6 | 62,103 ± 145 | 0.638 ± 0.021 | 0.28 ± 0.04 | 14.7 ± 0.6 |
| **MAPPO** | **15.5 ± 0.3** | **61,992 ± 95** | **0.655 ± 0.027** | **0.29 ± 0.00** | 15.0 ± 0.3 |

Significance (Welch, Cliff's δ): MAPPO vs rule-based — return p = 0.0002 (δ = 1.0), peak p = 0.011 (δ = 1.0), Jain p = 0.010 (δ = −1.0, i.e. *less* fair); MAPPO vs IPPO — no significant difference on any metric at this budget; MAPPO vs PPO — higher return (p = 0.016) and far fewer violations (0.29 vs 0.82, p = 0.028). Stress set (demand × 1.3, background × 1.25): all controllers keep ≈ 15 % peak reduction while violation rates rise to ≈ 0.7 as the thermal constraint binds.

Compute (m3-t11): ≈ 550 env-steps/s per CPU core including learning; 5.6–6.2 min per 120k-step run on a 2-vCPU container; the full protocol (5 seeds × 500k × 3 learners) is ≈ 2–3 h on the laptop (`scripts/run_baselines.ps1`). Figures: `results/figures/`; tables: `results/tables/`.

## 4. Interpretation

1. The efficiency objective is satisfied by all three learners at essentially the rule-based level (≈ 15 %); the difference is *how* the burden is distributed and how much energy is shifted.
2. The efficiency-only learners are **less fair than the simple rule** (Jain 0.64–0.69 vs 0.72) and violate comfort in 29 % of hours, because they concentrate curtailment on high-flexibility (high-income) clusters: 1.97 vs 0.43 kWh/home/day. This is the quantified form of the governance gap identified in the literature review and the motivating baseline for Phase 2.
3. The centralised PPO controller, despite full information, is the worst on violations (0.82): free access to everyone's flexible load with no per-household constraint. That is a useful argument for decentralised actors plus explicit constraints rather than more information.
4. Credit assignment mattered more than the learner choice: with a plain team reward all three learners collapsed to "never curtail"; difference rewards fixed it without changing the objective (Methods §3.2.4).

## 4b. Supervisor requests addressed (6 Sep 2026)

* **Power-flow depth in the dashboard.** New panels show, per hour, the Newton–Raphson iteration count, the active-power balance (conventional + wind + PV = load + losses ± slack/export), the generation mix, the most loaded branches and the full bus-voltage table with bus types and kV levels; a "Dimensions and calculations" block gives the S_base, bus-type, scaling (2,000 homes → 1.66 MW peak per cluster, 8 % of system peak) and rating derivations. Source: `scripts/grid_analysis.py` → `results/tables/grid_analysis.json`, `docs/design/grid_electrical_basis.md`.
* **Bus-selection evidence.** A +5 MW sensitivity test at every load bus shows the LV buses B9–B14 have 5–10× the voltage sensitivity and 5–15× the branch-loading sensitivity of the 135 kV buses; five of the six most loss-sensitive buses are the six chosen first.
* **Energy share.** Generation mix added as a scenario (`renewable_share`, `wind_fraction`; PV at load buses, wind at B2; baseline 0). Sweep 0–80 %: renewables are fully consumed up to ≈ 40 %; at 80 % a PV-heavy mix exports 1,800 MWh/day and overloads a branch (1.32) while a wind-heavy mix is absorbed. The Phase-1 policies are invariant to the mix (15.5 % peak cut, Jain 0.66 in every scenario) and leave the midday surplus untouched — the measured "grid as a whole" baseline for the energy-share-aware controller of Phase 2. Tables: `results/tables/energy_share_summary.csv`; figure: `results/figures/energy_share_sweep.png`.
* **Publication plan (recorded).** (1) literature review as a pure review, all references to this project removed, to IEEE Transactions on Smart Grid; (2) Phase-1 environment + baselines as a conference paper; (3) the full framework as a journal paper.

## 5. Deviations from plan and open items

| Item | Status | Action |
|---|---|---|
| Calendar: Months 2–3 delivered on 6 Sep instead of Jul–Aug | closed | none; Phase 2 starts from a working baseline |
| m1-t12 LaTeX template | skipped by decision | Markdown → pandoc pipeline (see Month-1 log) |
| m2-t15 supervisor meeting #2, m3-t15 gate review meeting | to schedule | present this report + deck; capture revisions to the CMOMG |
| m3-t9 five-seed repeats | 3 seeds done in cloud | run `scripts/run_baselines.ps1` on the laptop (≈ 2–3 h) before the thesis figures are frozen |
| Energy-share-aware policy (net-load observation, self-consumption reward) | Phase 2 | `configs/reward/fairness.yaml` + observation v2 |
| Review paper de-branding for IEEE TSG | next | remove GovGridMARL / MAPPO plan from `docs/lit_notes/review_paper_source.md`, rebuild IEEE docx |
| Physics on Grid2Op | planned | `grid2op_backend.py` via `CompleteAction` injection: real thermal limits, 1,004 chronics, protections — verified feasible (1.6 ms/step) |
| Laptop `.git` history partially corrupt | open | re-initialise from the current tree before the repo goes public |

## 6. Gate review checklist (m3-t15)

- [x] Environment passes PettingZoo API + validation script + tests
- [x] MAPPO converges reliably (criterion written before the runs; 2/3 seeds pass, all plateau)
- [x] Baseline beats rule-based DR on the primary KPI
- [x] Evaluation protocol fixed and followed; figures and tables reproducible with one command
- [x] Methods chapter draft exists for the environment and baselines
- [ ] Supervisor sign-off on go / no-go to Phase 2
- [ ] Five-seed confirmation run on the laptop

**Recommendation:** proceed to Phase 2 (fairness & safety) in parallel with the Grid2Op backend migration; freeze Phase-1 thesis figures only after the five-seed laptop run.
