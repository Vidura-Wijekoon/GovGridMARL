# Evaluation protocol (task m3-t7)

Fixed before any baseline number is reported; changes after this point are logged in CHANGELOG.

## 1. Metrics

| metric | definition | unit | direction | source |
|---|---|---|---|---|
| Peak reduction | $(\max_t P^{base}_t-\max_t P^{res}_t)/\max_t P^{base}_t$ on the residential aggregate; $P^{base}$ = same day, no control | % | ↑ | `episode_summary()["peak_reduction_pct"]` |
| Cost | total bill over all homes under the ToU tariff | currency/day | ↓ | `cost_total` |
| Episode return | team objective $\sum_t$ `info["team_reward"]` (peak + cost + comfort, difference form; do-nothing = 0) | — | ↑ | trainer `return_mean` (learning reward), `eval_return` |
| Jain index | Jain over per-household cumulative curtailment burden $B^i_T$ | [1/N, 1] | ↑ (target ≥ 0.85, Phase 2) | `jain_index` |
| Violation rate | share of steps with any positive constraint cost (voltage, thermal, comfort) | [0, 1] | ↓ (target 0, Phase 2) | `violation_rate` |
| Energy shed | $E^{base}-E^{res}$ | MWh/day | report | `energy_*_mwh` |
| Burden by group | mean $B^i_T$ per income group | kWh/home | report (energy justice) | `burden_by_group_kwh` |

Compute metrics (m3-t11): env-steps/s, wall-clock per run, peak RSS, hardware string.

## 2. Evaluation episodes

* **Seeds:** 20 held-out evaluation days, `seed ∈ {10_000, …, 10_019}`, `day_scale = 1.0` (nominal demand) — never used in training (training resets draw from `seed*1000+k`).
* **Stress set:** the same 20 seeds with `day_scale = 1.3` and `background_factor = 1.25`, reported separately (constraint behaviour under stress).
* **Policy:** deterministic (argmax) for learned agents; heuristics are deterministic by construction.
* **Checkpoint:** `checkpoints/final.pt` (end of training), not best-of-run, to avoid selection bias; `best.pt` is kept for diagnostics only.

## 3. Statistical treatment (m3-t9)

* ≥ 5 training seeds per learner (cloud trend runs use 3; the laptop script uses 5).
* Report mean ± 95 % CI over training seeds (t-distribution) of the per-seed means over evaluation days.
* Pairwise comparisons (learner vs rule-based; MAPPO vs IPPO) with Welch's t-test and Cliff's δ; α = 0.05, Holm correction across the metric family.
* Convergence (m3-t6): fitted change of the 5-point-smoothed training return over the last 15 updates within 5 % of the window mean and window std < 5 % of |mean| (the day-to-day demand noise alone gives ~3 %) — reported as a boolean per seed, and as "k/n seeds converged".

## 4. Baselines

| id | policy | role |
|---|---|---|
| `none` | never curtail | denominator for peak reduction; Jain = 1 by construction |
| `heuristic` | rule-based DR: level 4 when price ≥ 0.40 or aggregate load ≥ 0.85·P_ref, level 1 when price ≥ 0.26, stop after 35 % of flexible energy | industry reference (the proposal's "rule-based DR" comparison for the ≥ 15 % target) |
| `ppo` | centralised controller on global state, team reward | information upper bound |
| `ippo` | independent PPO, shared params + agent id | decentralised lower bound |
| `mappo` | CTDE MAPPO | thesis learner |

## 5. Figures & tables (m3-t10)

1. Training curves: mean return and peak reduction vs env-steps, mean ± std shading across seeds, one panel per learner.
2. Benchmark table: all metrics, mean ± CI, all policies.
3. Daily profile: residential aggregate MW for no-control vs each policy on one evaluation day.
4. Burden distribution: per-household curtailment kWh coloured by income group (motivates Phase 2).

Produced by `python experiments/evaluate.py` into `results/figures/` and `results/tables/`.

## 6. Phase-2 additions (fixed 2026-09-22, tasks m4-t10 / m5-t1; logged per §0 rule)

| addition | definition | used by |
|---|---|---|
| Jain (fraction basis) `jain_frac` | Jain over $B^i_T/F^i_T$ (burden relative to own flexible energy) — reported next to the gate metric, never instead of it | `experiments/pareto.py`, `evaluate_phase2.py` |
| Gini, group gap, vulnerable ratio | `fairness/metrics.py`: Gini of kWh burden; max/min of income-group mean burden; mean burden of vulnerable clusters / others | fairness tables |
| Violations by constraint | per-constraint counts (voltage / thermal / comfort); `violation_rate` unchanged (any safety cost > 0) | safety ablation |
| Fairness constraint cost | $\max(0,0.85-\mathrm{Jain}(B_t))/T$, measured every step; not a safety violation | joint Lagrangian agent (m5-t10) |
| Acceptance region | peak ≥ 15 % **and** Jain ≥ 0.85 **and** violation rate = 0 on one policy, mean over seeds (`docs/design/acceptance_thresholds.md`) | gate (m5-t11) |
| Pareto comparison | front size, 2-D hypervolume (ref. 0 %, 0.5) and C-metric between the weighted-sum sweep and MOPPO over λ ∈ {0, 0.5, 1, 2, 4, 8} | m4-t9 |
| N-1 contingency set | the three single-line outages with no-control loading in (1, 2] (binding, relievable); 10 protocol days each, ratings of the intact case | m5-t12 |
| Stress reporting rule | on the stress and N-1 sets the target is *fewer violations than the unconstrained learner at equal peak reduction*, not zero | m5-t12 |

Training runs for Phase 2 use 120k environment steps and 3 seeds (laptop, ≈ 6 min per run with 8 in parallel); the five-seed confirmation of the operating point is scheduled with the Phase-2 consolidation (m6-t4).

## 7. Population rule (clarified 2026-10-20, logged per §0 rule)

The synthetic household population is generated from the **training seed** (`GridMARLEnv(seed=cfg.seed)` → `generate_profiles(seed)`), so a policy trained with seed *k* has only ever seen population *k*. "Held-out" in §2 means held-out **days** (seeds 10 000–10 019, nominal demand) on the **training population**; this is what the trainer's in-run evaluation (`eval.csv`) has always done. `experiments/evaluate.py` (Phase 1) instead built population 42 for every policy, i.e. it reported a *transfer* result. Consequences:

* `experiments/pareto.py` and `experiments/evaluate_phase2.py` evaluate each policy on its own population (3 seeds = 3 populations); fixed policies (no control, rule-based DR) are run on all three populations and averaged.
* The population-42 result is kept as the explicit scenario `transfer_pop42` in `evaluate_phase2.py` (generalisation to unseen households).
* Population 0 is the hardest of the three (no-control-relative peak reduction ≈ 12.5 % for every learner vs ≈ 15 % on populations 1 and 2); the Phase-1 figure of 15.5 % was a population-42 number. The Phase-1 tables are regenerated on the training populations in the Month-6 consolidation (m6-t6) — the Phase-1 gate (convergence, beats the rule-based reference) is unaffected because the rule-based policy is evaluated on the same population.
