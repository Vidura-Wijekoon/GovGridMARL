# Efficiency–fairness trade-off on IEEE-14 (tasks m4-t7, m4-t9, m4-t11, m4-t12)

**Runs:** `scripts/run_phase2_month4.sh` (2026-10-20): MAPPO weighted-sum sweep λ ∈ {0, 0.5, 1, 2, 4, 8} × 3 seeds, fairness-form ablation at λ = 2 (gap/fraction, Jain/fraction) × 3 seeds, MOPPO × 3 seeds; 120k env-steps each. **Evaluation:** `experiments/pareto.py`, 20 held-out protocol days, deterministic policies. Tables: `results/tables/pareto_{sweep,summary,comparison}.csv`, `fairness_significance.csv`, `fairness_feasibility.csv`, `fairness_operating_point.json`. Figures: `results/figures/pareto_front.png`, `fairness_tradeoff.png`, `burden_by_group_fair.png`.

## 1. Sweep (mean over 3 seeds; ± = 95 % CI)

| λ (fairness weight) | peak reduction % | Jain kWh | Jain relative (B/F) | violation rate | group gap (max/min) | vulnerable / other burden |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (Phase-1 MAPPO) | 15.67 ± 0.38 | 0.648 ± 0.047 | 0.975 | 0.280 | 4.81 | 0.32 |
| 0.5 | 15.60 ± 0.23 | 0.653 ± 0.038 | 0.978 | 0.284 | 4.78 | 0.32 |
| 1 | 15.64 ± 0.37 | 0.659 ± 0.038 | 0.979 | 0.297 | 4.65 | 0.33 |
| 2 | 15.70 ± 0.48 | 0.670 ± 0.042 | 0.984 | 0.300 | 4.54 | 0.34 |
| 4 | 15.67 ± 0.63 | 0.684 ± 0.035 | 0.986 | 0.309 | 4.27 | 0.35 |
| 8 | 15.59 ± 0.55 | 0.694 ± 0.031 | 0.988 | 0.361 | 4.11 | 0.36 |
| rule-based DR | 15.00 | 0.718 | 0.997 | 0.290 | 4.22 | 0.34 |
| MOPPO, any λ | 15.56–15.60 | 0.679–0.680 | 0.986–0.987 | 0.308–0.319 | 4.30 | 0.35 |
| gap / fraction form, λ = 2 | 15.81 ± 0.42 | 0.661 ± 0.035 | 0.979 | 0.278 | 4.52 | 0.34 |
| Jain / fraction form, λ = 2 | 15.67 ± 0.23 | 0.663 ± 0.026 | 0.982 | 0.319 | 4.51 | 0.34 |

Significance vs λ = 0 (Welch, 3 vs 3 seeds): peak reduction never differs (p ≥ 0.55); Jain kWh rises monotonically and reaches significance only at λ = 8 (p = 0.030, Cliff's δ = 1.0; λ = 4: p = 0.062). The violation rate increases with λ (0.28 → 0.36): equalising kWh pushes low-flexibility clusters toward their comfort bound.

## 2. Where the acceptable region is

1. **The kWh-basis region is empty for every controller.** The scripted feasibility study (`scripts/fairness_feasibility.py`) shows the best peak reduction attainable at Jain$_{kWh}$ ≥ 0.85 is **9.0 %**, and that reaching 15 % requires every household at the 60 % cap in the peak hours, which yields Jain$_{kWh}$ = 0.735 and 25 % comfort violations. Flexible energy spans 0.81–5.91 kWh/home/day (ratio 7.3) by construction of the income groups; equal kWh means holding everyone to the least flexible home. The learned policies (0.65–0.69) sit just below the equal-fraction rule (0.735) because they curtail the high-flexibility clusters slightly harder. This is a property of the population, not of the learner. Decision and evidence: `docs/design/acceptance_thresholds.md` §Amendment (gate basis → relative burden, CHANGELOG 0.4.0).
2. **On the relative basis every learned configuration is inside the region** (Jain$_{frac}$ 0.975–0.988 ≥ 0.85 with 15.6–15.8 % peak reduction). The operating point by the pre-registered rule (smallest λ, ties by peak reduction) is **λ = 0**: the shared-parameter policy already distributes burden proportionally to flexibility. The fairness term is therefore not needed for the gate; what it buys is *kWh* equity: at λ = 2 the income-group gap falls from 4.8 to 4.5 and Jain$_{kWh}$ rises 0.648 → 0.670 at no cost in peak reduction or violations (p > 0.8), so **λ = 2 is carried into Month 5 as reward shaping** (`configs/reward/safety.yaml`), while λ ≥ 4 is rejected because the violation rate grows.
3. **Energy-justice reading.** The vulnerable/other burden ratio *rises* with λ (0.32 → 0.36): kWh-equality asks the vulnerable clusters for a larger share of their small flexibility. The two fairness notions pull in opposite directions on this population; the thesis reports both and argues for relative burden as the primary notion, with group-gap and vulnerable ratios as the audit metrics the governance layer should publish (Month 9).

## 3. Weighted-sum vs MOPPO (m4-t9)

| method | trained models | front points (non-dominated) | hypervolume (ref 0 %, 0.5) | C-metric (covers the other) |
|---|---:|---:|---:|---:|
| weighted-sum MAPPO | 18 (6 λ × 3 seeds) | 6 (3) | 7.66 | 0.33 |
| MOPPO | 3 | 6 (3) | 7.60 | 0.00 |

MOPPO reproduces the front of the six separately trained weighted-sum policies with one model (hypervolume within 1 %), but its policy is **insensitive to the preference input**: all six evaluation preferences give the same KPIs (Jain$_{kWh}$ 0.679–0.680). The fairness advantage is small relative to the efficiency advantage for every λ ≤ 8 (the fairness term is bounded by (1 − Jain)/T per step), so the preference-conditioned actor collapses to a single compromise policy near λ ≈ 4. Pareto *coverage* is thus equal, Pareto *control* is not; the weighted-sum sweep remains the reference for Month 5 and MOPPO is kept as the cheaper alternative, with the fairness advantage scaling (λ_max, per-objective normalisation) listed as future work.

## 4. Fairness-form ablation (m4-t3)

The individual "gap" form (penalise only agents above the mean relative burden) gives the best peak reduction of the sweep (15.81 %) with the lowest violation rate (0.278) and the same kWh Jain as the shared Jain form, because it never penalises an agent for other agents' choices. It is the form recommended for the joint Lagrangian agent's reward shaping; the shared Jain form stays the default for comparability with the proposal.

## 5. Statistics (m4-t12)

Per-seed means over 20 days, 3 seeds per configuration, mean ± t-based 95 % CI; Welch's t-test and Cliff's δ against λ = 0 for peak reduction, Jain (both bases), violation rate and return (`fairness_significance.csv`). With n = 3 only large effects are detectable (Jain kWh at λ = 8); the five-seed confirmation is scheduled with the Phase-2 consolidation (m6-t4).
