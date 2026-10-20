# Efficiency–fairness trade-off on IEEE-14 (tasks m4-t7, m4-t9, m4-t11, m4-t12)

**Runs:** `scripts/run_phase2_month4.sh` (2026-10-20): MAPPO weighted-sum sweep λ ∈ {0, 0.5, 1, 2, 4, 8} × 3 seeds, fairness-form ablation at λ = 2 (gap/fraction, Jain/fraction) × 3 seeds, MOPPO × 3 seeds; 120k env-steps each. **Evaluation:** `experiments/pareto.py`, 20 held-out protocol days on each policy's training population (protocol §7; seed *k* = population *k*), deterministic policies; the rule-based reference is run on the same three populations. Tables: `results/tables/pareto_{sweep,summary,comparison}.csv`, `fairness_significance.csv`, `fairness_feasibility.csv`, `fairness_operating_point.json`. Figures: `results/figures/pareto_front.png`, `fairness_tradeoff.png`, `burden_by_group_fair.png`.

*Note.* The first evaluation of 2026-10-20 scored every policy on population 42 (the Phase-1 convention) and gave 15.6–15.7 % peak reduction at Jain$_{kWh}$ 0.65–0.69; those numbers are superseded by the own-population numbers below (CHANGELOG 0.4.0 / 0.5.0).

## 1. Sweep (mean over 3 populations)

| λ (fairness weight) | peak reduction % | Jain kWh | Jain relative (B/F) | violation rate | group gap (max/min) | vulnerable / other burden |
|---:|---:|---:|---:|---:|---:|---:|
| 0 (Phase-1 MAPPO) | 14.77 | 0.589 | 0.949 | 0.234 | 9.41 | 0.148 |
| 0.5 | 14.67 | 0.596 | 0.953 | 0.238 | 9.18 | 0.151 |
| 1 | 14.72 | 0.605 | 0.959 | 0.250 | 8.89 | 0.155 |
| 2 | 14.72 | 0.617 | 0.962 | 0.250 | 8.49 | 0.161 |
| 4 | 14.72 | 0.637 | 0.974 | 0.264 | 7.81 | 0.173 |
| 8 | 14.69 | 0.652 | 0.980 | 0.290 | 7.26 | 0.184 |
| rule-based DR | 11.16 | 0.675 | 0.999 | 0.276 | 6.57 | 0.206 |
| MOPPO, any λ | 14.67–14.68 | 0.630–0.635 | 0.973–0.975 | 0.250–0.255 | 7.80–7.87 | 0.172 |
| gap / fraction form, λ = 2 | 14.81 | 0.599 | 0.958 | 0.220 | 8.81 | 0.157 |
| Jain / fraction form, λ = 2 | 14.75 | 0.608 | 0.969 | 0.257 | 8.50 | 0.162 |

Peak reduction is the same at every λ (14.7 %); its 95 % CI (± 3 pp) is between-population variance: population 0 gives ≈ 13 % for every learner, populations 1 and 2 ≈ 15.4 %. Jain$_{kWh}$ rises monotonically with λ (0.589 → 0.652), the income-group gap falls from 9.4 to 7.3, the violation rate rises from 0.23 to 0.29 and the vulnerable/other burden ratio from 0.15 to 0.18: equalising kWh pushes the least flexible clusters toward their comfort bound. The learners beat the rule-based reference by 3.5 pp on the same populations (the Phase-1 margin of 0.5 pp was a population-42 artefact).

## 2. Where the acceptable region is

1. **The kWh-basis region is empty for every controller.** The scripted feasibility study (`scripts/fairness_feasibility.py`, population 42) shows the best peak reduction attainable at Jain$_{kWh}$ ≥ 0.85 is **9.0 %**; reaching 15 % needs every household at the 60 % cap, which gives Jain$_{kWh}$ = 0.735 and 25 % comfort violations. Flexible energy spans 0.81–5.91 kWh/home/day (ratio 7.3) by construction of the income groups. Decision and evidence: `docs/design/acceptance_thresholds.md` §Amendment (gate basis → relative burden, CHANGELOG 0.4.0).
2. **On the relative basis the Jain target holds at every λ** (0.949–0.980 ≥ 0.85). The peak target holds on populations 1 and 2 at every λ and fails on population 0 for every learner, including λ = 0 — a Phase-1 efficiency property, not a fairness cost. By the pre-registered rule (smallest λ, ties by peak) the operating point is **λ = 0**; **λ = 2** is carried into Month 5 as reward shaping because it narrows the income-group gap (9.4 → 8.5) and raises Jain$_{kWh}$ (0.589 → 0.617) at no cost in peak reduction (p > 0.8); λ ≥ 4 is rejected for the violation increase.
3. **Energy-justice reading.** The vulnerable/other burden ratio *rises* with λ: kWh-equality asks the vulnerable clusters for a larger share of their small flexibility. The two notions pull in opposite directions; the thesis reports both and argues for relative burden as the primary notion, with group-gap and vulnerable ratios as the audit metrics the governance layer publishes (Month 9).

## 3. Weighted-sum vs MOPPO (m4-t9)

| method | trained models | front points (non-dominated) | hypervolume (ref 0 %, 0.5) | C-metric (covers the other) |
|---|---:|---:|---:|---:|
| weighted-sum MAPPO | 18 (6 λ × 3 seeds) | 6 (4) | 7.09 | 1.00 |
| MOPPO | 3 | 6 (4) | 6.97 | 0.00 |

One MOPPO model reproduces the six-model front to within 2 % hypervolume but is **insensitive to its preference input**: all six evaluation preferences give the same policy (Jain$_{kWh}$ 0.630–0.635). The fairness advantage is small relative to the efficiency advantage for every λ ≤ 8 (the fairness term is bounded by (1 − Jain)/T per step), so the conditioned actor collapses to one compromise policy near λ ≈ 4. MOPPO gives Pareto *coverage* cheaply but not Pareto *control*; the weighted-sum sweep stays the reference, and per-objective advantage normalisation is listed as future work.

## 4. Fairness-form ablation (m4-t3)

The individual "gap" form gives the best peak reduction (14.8 %) and the lowest violation rate (0.22) at λ = 2, because an agent is never charged for others' choices; it is the form recommended for the constrained agent's shaping. The shared Jain form stays the default for comparability with the proposal.

## 5. Statistics (m4-t12)

Per-seed means over 20 days, 3 seeds (= populations) per configuration, t-based 95 % CIs; Welch's t-test and Cliff's δ against λ = 0 for peak reduction, Jain (both bases), violation rate and return (`fairness_significance.csv`). Because seed and population coincide, between-population variance dominates the CIs; decoupling the dataset seed from the training seed and the five-seed confirmation are scheduled for the Month-6 consolidation (m6-t4).
