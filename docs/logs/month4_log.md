# Month-4 Log — Multi-objective & fairness formulation (task m4-t15)

**Period:** September 2026 (calendar slipped; closed 20 Oct 2026) · **Phase:** P2 Fairness & Safety · **Author:** Vidura

## Deliverables

| # | Task | Status | Evidence |
|---|------|--------|----------|
| m4-t1 | Jain fairness index module | Done | `fairness/metrics.py` (Jain kWh / relative, Gini, group gap, vulnerable ratio, `fairness_report`), `tests/test_fairness.py` |
| m4-t2 | Vulnerability / income tagging | Done | `data/synthetic.py::tag_vulnerability`, `LoadDataset.vulnerable`, loaders; `info["vulnerable"]`, episode KPIs |
| m4-t3 | Fairness reward term | Done | `GridMARLEnv(fairness_form, fairness_basis)`, per-step F/T term, `info["objectives"]`; `jain`, `jain_frac`, `vulnerable_burden_ratio` in `metrics.csv` / `eval.csv` |
| m4-t4 | Weighted-sum scalarisation | Done | `agents/scalarization.py` (WeightedSum, Chebyshev, λ grid), `configs/experiment/fairness_sweep.yaml` |
| m4-t5 | Pareto weight sweep | Done | `scripts/run_phase2_month4.sh` (λ ∈ {0, 0.5, 1, 2, 4, 8} × 3 seeds), `results/tables/pareto_sweep.csv` |
| m4-t6 | Fairness-augmented MAPPO training | Done | 24 runs under `results/runs_m4/`, logs `results/logs/m4_*.log` |
| m4-t7 | Trade-off analysis | Done | `docs/reports/fairness_tradeoff.md`, `scripts/fairness_feasibility.py` |
| m4-t8 | MOPPO | Done | `agents/moppo.py` (preference-conditioned, vector critic), trainer multi-objective path, `configs/agent/moppo.yaml` |
| m4-t9 | Scalarisation vs MOPPO | Done | `results/tables/pareto_comparison.csv` (hypervolume, C-metric) |
| m4-t10 | Acceptance thresholds | Done, **amended 2026-10-20** | `docs/design/acceptance_thresholds.md`, `evaluation/targets.py` |
| m4-t11 | Tune to the fairness target | Done | Jain (relative basis) met at every λ; peak ≥ 15 % on populations 1–2 only (population 0 ≈ 13 % for every learner); λ = 2 carried to Month 5 |
| m4-t12 | Statistical analysis | Done | `results/tables/fairness_significance.csv` (Welch, Cliff's δ, 95 % CIs) |
| m4-t13 | Fairness figures | Done | `results/figures/pareto_front.png`, `fairness_tradeoff.png`, `burden_by_group_fair.png` |
| m4-t14 | Fairness section draft | Done | `docs/thesis/fairness_section.md` (Chapter 4) |
| m4-t15 | Supervisor meeting #3 | Notes below | — |

Also closed this month: **m1-t12** (thesis template) — `docs/thesis/thesis.yaml` + `scripts/build_thesis.py` (Markdown → HTML, docx/pdf via pandoc), `make thesis`; **m2-t15 / m3-t15 meeting items** folded into meeting #3 below.

## Decisions

1. **Gate basis amended** (rule 5 of the thresholds note): Jain ≥ 0.85 is evaluated on relative burden $B_i/F_i$. Evidence: the kWh-basis region is empty for every policy (feasibility bound 9.0 % peak reduction at Jain$_{kWh}$ ≥ 0.85). kWh Jain, Gini, group gap and vulnerable ratio remain reported. CHANGELOG 0.4.0.
2. **Operating point** λ = 0 by the rule; λ = 2 (Jain form, kWh basis) is used as reward shaping in Month 5 because it improves kWh equity (group gap 9.4 → 8.5) at no cost. λ ≥ 4 rejected (violation rate 0.26–0.29).
5. **Population rule** (found while closing Month 5, protocol §7): policies are scored on their training population; the first Month-4 evaluation used population 42 and overstated peak reduction (15.7 % vs 14.7 %) — all Month-4 tables and this log were regenerated on the training populations.
3. **MOPPO** kept as the one-model alternative; its preference insensitivity is recorded as a limitation and a future-work item (advantage normalisation per objective).
4. **Gap form** recommended for the constrained agent's shaping (best peak, lowest violations).

## Known issues / open items

* Preference-conditioned MOPPO collapses to one policy for λ ≤ 8 (fairness advantage ≪ efficiency advantage).
* Only 3 seeds per configuration; 5-seed confirmation of the operating point planned with m6-t4.
* `results/` is git-ignored: tables and figures are reproduced with `make phase2-m4 pareto` (≈ 20 min on the laptop).

## Supervisor meeting #3 — agenda and record

**Agenda.** (1) Phase-1 gate sign-off (carried over from m3-t15) and Month-2 environment revisions (m2-t15); (2) the feasibility finding and the proposed amendment of the Jain basis; (3) choice of operating point for the safety phase; (4) MOPPO result; (5) plan for Month 5 (Lagrangian / PID-Lagrangian, N-1 tests).

**Points to record.** Approval of the Phase-1 gate (report §6); whether the relative-burden basis is accepted as the primary fairness notion for the thesis (with the kWh notion reported as the energy-justice counter-view) or whether a population with a narrower flexibility spread should be added as a sensitivity case (m6-t5); agreement that λ = 2 shaping + Lagrangian constraints is the Month-5 configuration; whether MOPPO stays in the thesis as a negative result or is dropped from the main text.

**Outcome (to be filled after the meeting).**
