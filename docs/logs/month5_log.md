# Month-5 Log — Safe RL & constraint satisfaction (task m5-t15)

**Period:** October 2026 (closed 20 Oct 2026) · **Phase:** P2 Fairness & Safety · **Author:** Vidura

## Deliverables

| # | Task | Status | Evidence |
|---|------|--------|----------|
| m5-t1 | Formalise safety constraints | Done | `docs/design/safety_constraints.md`, `safety/constraints.py` (voltage, loading, comfort, no-blackout, fairness) |
| m5-t2 | Constraint cost signals in the env | Done | `info["cost_vec"]` / `info["cost_ind"]`, `violations_by_constraint`, `tests/test_safety.py` |
| m5-t3 | PPO-Lagrangian | Done | `agents/lagrangian.py` (`LagrangianController`, `PPOLagrangianAgent`), cost critics in `ppo_core.py`, cost GAE in the trainer |
| m5-t4 | Multipliers in MAPPO | Done | `MAPPOLagrangianAgent` (cost critics on the global state), `configs/agent/mappo_lagrangian.yaml` |
| m5-t5 | Tune multiplier learning rate | Done | `results/tables/multiplier_lr_sweep.csv`; finding: cost units, not the rate — indicator costs adopted |
| m5-t6 | PID-Lagrangian | Done | `method: pid_lagrangian`, `multiplier_dynamics.png` |
| m5-t7 | Train constrained MARL | Done | 33 runs under `results/runs_m5/` (two batches + sweep), violation rate tracked per update |
| m5-t8 | Drive violations to zero | **Done** | 0.000 violating hours on 20 days × 3 populations, deterministic and stochastic, no peak loss |
| m5-t9 | Ablation | Done | `results/tables/safety_ablation.md` (unconstrained / magnitude / indicator × dual ascent / PID / centralised) |
| m5-t10 | Fairness + safety agent | Done | `agent.constraints=[voltage,loading,comfort,fairness]`, joint runs |
| m5-t11 | Verify joint targets | Done (2 of 3 met everywhere; peak on 2/3 populations) | `results/tables/phase2_targets.csv`, `docs/reports/safety_results.md` §3 |
| m5-t12 | Stress + N-1 tests | Done | `stress_tests.csv`, `n1_screening.csv`, `GridMARLEnv(contingency=k)`, `stress_violations.png` |
| m5-t13 | Safety figures | Done | `violation_curves.png`, `multiplier_dynamics.png`, `multiplier_lr_sweep.png`, `safety_benchmark.png` |
| m5-t14 | Safety section draft | Done | `docs/thesis/safety_section.md` (Chapter 5) |
| m5-t15 | Supervisor meeting #4 | Notes below | — |

## Decisions

1. **Indicator cost signal** (violating hours per episode) is the default for all Lagrangian agents (`cost_signal: indicator`); magnitude costs are kept for the ablation. Reason: magnitude episode costs are O(0.03) on IEEE-14 and leave the multipliers inert (§1 of the safety report).
2. **Evaluation population rule** clarified (protocol §7): held-out *days* on the *training* population; population 42 is a transfer scenario. The Phase-1 15.5 % figure was a population-42 number; on the training populations the Phase-1 baseline is 14.7 % (13.2 / 15.4 / 15.5) against 11.2 % for the rule-based DR, so the Phase-1 gate conclusions hold with a larger margin over the reference. Phase-1 tables are regenerated in m6-t6.
3. **Phase-2 status at the end of October:** safety target met (zero violations, no peak cost), fairness target met on the relative basis, peak target met on two of three populations; population 0 is 2 pp short for every learner. Pre-registered fallback applies: the trade-off is reported and the gap is carried as the first Month-6 item (longer runs, 5 seeds, gap-form shaping).
4. Thermal overloads under N-1 / stress are reported, not targeted (rule 4): they are identical for no control and every policy.
5. The transfer gap (constraints learned per population) is recorded as the motivation for the deterministic runtime guardrail (m9-t4).

## Known issues / open items

* Peak target on population 0; convergence flag false for all Phase-2 runs under the 5 %-slope rule at 120k steps (the fairness/constraint terms add drift late in training) — rerun at 500k.
* Only 3 populations × 3 seeds coincide (seed = population); decouple dataset seed from training seed in Month 6 so that seeds and populations can be varied independently.
* `results/` is git-ignored; `make phase2-m5 evaluate-phase2` reproduces everything (≈ 45 min on the laptop).

## Supervisor meeting #4 — Phase-2 mid-review: agenda and record

**Agenda.** (1) Month-4 outcome and the Jain-basis amendment; (2) zero-violation result and the cost-units finding; (3) joint-target status and the population-0 shortfall; (4) stress / N-1 interpretation (planning vs control); (5) transfer gap → guardrail design; (6) Month-6 plan: consolidation, IEEE-118 feasibility, 5-seed confirmation, sensitivity to the flexibility spread and to the comfort tolerances.

**Points to record.** Acceptance of the amended fairness basis and of the population rule; whether the peak target should be assessed per population or on the mean; agreement on the Month-6 compute budget (5 seeds × 500k × 4 learners ≈ 6 h laptop); whether the workshop paper (m6-t11) is built on the safety result.

**Outcome (to be filled after the meeting).**
