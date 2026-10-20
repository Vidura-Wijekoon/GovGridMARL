# Phase-2 acceptance thresholds (task m4-t10)

Fixed on 2026-09-22, before the fairness-weight sweep was evaluated, from proposal §6.7 and `configs/eval/default.yaml`. Implemented in `govgridmarl.evaluation.targets` (`check_targets`) and used by `experiments/pareto.py` (Month 4) and `experiments/evaluate_phase2.py` (Month 5).

| KPI | Threshold | Definition (evaluation protocol §1) | Checked on |
|---|---|---|---|
| Peak reduction | **≥ 15 %** | residential aggregate peak vs the same day under no control | 20 held-out days, mean over ≥ 3 training seeds |
| Jain index | **≥ 0.85** | Jain over per-household **kWh** curtailment burden $B^i_T$ | same |
| Safety violations | **0** | share of hours with any positive constraint cost (voltage, thermal, comfort; power-flow divergence counts as all three) | same, plus the stress set reported separately |

## Rules

1. All three must hold **simultaneously on one policy** (one checkpoint, one configuration). A configuration that reaches Jain 0.9 at 12 % peak reduction does not pass, nor does one that reaches 16 % at Jain 0.8.
2. ~~The **kWh basis** is the gate metric~~ **Amended 2026-10-20 (see §Amendment below):** the gate metric is Jain on the **relative burden** $B^i_T/F^i_T$ (share of own flexible energy). The **kWh basis** stays in every table because it was the proposal's definition and the Phase-1 numbers were reported in it; the two disagree by design (`docs/design/reward_and_constraints.md`, `fairness/metrics.py`).
3. Thresholds are minima, not optimisation targets: among configurations inside the region the one with the **smallest fairness weight** (least distortion of the efficiency objective) is the operating point (m4-t11), ties broken by peak reduction.
4. "Zero violations" is evaluated on the nominal protocol days. On the stress set (demand × 1.3, background × 1.25) and under N-1 contingencies (m5-t12) the violation rate is **reported**, with the target that the constrained learner has strictly fewer violations than the unconstrained one at equal peak reduction.
5. Any change to these thresholds after this date is recorded in `CHANGELOG.md` with the reason.

## Fallback (pre-registered in the scope note)

If no configuration enters the region at laptop scale, the full trade-off curve (`results/figures/pareto_front.png`) is reported against the thresholds and the thesis discusses the gap; the thresholds themselves are not moved.

## Amendment of 2026-10-20 — gate basis changed from kWh to relative burden (rule 5)

**Evidence.** `scripts/fairness_feasibility.py` (log `results/logs/fairness_feasibility_20261020.log`, table `results/tables/fairness_feasibility.csv`) runs *scripted* curtailment policies on the 20 protocol days, from "everyone curtails the same fraction" to "everyone curtails the same kWh", at three curtailment depths. On this population the flexible energy per home spans 0.81–5.91 kWh/day (ratio 7.3, by construction of the income groups: 15/25/35 % flexible share × 4.9/7.4/11.2 kWh). Consequences:

| policy family | best peak reduction at Jain$_{kWh}$ ≥ 0.85 | Jain$_{kWh}$ at 15 % peak reduction |
|---|---|---|
| any equal-fraction / equal-kWh blend | **9.0 %** | 0.735 (equal fraction, every home at the 60 % cap, 25 % comfort violations) |
| learned MAPPO, λ = 0 … 8 | – | 0.648 → 0.694 (λ = 8, p = 0.03 vs λ = 0) at 15.6–15.7 % |

Equal kWh burden can only be reached by holding every household to the *least* flexible one (0.81 kWh/day), which forfeits two thirds of the available flexibility; the pre-registered region (≥ 15 % **and** Jain$_{kWh}$ ≥ 0.85) is therefore **empty for every controller**, learned or scripted. The learners' kWh Jain (0.65–0.69) sits where proportional burden sharing puts it (0.735 for the exact equal-fraction rule).

**Decision.** The Jain target keeps its value (0.85) and its role, but is evaluated on the relative burden $B^i_T/F^i_T$ — each household giving the same share of what it can give. This is the ability-to-pay reading of energy justice used in the review (Area 3) and it is the quantity the comfort constraint already bounds per group. Jain$_{kWh}$, the Gini coefficient, the income-group gap ratio and the vulnerable/other burden ratio are reported alongside so that the thesis can show the price of kWh-equality (the vulnerable/other burden ratio *rises* from 0.32 to 0.36 as λ increases — kWh-equality asks the vulnerable for more).

**Consequence for the gate.** `experiments/pareto.py --gate-basis fraction` (default) and `experiments/evaluate_phase2.py` check Jain$_{frac}$; the joint Lagrangian agent's fairness constraint (`env.fairness_target_basis = fraction`) is on the same basis. Recorded in CHANGELOG 0.4.0.
