# Chapter 4 — Fairness: multi-objective demand response (draft, task m4-t14)

*Scope: the fairness formulation (m4-t1 … m4-t4), the weighted-sum and MOPPO learners (m4-t5 … m4-t9), the acceptance thresholds (m4-t10) and the Month-4 results (m4-t11 … m4-t13). Numbers from `results/tables/pareto_*.csv` and `docs/reports/fairness_tradeoff.md`; regenerate with `make phase2-m4 pareto`.*

## 4.1 What is being equalised

Residential demand response redistributes discomfort. For household cluster $i$ the environment accumulates the curtailed energy $B^i_T$ (kWh per home per day) and the flexible energy $F^i_T$ that was available. Two fairness notions follow:

* **absolute burden** — Jain's index over $B^i_T$: everyone sheds the same energy (the proposal's definition, §6.7);
* **relative burden** — Jain's index over $B^i_T/F^i_T$: everyone gives the same share of what it can give (the "ability-to-pay" reading of the energy-justice literature, Chapter 2, Area 3).

The population is heterogeneous by design (Chapter 3): low/mid/high-income clusters have 15/25/35 % flexible load on 4.9/7.4/11.2 kWh/day, so $F^i_T$ ranges from 0.81 to 5.91 kWh (ratio 7.3). Section 4.5 shows that this spread makes the two notions incompatible with the efficiency target, which is itself a finding.

Households additionally carry a *vulnerability* flag (all low-income clusters plus 20 % of mid-income clusters as a medical/elderly proxy); it never changes the physics and is used only to report the burden carried by vulnerable homes relative to the rest. Together with the Gini coefficient and the max/min ratio of income-group mean burdens these are the equity metrics the governance layer publishes (`govgridmarl.fairness.metrics`).

## 4.2 Fairness as an objective

The reward of Chapter 3 is extended by a fairness term with weight $w_{fair}$ and two forms (`fairness_form`):

$$r^i_t = r^{i,\mathrm{eff}}_t + w_{comf}\,J^i_{comf} - w_{fair}\,\frac{\mathcal F^i_t}{T},\qquad
\mathcal F^i_t = \begin{cases}1-\mathrm{Jain}(B_t) & \text{(shared, "Jain" form)}\\ \big[s^i_t-\bar s_t\big]_+ & \text{(individual, "gap" form)}\end{cases}$$

where $s^i_t$ is the burden on the chosen basis (`fairness_basis`: kWh or relative). Dividing by the horizon makes one day at Jain 0.65 cost $0.35\,w_{fair}$ — the order of the daily efficiency return (≈ 1.25) — so the sweep $w_{fair}\in[0,8]$ spans "ignored" to "dominant". The environment also returns the *unweighted* objective vector (efficiency, fairness, comfort, safety) per agent so that the scalar reward is exactly its weighted sum; this is what makes weighted-sum scalarisation and MOPPO comparable on the same signal.

## 4.3 Learners

**Weighted-sum MAPPO** (m4-t4/t5): the Phase-1 learner trained once per $w_{fair}\in\{0,0.5,1,2,4,8\}$; six models trace the front. Linear scalarisation recovers only the convex part of a front (Das & Dennis, 1997), which motivates the second learner.

**MOPPO** (m4-t8): one preference-conditioned actor–critic. The preference $\lambda\sim U(0,8)$ (30 % of episodes on the evaluation grid) is drawn per episode and environment, enters actor and critic as $\lambda/\lambda_{max}$, the critic has one head per objective, advantages are estimated per objective and scalarised as $A_{eff}+\lambda A_{fair}$ before the clipped-PPO step. At evaluation the preference is fixed; sweeping it traces the front with a single model (Yang et al., 2019; Abels et al., 2019).

## 4.4 Acceptance thresholds and their amendment

Pre-registered on 2026-09-22 (`docs/design/acceptance_thresholds.md`): peak reduction ≥ 15 %, Jain ≥ 0.85 and zero safety violations, all on one policy, mean over seeds on the 20 protocol days. The Jain basis was amended on 2026-10-20 from absolute to relative burden after the feasibility study of §4.5; the amendment, its evidence and the rule that absolute-burden Jain stays in every table are recorded there and in the changelog.

## 4.5 Results

**Feasibility.** Scripted policies that blend "equal fraction" and "equal kWh" curtailment at the peak hours bound what any controller can do (Table 4.1, `fairness_feasibility.csv`): at Jain$_{kWh}$ ≥ 0.85 the best peak reduction is 9.0 %; 15 % is reached only with every home at the 60 % cap, where Jain$_{kWh}$ = 0.735 and comfort is violated in 25 % of hours. The pre-registered region is empty on this population for every policy.

**Sweep.** Policies are evaluated on the population they were trained on (protocol §7). Peak reduction is flat in $w_{fair}$ (14.7 % at every weight; ≈ 13 % on population 0 and ≈ 15.4 % on populations 1–2 for every learner), 3.5 pp above the rule-based reference on the same populations (11.2 %). Jain$_{kWh}$ rises monotonically from 0.589 to 0.652 and the income-group gap falls from 9.4 to 7.3, while the violation rate rises from 0.23 to 0.29 and the vulnerable/other burden ratio from 0.15 to 0.18: kWh-equality is bought by asking the least flexible — and vulnerable — clusters for more. Relative-burden Jain is 0.949–0.980 at every weight: shared-parameter policies already distribute burden proportionally to flexibility, as does the rule-based reference (0.999).

**Operating point** (m4-t11). The Jain target (relative basis) holds at every weight; the peak target holds on two of three populations and fails on population 0 for every learner including $w_{fair}=0$, so the shortfall is a Phase-1 efficiency matter. By the pre-registered rule the operating point is $w_{fair}=0$; $w_{fair}=2$ is carried into Month 5 as reward shaping because it improves kWh equity (group gap 9.4 → 8.5) at no measurable cost; weights ≥ 4 are rejected for the violation increase.

**Weighted-sum vs MOPPO** (m4-t9). One MOPPO model reproduces the six-model front (hypervolume 6.97 vs 7.09, four non-dominated points each) but is insensitive to its preference input — all six evaluation preferences give the same policy. The fairness advantage is too small relative to the efficiency advantage for the conditioning to matter at this scale; MOPPO therefore gives Pareto *coverage* cheaply but not Pareto *control*. The weighted-sum sweep stays the reference.

**Form ablation.** The individual gap form gives the best peak reduction (14.8 %) and the lowest violation rate (0.22) at $w_{fair}=2$, because an agent is never charged for others' behaviour; it is the form recommended for the constrained agent of Chapter 5.

## 4.6 Figures and tables

Figure 4.1 `pareto_front.png` — fronts on both bases with the scripted feasibility points and the acceptance region; Figure 4.2 `fairness_tradeoff.png` — KPIs along $w_{fair}$ for both learners; Figure 4.3 `burden_by_group_fair.png` — per-household burden by income group at $w_{fair}$ = 0, 2, 8; Table 4.1 feasibility bound; Table 4.2 sweep with 95 % CIs; Table 4.3 Pareto comparison.
