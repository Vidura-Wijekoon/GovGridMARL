# Chapter 5 — Safety: constrained learning (draft, task m5-t14)

*Scope: constraint formalisation (m5-t1/t2), PPO-Lagrangian and its MAPPO integration (m5-t3/t4), multiplier tuning and PID-Lagrangian (m5-t5/t6), training and results (m5-t7 … m5-t13). Numbers from `docs/reports/safety_results.md`; regenerate with `make phase2-m5 evaluate-phase2`.*

## 5.1 Constraints

The constrained Markov game of Chapter 3 is solved with three safety constraints and one objective constraint, each a non-negative per-step cost that is zero exactly when the constraint holds (`govgridmarl.safety`): bus-voltage band 0.95–1.10 pu, branch loading ≤ rating, household comfort $B^i_t/F^i_t\le\theta_{g_i}$ with $\theta=(0.30,0.40,0.50)$, and — for the joint agent — Jain of relative burden ≥ 0.85. Power-flow divergence (the no-blackout condition) is charged as a voltage cost of 1 and a loading cost of 2. All budgets are zero: the Phase-2 gate asks for zero violating hours on the protocol days, not for an average. Frequency deviation is out of scope because the quasi-static AC power flow has no frequency dynamics (scope note).

## 5.2 Lagrangian learners

Constraint costs enter the learner through one *cost critic per constraint* on the same input as the reward critic (the global state for MAPPO), cost advantages are estimated by GAE, and the clipped-PPO step uses the penalised advantage $A=(A_R-\sum_k\lambda_kA_{C,k})/(1+\sum_k\lambda_k)$ (Ray et al., 2019). The multipliers are updated after each policy step from the batch's mean episode costs, either by dual ascent, $\lambda_k\leftarrow[\lambda_k+\eta(\hat J_{C,k}-d_k)]_+$, or by the PID rule of Stooke et al. (2020). Two input regimes are compared: the centralised PPO-Lagrangian controller (reference) and MAPPO-Lagrangian with local actors — the thesis learner, since actors must run on household edge devices in the privacy layer of Chapter 6. The joint agent of §5.5 is the same class with the fairness constraint added.

## 5.3 What the cost signal has to be

The first batch used the magnitude costs of §5.1. Their episode sums on IEEE-14 are of order 0.03, so the dual variables move by 10⁻³ per update and never bind within 120k steps; violations stayed at 21 % of hours whatever the rate, and raising the rate to 1.0 only trades peak reduction for an over-large multiplier (Table 5.1). Replacing the magnitude by the *indicator* of a violation — so that an episode cost counts violating hours, as Safety-Gym-style budgets do — puts the constraint error on the scale of the reward advantage. With indicator costs every learner drives training violations below 1 % within 10–50 updates and the multipliers settle below 1 (Figure 5.2). Dual ascent and PID differ only in the speed of the initial rise; no oscillation occurs because the constraint error is monotone, so the derivative term is idle. This is the finding behind task m5-t5: the learning rate was never the problem, the units of the cost were.

## 5.4 Results

On the 20 protocol days of each training population (Chapter 3, protocol §7), all indicator-cost learners — PPO-Lagrangian, MAPPO-Lagrangian, MAPPO-PID-Lagrangian and the joint agent — achieve **zero violating hours** under both the deterministic and the stochastic policy, against 25 % for the unconstrained learner with the same reward (Welch p < 10⁻⁴, Cliff's δ = −1), while peak reduction is unchanged (14.6 vs 14.7 %, p ≥ 0.89) and the relative-burden Jain stays at 0.94–0.95. The violations of the unconstrained policy were comfort violations of the low-tolerance (low-income) clusters; the constrained policies stop curtailing those clusters at their daily bound and recover the lost peak reduction from the others. Figure 5.1 shows the training curves, Figure 5.3 the benchmark.

Under stress (demand × 1.3, background × 1.25) and after each of the three binding N-1 line outages, thermal overloads are identical for every policy including no control — residential flexibility is about 8 % of the system peak and cannot relieve a 30–40 % branch overload — whereas comfort violations fall from 0.23–0.28 to 0 (N-1) or 0.09 (peak-demand day) at equal peak reduction (Table 5.3). On households never seen in training (population 42) the learned constraints transfer only partly (0.21 violating hours vs 0.30 unconstrained): the policy has learned its population's comfort bounds, not the rule. This is the empirical motivation for the deterministic runtime guardrail of Chapter 8.

## 5.5 Joint targets

The joint agent meets the fairness target (0.951 ≥ 0.85) and the safety target (0.000) on every population, and the peak target (≥ 15 %) on two of three (15.4, 15.3, 13.0 %). The shortfall on population 0 is shared by every learner including the Phase-1 baseline (13.2 %), so it is an efficiency matter rather than a cost of the constraints; the feasibility study shows that a comfort-respecting policy concentrated on 17–19 h reaches 19.3 % on the same days. Closing that gap — longer runs, five seeds, the gap-form shaping, entropy and learning-rate tuning — is the first item of the Month-6 consolidation.

## 5.6 Figures and tables

Figure 5.1 `violation_curves.png`; Figure 5.2 `multiplier_dynamics.png` and `multiplier_lr_sweep.png`; Figure 5.3 `safety_benchmark.png`; Figure 5.4 `stress_violations.png`. Table 5.1 multiplier tuning; Table 5.2 ablation; Table 5.3 stress and N-1; Table 5.4 joint targets (`phase2_targets.csv`).
