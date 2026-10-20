# Safe RL results on IEEE-14 (tasks m5-t5, m5-t7 … m5-t12)

**Runs** (`scripts/run_phase2_month5.sh`, `run_phase2_month5b.sh`, 2026-10-20; 120k env-steps, 3 seeds = 3 populations): unconstrained MAPPO with the Month-4 reward (λ = 2) · MAPPO-Lagrangian (dual ascent) · MAPPO-PID-Lagrangian · joint agent (safety + fairness constraints) · PPO-Lagrangian reference, each with magnitude costs (first batch) and indicator costs (second batch); dual-ascent rate sweep η ∈ {0.01, 0.05, 0.2, 1.0}. **Evaluation** `experiments/evaluate_phase2.py`: 20 protocol days on the training population, deterministic policy; stochastic policy as a second column; stress set; three N-1 outages; transfer to population 42. Tables `results/tables/safety_ablation.*`, `phase2_targets.csv`, `stress_tests.csv`, `multiplier_lr_sweep.csv`, `n1_screening.csv`, `safety_feasibility.csv`; figures `violation_curves.png`, `multiplier_dynamics.png`, `multiplier_lr_sweep.png`, `stress_violations.png`, `safety_benchmark.png`.

## 1. Multiplier tuning (m5-t5) — why the cost signal matters more than the rate

| cost signal | η | λ_comfort after 156 updates | training violation rate | own-population eval: peak % / violation rate |
|---|---:|---:|---:|---:|
| magnitude (band / overload / excess fraction) | 0.05 | 0.43 | 0.21 | 12.7 / 0.208 (seed 0) |
| indicator (violating hours per episode) | 0.01 | 0.28 | 0.015 | 12.3 / 0.000 |
| indicator | 0.05 | 0.93 | 0.009 | 12.5 / 0.000 |
| indicator | 0.2 | 3.3 | 0.005 | 12.8 / 0.000 |
| indicator | 1.0 | 15.6 | 0.003 | 10.9 / 0.000 |

With magnitude costs an episode's comfort cost is O(0.03) (excess share × hours), so dual ascent at any rate that is stable moves λ by ≈ 10⁻³ per update and the constraint never binds within a run: violations stay at 21 % of hours. Counting violating hours instead (0 … 24 per episode) puts the constraint error on the scale of the reward advantage; λ settles below 1 and violations vanish. η = 1.0 overshoots (λ = 15.6, peak reduction −1.9 pp with no further safety gain); η = 0.2 is the fastest setting without loss; **0.05 with indicator costs** is kept as the default because the PID variant is insensitive to it. No oscillation was observed at any rate (std of λ over the last 30 updates ≤ 0.05): on this problem the constraint error is monotone, so the derivative term of the PID controller is idle and its advantage over dual ascent is the faster initial rise (λ ≈ 0.75 after 10 updates vs 50).

## 2. Ablation on the 20 protocol days (m5-t9; mean ± 95 % CI over 3 populations)

| learner | peak reduction % | Jain relative (gate) | Jain kWh | violation rate | stochastic policy: peak / violations |
|---|---:|---:|---:|---:|---:|
| no control | 0.0 | 1.000 | 1.000 | 0.020 (thermal, population-specific) | — |
| rule-based DR | 11.2 ± 7.1 | 0.999 | 0.675 | 0.276 | — |
| unconstrained MAPPO (λ = 2) | 14.7 ± 3.3 | 0.962 | 0.617 | 0.250 ± 0.000 | 14.7 / 0.251 |
| MAPPO-Lagrangian, magnitude costs | 14.8 ± 3.0 | 0.956 | 0.602 | 0.208 ± 0.003 | 14.7 / 0.201 |
| MAPPO-PID-Lagrangian, magnitude | 14.8 ± 3.2 | 0.958 | 0.603 | 0.206 ± 0.012 | 14.7 / 0.203 |
| PPO-Lagrangian (centralised), indicator | 14.6 ± 4.0 | 0.918 | 0.605 | **0.000** | 14.3 / 0.001 |
| **MAPPO-Lagrangian, indicator** | 14.6 ± 3.1 | 0.943 | 0.587 | **0.000** | 14.6 / 0.000 |
| **MAPPO-PID-Lagrangian, indicator** | 14.6 ± 3.3 | 0.948 | 0.592 | **0.000** | 14.6 / 0.001 |
| **joint (safety + fairness constraints)** | 14.6 ± 3.3 | 0.951 | 0.591 | **0.000** | 14.6 / 0.001 |

Welch tests vs unconstrained: violation rate 0.000 vs 0.250, p < 10⁻⁴, Cliff's δ = −1 for every indicator-cost learner; peak reduction unchanged (p ≥ 0.89). **Zero safety violations are reached at no measurable cost in peak reduction or cost** (m5-t8): the unconstrained policy's violations were all *comfort* violations of the low-tolerance clusters, which the constrained policies avoid by not curtailing those clusters past their daily bound. The wide CI on peak reduction is between-population variance (population 0: 13.0 %, populations 1–2: 15.3–15.5 % for every learner), not between-learner variance.

## 3. Joint targets (m5-t11)

| target | joint agent | verdict |
|---|---:|---|
| Jain ≥ 0.85 (relative burden) | 0.951 | met on all 3 populations |
| zero safety violations | 0.000 (deterministic and stochastic) | met on all 3 populations |
| peak reduction ≥ 15 % | 14.6 % (13.0 / 15.4 / 15.3) | met on 2 of 3 populations |

Two of three targets hold everywhere; the peak target fails on population 0 for **every** learner including the Phase-1 baseline (13.2 % unconstrained), so it is a Phase-1 efficiency matter, not a price of the constraints. The feasibility study (`scripts/safety_feasibility.py`) shows the region is reachable: a scripted policy that respects every comfort cap and concentrates curtailment on 17–19 h achieves 19.3 % peak reduction with 2.5 % violating hours (curtailing from 16 h rebounds into the peak; 14.5 % at exactly zero violations for the 16–20 h variant). The learners spread curtailment over more hours than optimal; longer training (500k steps), the gap-form shaping and entropy/learning-rate tuning are scheduled for the Month-6 consolidation (m6-t4), together with the five-seed confirmation.

## 4. Stress and contingency tests (m5-t12)

| scenario | unconstrained: peak % / violations (thermal, comfort) | joint: peak % / violations (thermal, comfort) |
|---|---:|---:|
| peak demand (demand × 1.3, background × 1.25) | 14.6 / 0.69 (0.53, 0.28) | 14.6 / 0.61 (0.53, 0.09) |
| N-1 line 1 (B1–B5), loading 1.42 | 14.6 / 1.00 (1.00, 0.23) | 14.7 / 1.00 (1.00, 0.00) |
| N-1 line 12 (B10–B11), loading 1.35 | 14.6 / 0.82 (0.79, 0.23) | 14.5 / 0.79 (0.79, 0.00) |
| N-1 line 14 (B13–B14), loading 1.31 | 14.7 / 0.75 (0.60, 0.25) | 14.7 / 0.60 (0.60, 0.00) |
| transfer to population 42 (unseen households) | 15.7 / 0.30 | 15.2 / 0.21 |

Thermal overloads under stress and after an outage are identical for every policy, no-control included: residential flexibility (≈ 8 % of system peak) cannot relieve a 30–40 % branch overload, which is a planning case (rule 4 of the thresholds note; outages with > 200 % loading were excluded as unrelievable, `n1_screening.csv`). The constrained learners keep comfort violations at zero (N-1) or cut them by two thirds (peak-demand day, where the 1.3 × demand pushes cumulative burden over the bound) at equal peak reduction — the stress criterion is met. On **unseen households** the constraints are only partly carried over (0.21 violating hours vs 0.30): the policy has learned the comfort bounds of *its* population, which is the argument for the deterministic runtime guardrail of Month 9 rather than relying on the learned policy alone.

## 5. Figures (m5-t13)

`violation_curves.png` — training violation rate, peak reduction and Jain vs steps per learner; `multiplier_dynamics.png` — λ per constraint vs update for dual ascent, PID and joint (fairness multiplier settles at ≈ 1.8 while the comfort multiplier stays < 1); `multiplier_lr_sweep.png` — Σλ for η ∈ {0.01 … 1.0}; `stress_violations.png`, `safety_benchmark.png`.
