# MARL-DR Framework Comparison Matrix

Tabulates existing MARL demand-response frameworks against operational metrics and governance
gaps. Source: 105-paper literature database (`Literature_Review/GovGridMARL_Literature_Review_Database.xlsx`);
`[#N]` = row ID in the Paper Database sheet. Y = explicit component, P = partial/implicit, N = absent.
Full formatted version: `Literature_Review/GovGridMARL_Framework_Comparison.xlsx`.

| Framework | Algorithm | Metrics reported | Fair | Priv | XAI | Safe | Gov |
|---|---|---|---|---|---|---|---|
| MAPPO fast-timescale DR [#2, 2023] | MAPPO, localized rewards | Freq support, 100s of loads | N | N | N | N | N |
| MARL-iDR [#16, 2023] | Decentralized incentive MARL | PAR -14.5% | N | P | N | N | N |
| Urban MARL [#13, 2025] | MARL coordination, 1247 bldgs | Peak -23.4%, cost -18.7% | N | N | N | N | N |
| Temporal DR [#3, 2022] | MA deep RL, price signals | Scheduling cost | N | N | N | N | N |
| Price-setting MARL [#4, 2020] | Model-free deep RL | Cost, participation | N | N | N | N | N |
| GridLearn [#6, 2022] | MARL + power flow | Undervoltage, comfort | N | P | N | P | N |
| MAHTM [#5, 2023] | Hierarchical transactive MARL | Economic coordination | N | N | N | N | N |
| MO-PPO-Lagrangian [#12, 2025] | PPO + Lagrangian | Cost, stability, renewables | N | N | N | P | N |
| MAPPO-ESS [#21, 2025] | MAPPO storage scheduling | Cost -14.7% | N | N | N | N | N |
| CTDE-DDPG EV [#23, 2024] | CTDE-DDPG | PAR, charging cost | N | P | N | N | N |
| Blockchain REM [#24, 2024] | MA DRL + blockchain | Cost, autonomy | N | P | N | N | P |
| Fairness-First DR [#46, 2026] | Heuristic (non-RL) | Jain 0.65->0.92 | Y | N | N | N | N |
| Emergency DR fairness [#49, 2021] | Optimization (non-RL) | Curtailment equity | Y | N | N | N | N |
| Fair-PPO [#53, 2025] | PPO + fairness metrics | Fairness-utility trade-off | Y | N | N | N | N |
| Welfare MORL [#52, 2022] | MORL welfare functions | Pareto fairness | Y | N | N | N | N |
| Safe RL low-carbon [#105, 2024] | Safe RL + Lagrangian | Safety violations, carbon | N | N | N | Y | N |
| **GovGridMARL (this thesis)** | **MAPPO + Lagrangian, CTDE** | **Peak >=15%, Jain >=0.85, eps<=1** | **Y** | **Y** | **Y** | **Y** | **Y** |

## Gap summary (drives contribution C1)

Operationally strong MARL-DR frameworks (#2, #13, #16, #21) have no fairness, privacy, XAI, or
governance components. Fairness-aware controllers exist (#46, #49) but are not learning-based,
while fair-RL methods (#52, #53) are not energy-specific. Privacy appears only as a side-effect
of decentralization, never as calibrated differential privacy with a stated budget. Safety
constraints appear in isolation (#105) and are never combined with fairness in one constrained
optimizer. Runtime governance (policy gates, HITL override, audit logging) and regulatory
alignment are absent from all 16 surveyed frameworks — the intersection GovGridMARL occupies.
