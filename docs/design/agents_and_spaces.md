# Agents, observation and action spaces (task m2-t7)

Executable spec for `GridMARLEnv` (PettingZoo `ParallelEnv`, `metadata.name = govgrid_marl_v0`). Every row below maps to one index in the observation vector so that SHAP attributions (Phase 3) have named features.

## 1. Agent roster

| Role | Count | Learning? (Phase 1) | Where |
|---|---|---|---|
| Household cluster `household_i` | N = 14 (config) | **Yes** | `possible_agents` |
| Aggregator | 1 | No — scripted: computes aggregate load, network state, burden statistics | inside `step()` |
| DSO / grid operator | 1 | No — scripted: publishes ToU tariff, PV forecast, thermal/voltage limits | `price`, `pv`, constraint bounds |

Each household cluster stands for `homes_per_agent = 2000` homes of one income group. Groups are assigned by tercile (`assign_groups`): 34 % low, 33 % mid, 33 % high. Clusters are attached to IEEE-14 load buses round-robin with the weak LV buses first: `[8, 9, 10, 11, 12, 13, 4, 3, 1, 2, 5, 8, 9, 10]` (0-based pandapower indices).

## 2. Observation space — `Box(-5, 5, (17,), float32)` per household

| idx | feature | definition | normalisation |
|---|---|---|---|
| 0 | `sin_t` | $\sin(2\pi t/T)$ | — |
| 1 | `cos_t` | $\cos(2\pi t/T)$ | — |
| 2 | `own_inflexible` | $(b^i_t-f^i_t)$ kW/home | ÷ $p_{ref}$ |
| 3 | `own_flexible` | $f^i_t$ kW/home | ÷ $p_{ref}$ |
| 4 | `own_forecast` | $b^i_{t+1}$ | ÷ $p_{ref}$ |
| 5 | `price_now` | $\pi_t$ | ÷ 0.40 |
| 6 | `price_next` | $\pi_{t+1}$ | ÷ 0.40 |
| 7 | `pv` | PV availability | ÷ max PV |
| 8 | `bus_voltage` | $(V_{\beta(i),t}-1)\times 20$ | centred |
| 9 | `max_loading` | $\max_{br}\ell_{br,t}$ (fraction of rating) | — |
| 10 | `agg_load` | $\bar p_{t-1}/p_{ref}$ | — |
| 11 | `own_burden` | $B^i_t/F^i_t$ | fraction |
| 12 | `burden_gap` | $B^i_t/F^i_t-\overline{B_t/F_t}$ | fairness signal |
| 13 | `prev_action` | $\kappa_{a^i_{t-1}}/\bar\kappa$ | [0, 1] |
| 14–16 | `group_onehot` | low / mid / high | one-hot |

$p_{ref}$ = no-control aggregate peak per home (dataset constant). Values are clipped to [−5, 5].

**Privacy boundary (for Phase 3):** indices 2–4, 11 and 13 are *private* household data; 5–10 are public DSO/aggregator broadcasts; 12 requires only the population mean burden, which the aggregator can publish without per-household data.

## 3. Global state — `state()` ∈ ℝ^{17N+3}

Concatenation of all local observations followed by `[(V_min−1)·20, (V_max−1)·20, max_loading]`. Used by the centralised critic (MAPPO) and the centralised controller (PPO); never by MAPPO/IPPO actors.

## 4. Action space — `Discrete(n_levels = 5)` per household

| level | curtailment of flexible load |
|---|---|
| 0 | 0 % |
| 1 | 15 % |
| 2 | 30 % |
| 3 | 45 % |
| 4 | 60 % (`max_curtail`) |

Half of curtailed energy rebounds over the next two steps (`rebound = 0.5`, `rebound_steps = 2`). Discrete levels were chosen over a continuous set-point because (a) real DR programmes offer tiered participation, (b) they make PPO's categorical policy directly comparable across PPO/IPPO/MAPPO, and (c) SHAP on a categorical head yields per-level attributions.

## 5. Extension points

* **Aggregator / DSO as learners**: add roles to `possible_agents` with their own spaces (tariff level, curtailment request) — the trainer is role-agnostic because all agents share the `(obs, state, action)` interface.
* **Continuous actions**: swap `Discrete` for `Box(0, max_curtail)` and the categorical head for a Beta policy; nothing else changes.
* **IEEE-118** (Month 6): `household_bus_map` and `Ieee14PowerFlow` are the only case-specific pieces.
