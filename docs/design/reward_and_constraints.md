# Reward and constraint design (task m2-t8)

Companion to `cmomg_formulation.md` §3–4; this note explains the *choices* and gives the numbers used to check them (`scripts/validate_env.py`, `tests/test_env.py`).

## 1. Reward terms

| term | formula (per step) | weight (Phase 1) | why this form |
|---|---|---|---|
| peak | $-(\bar p_t/p_{ref})^2$ | 1.0 | A quadratic in aggregate load is the standard smooth proxy for a max-over-time peak objective: it penalises the evening hours far more than the night, so agents learn *when* to curtail without an episode-end-only signal. |
| cost | $-\pi_t\bar p_t\Delta t/C_{ref}$ | 0.5 | Energy bill under the ToU tariff; $C_{ref}$ = mean hourly no-control bill so the term is O(1). |
| comfort | $-10\,\kappa^2\,(f^i_t/b^i_t)$ | 0.1 | Convex in the curtailment fraction (mild curtailment is nearly free, deep curtailment hurts), scaled by the household's flexible share so low-flexibility homes are not over-penalised for the same fraction. |
| fairness | $-(1-\mathrm{Jain}(B_t))$ | 0 | Kept at zero in Phase 1 (efficiency-only baseline). Phase 2 turns it on (0.5) and compares with the Lagrangian formulation. |
| safety | $-(10c_V+c_\ell+c^i_C)$ | 0 | Available for reward-shaping ablations; the thesis' primary mechanism is the Lagrangian on measured costs, not shaping. |

Shared terms (peak, cost, fairness) are identical for all agents → a cooperative game; the comfort term is individual → mild mixed-motive tension that makes fairness non-trivial.

**Magnitude check (validation log, 18:00 on a nominal day, N=14):** no-control joint reward −36.96 vs moderate curtailment (level 2 for all) −31.33 — curtailing at the peak is rewarded; at night the same action is penalised by the comfort term with almost no peak benefit.

## 2. Constraint costs (always measured, never enforced in Phase 1)

| cost | formula | bound | Phase-1 observation |
|---|---|---|---|
| voltage $c_V$ | band violation of min/max bus voltage | [0.95, 1.10] pu | never fires on case14 (PV buses hold voltage); retained for IEEE-118 |
| thermal $c_\ell$ | $\max(0,\max\ell-1)$ | rating = 1.5× nominal flow | fires on stressed days (`day_scale` 1.6, background 1.25 → max loading 1.33); DR relieves it (→ 1.26 under full curtailment) |
| comfort $c^i_C$ | $\max(0, B^i_t/F^i_t-\theta_{g_i})$ | $\theta$ = 0.30 / 0.40 / 0.50 | fires under permanent maximal curtailment; silent under no-control |
| blackout proxy | power-flow divergence | — | treated as $c_V=1$, $c_\ell=2$ |

`info["violation"]` is true if any cost is positive; `episode_summary()["violation_rate"]` is the fraction of steps with a violation — the Phase-2 "zero safety violations" KPI.

## 3. Episode KPIs (`episode_summary()`)

* `peak_reduction_pct` — $(\max_t P^{base}_t-\max_t P^{res}_t)/\max_t P^{base}_t$ on the residential aggregate (MW). Target ≥ 15 % vs rule-based DR.
* `jain_index` — Jain over cumulative curtailment burden $B^i_T$ (kWh/home). Target ≥ 0.85. Note that "no DR" trivially gives Jain = 1; fairness is only meaningful jointly with peak reduction.
* `cost_total` — bill over all homes; `energy_res_mwh` vs `energy_base_mwh` shows shed vs shifted energy.
* `violations`, `violation_rate`, `burden_by_group_kwh` (low/mid/high — the energy-justice view).

## 4. Sanity properties (enforced by tests)

1. All-zero actions reproduce the baseline exactly (0 % reduction, Jain 1, no violations on nominal days).
2. Maximal curtailment reduces peak (≈ 15 %) and energy and triggers the comfort constraint.
3. One agent carrying all curtailment gives Jain = 1/N.
4. With `rebound = 1` energy is conserved up to the horizon boundary.
5. LightSim2Grid and pandapower backends agree to 1e-6 pu in voltage.
6. Seeded episodes are bit-reproducible.
