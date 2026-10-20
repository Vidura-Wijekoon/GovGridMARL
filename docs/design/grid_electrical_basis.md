# Electrical basis of the IEEE-14 environment (m2-t11 addendum)

*Written for two readers at once: the electrical engineer who wants to see the power-flow method, the balance and the reasons behind each modelling choice, and the software engineer who wants to know which line of code each number comes from. All figures are produced by `scripts/grid_analysis.py` (log in `results/logs/grid_analysis_*.log`, data in `results/tables/grid_analysis.json`) and are shown live in the dashboard's "Power-flow analysis" and "Why these buses" panels.*

## 1. The network in numbers

| quantity | value | where in code |
|---|---|---|
| test system | IEEE 14-bus (pandapower `case14`, from MATPOWER) | `envs/govgrid_envs/ieee14.py::load_ieee14` |
| base power | S_base = 100 MVA (all per-unit quantities refer to it) | `net.sn_mva` |
| buses | 14: 1 slack, 4 PV (generator) buses, 9 pure PQ buses; 11 buses carry load | `grid_analysis.json["buses"]` |
| branches | 15 lines + 5 transformers = 20 | `net.line`, `net.trafo` |
| generation | slack B1 (V = 1.06 pu, θ = 0), B2 fixed 40 MW at 1.045 pu, B3 / B6 / B8 synchronous condensers (0 MW, voltage set-points 1.01 / 1.07 / 1.09 pu) | `net.ext_grid`, `net.gen` |
| shunt | 19 Mvar capacitor at B9 (−21.18 Mvar at solved voltage) | `net.shunt` |
| nominal load | 259.0 MW / 73.5 Mvar | `net.load` |

**Voltage levels.** pandapower's conversion labels B1–B5 as 135 kV, B7 14 kV, B8 12 kV and B6, B9–B14 as 0.208 kV. The last value is an artefact of MATPOWER's `baseKV = 0` for those buses (the literature describes them as the 33 kV / 13.8 kV sub-transmission side). It has **no effect on the per-unit solution** — voltages, angles, losses and loadings are computed in per-unit on S_base and the branch impedances — but it does mean that ampere values on the LV side should be read as per-unit currents scaled by an arbitrary base. That is one more reason the thermal constraint is defined relative to the *nominal-case* current of each branch rather than to an absolute ampere rating (§4).

## 2. How the power flow is solved (Newton–Raphson)

At every environment step the network is solved as a full AC power flow. Unknowns: the voltage angle θ at the 13 non-slack buses and the voltage magnitude |V| at the 11 PQ buses (24 unknowns; PV buses hold |V| at their set-point and supply whatever Q is needed within limits). The equations are the nodal active/reactive power balances

P_i(V, θ) − P_i^spec = 0,  Q_i(V, θ) − Q_i^spec = 0,

with P_i = |V_i| Σ_k |V_k| (G_ik cos θ_ik + B_ik sin θ_ik) and Q_i = |V_i| Σ_k |V_k| (G_ik sin θ_ik − B_ik cos θ_ik) from the bus admittance matrix Y = G + jB. Newton–Raphson linearises with the Jacobian [∂P/∂θ ∂P/∂|V|; ∂Q/∂θ ∂Q/∂|V|] and iterates until the largest mismatch is below 10⁻⁸ MVA.

Measured on the nominal case (flat start): **3 iterations**, and the balance closes to solver precision:

| | P (MW) | Q (Mvar) |
|---|---:|---:|
| generation (slack + gens) | 272.39 | 82.44 |
| load | 259.00 | 73.50 |
| series losses (I²R, I²X) | 13.39 | 30.12 |
| shunt capacitor | — | −21.18 |
| **mismatch** | −5.6 × 10⁻⁹ | −1.6 × 10⁻⁸ |

Losses are 4.9 % of generation; the slack bus B1 supplies 232 MW (85 %) because case14 fixes only B2 at 40 MW. That is why, in the energy-share scenarios (§5), renewables displace slack generation first.

**Two solvers, one result.** The training loop uses LightSim2Grid's C++ Newton–Raphson (same equations, sparse KLU factorisation, ≈ 0.1 ms per solve on this network); the analysis and the dashboard's per-hour readout use pandapower's Python implementation (≈ 20 ms), which also exposes the iteration count. `tests/test_env.py::test_backends_agree` checks that both give the same voltages to 10⁻⁶ pu. Divergence (no solution within 30 iterations) is treated as a blackout proxy by the environment.

## 3. Household clusters: from homes to MW

| step | value | code |
|---|---|---|
| synthetic profile per home | mean 0.35 kW, aggregate evening peak 0.83 kW/home | `data/synthetic.py` |
| homes per agent | 2,000 | `homes_per_agent` |
| cluster demand | 0.70 MW mean, 1.66 MW peak | `served_kw × homes / 1000` in `_solve_grid` |
| 14 clusters | 23.2 MW residential peak | `episode_summary()["base_peak_mw"]` |
| background (case14 loads × 0.85–1.10 diurnal factor) | 284 MW peak | `background_shape()` |
| residential share of system peak | ≈ 8 % | — |
| reactive power | Q = P · tan(acos 0.95) = 0.329 P, lagging | `Ieee14PowerFlow.solve` |

2,000 homes is the smallest cluster size at which residential demand response moves branch loadings by whole percentage points on a transmission-level test case while staying a minority of the system load (so the background still sets the operating point). Sensitivity to this choice is part of the Month-11 analysis.

## 4. Thermal ratings

`case14` ships `max_i_ka = 42.3 kA` on every line and 9,900 MVA transformers — limits that can never bind. The environment therefore defines each branch rating as **1.5 × its nominal-case current (lines) or apparent power (transformers)**, i.e. every branch sits at 66.7 % loading on the nominal day. Ratings are stored on the solver object, not in `net.trafo.sn_mva` (pandapower defines `vk_percent` relative to `sn_mva`; writing there would change the transformer impedance). The factor is a config knob (`rating_factor`) and is swept in the sensitivity analysis.

## 5. Why the clusters sit where they do

Clusters are attached round-robin to the load buses in the order **B9, B10, B11, B12, B13, B14, B5, B4, B2, B3, B6** — transformer-fed buses first. The evidence is a numerical sensitivity test: inject +5 MW (one cluster at peak, times three) at each load bus and record the change in local voltage, in the most-loaded branch and in system losses.

| bus | side | ΔV local (pu / 5 MW) | Δ max loading | Δ losses (MW) | priority |
|---|---|---:|---:|---:|---:|
| B14 | LV, behind T4-9 | −0.0083 | +0.257 | +0.747 | 6 |
| B13 | LV | −0.0037 | +0.322 | +0.630 | 5 |
| B12 | LV | −0.0067 | +0.257 | +0.609 | 4 |
| B10 | LV | −0.0046 | +0.355 | +0.604 | 2 |
| B9 | LV | −0.0031 | +0.160 | +0.576 | 1 |
| B11 | LV | −0.0048 | +0.251 | +0.574 | 3 |
| B4 | 135 kV | −0.0015 | +0.025 | +0.571 | 8 |
| B5 | 135 kV | −0.0015 | +0.022 | +0.486 | 7 |
| B3 | 135 kV (94 MW load) | 0.0000 | +0.060 | +0.700 | 10 |
| B2 | 135 kV (gen) | 0.0000 | +0.019 | +0.280 | 9 |
| B6 | gen bus | 0.0000 | +0.056 | +0.485 | 11 |

Reading the table: the six LV buses (B9–B14) show 5–10× the voltage sensitivity and 5–15× the branch-loading sensitivity of the 135 kV buses, because every MW drawn there flows through the three transformers T4-7, T4-9, T5-6 and the radial 9-10-11 / 9-14-13 loops. Household flexibility at those buses therefore has the largest effect on the constraints that Phase 2 will enforce. Generator buses (B2, B3, B6) hold their voltage by construction, so a cluster there would never see a voltage signal. Five of the six most loss-sensitive buses are the six chosen ones (B3 is the exception: its huge fixed load makes any addition costly in losses, but it is a stiff 135 kV generator bus).

## 6. Energy share: generation mix as a scenario

The environment now carries a generation-mix scenario (`renewable_share`, `wind_fraction`). `renewable_share` is the fraction of the *nominal daily system energy* (background + residential, ≈ 6,600 MWh) supplied by renewables; it is split into a transmission-connected wind farm at B2 (night-peaking profile, log-normal day factor) and rooftop PV distributed over the 11 load buses in proportion to their load (clear-sky bell). Both are modelled as unity-power-factor static generators, so they displace slack generation and change flows and voltages. Under no control on the nominal day:

| share | wind/PV | consumed locally | surplus (MWh) | net-import peak (MW) | losses (MWh) | max loading | violations |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | – | 0 % | 0 | 269 | 319 | 0.90 | 0 |
| 0.2 | 50/50 | 20 % | 0 | 219 | 225 | 0.90 | 0 |
| 0.4 | 50/50 | 40 % | 0 | 196 | 168 | 0.90 | 0 |
| 0.6 | 50/50 | 54 % | 345 | 172 | 145 | 0.90 | 0 |
| 0.8 | 50/50 | 65 % | 933 | 149 | 155 | 0.90 | 0 |
| 0.8 | 80/20 (wind-heavy) | 77 % | 202 | 108 | 163 | 0.92 | 0 |
| 0.8 | 20/80 (PV-heavy) | 51 % | 1,799 | 194 | 256 | **1.32** | 0.12 |

Three things the table shows, and which the "grid as a whole" view could not: (1) up to ≈ 40 % share every renewable MWh is consumed and losses fall (generation moves closer to load); (2) beyond that the *shape* matters more than the share — PV-heavy mixes produce large midday surpluses, reverse flows on the LV loops and thermal violations, while a wind-heavy 80 % mix is absorbed almost entirely; (3) the demand-response objective changes with the mix: with high PV the useful action is to shift load *into* midday rather than only away from the evening. Phase 1 reports these KPIs for the trained policies (`experiments/evaluate.py --energy-share`); Phase 2 adds the net-load signal to the observation and prices the resulting constraints, and the fairness question becomes who gets to absorb the cheap midday energy.
