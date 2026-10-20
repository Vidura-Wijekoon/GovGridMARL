# Constrained Multi-Objective Markov Game (CMOMG) — formal model (task m2-t6)

This note fixes the mathematical object that `envs/govgrid_envs/grid_marl_env.py` implements and that Phases 2–3 extend. It follows proposal §5.1 and makes every symbol concrete enough to be checked against code.

## 1. Tuple

The residential demand-response problem is the tuple

$$\mathcal{G} = \langle \mathcal{N},\ \mathcal{S},\ \{\mathcal{O}^i\}_{i\in\mathcal{N}},\ \{\mathcal{A}^i\}_{i\in\mathcal{N}},\ P,\ \mathbf{r},\ \{c_k\}_{k=1}^{K},\ \gamma,\ T \rangle$$

| Symbol | Meaning | Implementation |
|---|---|---|
| $\mathcal{N}=\{1,\dots,N\}$ | household cluster agents (each = `homes_per_agent` homes, one income group $g_i\in\{\text{low},\text{mid},\text{high}\}$) | `possible_agents`, `groups` |
| $\mathcal{S}=\mathcal{S}_{grid}\times\mathcal{S}_{price}\times\mathcal{S}_{pv}\times\prod_i\mathcal{S}^i_{hh}$ | grid state (bus voltages, branch loadings), tariff, PV availability, household states (inflexible/flexible load, pending rebound, cumulative burden) | `state()` (concatenated local observations + grid summary) |
| $\mathcal{O}^i$ | local observation of agent $i$: $\mathbb{R}^{17}$ | `_obs(i)`, see `agents_and_spaces.md` |
| $\mathcal{A}^i=\{0,\dots,L-1\}$ | discrete curtailment level; level $k$ curtails $\kappa_k=\tfrac{k}{L-1}\,\bar\kappa$ of the flexible load | `Discrete(n_levels)`, `max_curtail` |
| $P(s'\mid s,\mathbf{a})$ | transition: load dynamics with rebound + AC power flow | `step()` |
| $\mathbf{r}=(r^1,\dots,r^N)$ | vector of per-agent scalarised rewards (§3) | `rewards` |
| $c_k$ | constraint cost functions, $c_k\ge 0$ (§4) | `info["cost_*"]` |
| $\gamma$, $T$ | discount 0.99, horizon 24 (hourly day) | `gamma`, `horizon` |

The aggregator and DSO of the proposal are *scripted* environment entities in Phase 1 (they publish the tariff, PV forecast and network state); their promotion to learning agents is an explicit extension point (`agents_and_spaces.md` §5).

## 2. Dynamics

Let $b^i_t$ be the base (uncontrolled) demand and $f^i_t\le b^i_t$ its flexible component (kW per home), both drawn per episode from the calibrated dataset with a day-level multiplier $\delta\sim\mathrm{LogNormal}(0,\sigma_d)$. With action $a^i_t=k$:

$$u^i_t = \kappa_k f^i_t \quad(\text{curtailed kW}),\qquad
p^i_t = b^i_t - u^i_t + \rho^i_t,\qquad
\rho^i_{t+j} \mathrel{+}= \frac{\beta}{J}\,u^i_t\quad j=1..J$$

where $\beta\in[0,1]$ is the rebound share and $J$ the rebound length (defaults 0.5 and 2: half of curtailed energy is *shifted*, half is *shed*). The cluster injects $P^i_t = p^i_t\cdot h/1000$ MW at bus $\beta(i)$ (`household_bus_map`) at power factor 0.95; case14 background loads are scaled by $\lambda_t=\lambda\,\sigma_{bg}(t)$. The AC power flow $\Phi$ returns bus voltages $V_t$, branch loadings $\ell_t$ and losses:

$$(V_t,\ \ell_t) = \Phi\big(\{P^i_t\}_i,\ \lambda_t\big).$$

Cumulative burden and flexible energy: $B^i_t=\sum_{\tau\le t}u^i_\tau\Delta t$, $F^i_t=\sum_{\tau\le t}f^i_\tau\Delta t$.

## 3. Objectives

Per-step, with $\bar p_t=\tfrac1N\sum_i p^i_t$ and reference $p_{ref}$ (no-control aggregate peak per home), price $\pi_t$, cost reference $C_{ref}$:

| Objective | Term | Scope |
|---|---|---|
| Peak pressure | $J_{peak}=-(\bar p_t/p_{ref})^2$ | shared |
| Energy cost | $J_{cost}=-\pi_t\,\bar p_t\,\Delta t / C_{ref}$ | shared |
| Comfort (disutility) | $J^i_{comf}=-10\,\kappa_{a^i_t}^2\,(f^i_t/b^i_t)$ | individual |
| Fairness | $J_{fair}=-(1-\mathrm{Jain}(B_t))$ | shared |

The scalarised reward used by the Phase-1 learners is

$$r^i_t = w_{peak}J_{peak} + w_{cost}J_{cost} + w_{comf}J^i_{comf} + w_{fair}J_{fair} + w_{safe}J_{safe},\qquad (w_{peak},w_{cost},w_{comf},w_{fair},w_{safe}) = (1,\,0.5,\,0.1,\,0,\,0).$$

$w_{fair}=w_{safe}=0$ in Phase 1 by design: the baseline is *efficiency-only* MARL, the reference against which the governed variants are measured. Episode-level KPIs (peak reduction %, Jain, cost, violations) are computed in `episode_summary()`.

Multi-objective view (Phase 2, m4-t4/t8): the vector $\mathbf{J}=(J_{peak}+J_{cost},\ J_{fair},\ J_{comf})$ is optimised by weighted-sum scalarisation and by MOPPO; the Pareto front is traced over $w_{fair}$.

## 4. Constraints

$$\mathcal{F}(\pi)=\mathbb{E}\big[\mathrm{Jain}(B_T)\big]\ \ge\ \tau_f = 0.85$$
$$c_V(s_t)=\max(0,\,V_{min}-\min_b V_{b,t})+\max(0,\,\max_b V_{b,t}-V_{max}),\quad V_{min}=0.95,\ V_{max}=1.10$$
$$c_\ell(s_t)=\max\big(0,\,\max_{br}\ell_{br,t}-1\big)$$
$$c^i_C(s_t)=\max\big(0,\,B^i_t/F^i_t-\theta_{g_i}\big),\quad \theta=(0.30,0.40,0.50)\ \text{for (low, mid, high)}$$
$$\text{privacy: no } x^i_t \text{ leaves the household edge; only model updates are transmitted (Phase 3).}$$

Safety constraints are required to hold with $\mathcal{C}_k(\pi)=\mathbb{E}[\sum_t c_k]\le \tau_{s,k}=0$ ("zero violations", Phase-2 milestone). The Lagrangian used from Month 5:

$$\mathcal{L}(\pi,\lambda)=J_{eff}(\pi)-\lambda_f\max(0,\tau_f-\mathcal{F}(\pi))-\sum_k\lambda_{s,k}\,\mathcal{C}_{s,k}(\pi),\qquad \lambda\leftarrow\max(0,\lambda+\eta\,\partial_\lambda\mathcal{L}).$$

Phase 1 only *measures* $c_V, c_\ell, c^i_C$ each step (`info`) and reports the violation rate; nothing is enforced yet. The environment already exposes the costs in the exact form the Lagrangian controller will consume (`govgridmarl.agents.lagrangian`, Month 5).

## 5. Solution concept

CTDE: decentralised policies $\pi^i(a^i\mid o^i;\theta)$ with a shared $\theta$ and agent-id conditioning, trained with a centralised value $V(s;\phi)$ (MAPPO). The Phase-1 gate asks for *reliable convergence* of this learner under the efficiency-only reward; IPPO (decentralised critic) and PPO (centralised controller on $s$) bracket it from below and above in information.

## 6. Assumptions and their justification

* Hourly resolution: sufficient for peak-shaving DR; half-hourly is a config change (`horizon: 48`).
* Cluster agents rather than single homes: keeps N=14 tractable and makes loads visible to a transmission-level test case; the privacy layer still treats each cluster as one data owner.
* Synthetic thermal ratings at 1.5× nominal flow: case14 has no ratings; documented in `ieee14.py` and the Methods chapter.
* Rebound $\beta=0.5$: literature reports 40–70 % payback for residential DR events; the value is a config knob and is part of the sensitivity analysis planned for Month 11.
