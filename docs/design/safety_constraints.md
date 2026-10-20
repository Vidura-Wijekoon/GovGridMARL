# Safety constraints — formal specification (task m5-t1)

Companion to `cmomg_formulation.md` §4 and `reward_and_constraints.md` §2. This note fixes the constraint set that the Lagrangian learners enforce from Month 5 (`govgridmarl.safety.constraints`, `govgridmarl.agents.lagrangian`) and the way violations are counted for the Phase-2 gate.

## 1. Constrained problem

$$\max_\pi\ J_R(\pi)\quad\text{s.t.}\quad J_{C,k}(\pi)=\mathbb{E}_\pi\Big[\sum_{t=0}^{T-1} c_k(s_t,\mathbf a_t)\Big]\le d_k,\qquad k\in\mathcal K$$

with $J_R$ the scalarised Month-4 objective (efficiency + $w_{fair}$ · fairness + comfort, difference form) and non-negative per-step costs $c_k$. All safety budgets are $d_k=0$: the gate asks for **zero violations**, not an average rate.

## 2. Constraint set $\mathcal K$

| $k$ | cost $c_k(s_t)$ | bound | scope | physical meaning |
|---|---|---|---|---|
| voltage | $\max(0,V_{min}-\min_b V_{b,t})+\max(0,\max_b V_{b,t}-V_{max})$ | $[0.95,\,1.10]$ pu | shared | statutory voltage band; 1.10 because case14 set-points hold B8 at 1.09 pu (m1-t14) |
| loading | $\max(0,\max_{br}\ell_{br,t}-1)$ | rating = 1.5 × nominal flow | shared | thermal limit of lines / transformers |
| comfort | $\max(0,\,B^i_t/F^i_t-\theta_{g_i})$ | $\theta=(0.30,0.40,0.50)$ | per agent | daily curtailment share a household accepts |
| no-blackout | power-flow divergence ⇒ voltage cost 1.0, loading cost 2.0 | — | shared | loss of a feasible operating point (proxy for load shedding / collapse) |
| fairness (objective constraint, m5-t10) | $\max(0,\tau_f-\mathrm{Jain}(B_t))/T$ | $\tau_f=0.85$ | shared | Jain ≥ 0.85 on the kWh burden, spread over the day |

"Frequency deviation < 0.2 Hz" from the proposal is not modelled: the AC power flow is quasi-static and the slack bus absorbs every imbalance; it stays out of scope (scope note).

## 3. Violation accounting (gate metric)

* `info["violation"]` is true at step $t$ iff any **safety** cost (voltage, loading, comfort; blackout counts as both grid costs) is positive. The fairness cost is measured and logged but is **not** a safety violation.
* `episode_summary()["violation_rate"]` = violating steps / $T$; `violations_by_constraint` gives the count per constraint; the Phase-2 gate requires a mean rate of **0.0** over the 20 protocol days.
* The stress set (demand × 1.3, background × 1.25) and the N-1 contingency set (one line out of service, ratings unchanged, `GridMARLEnv(contingency=k)`) are reported separately: the requirement there is *fewer* violations than the unconstrained learner at equal peak reduction, since a constraint that cannot be met by any curtailment policy (e.g. a background overload) is a planning problem, not a control problem.

## 4. How the costs reach the learner

`info[agent]["cost_vec"] = {voltage, loading, comfort, fairness}` every step. The trainer collects the subset the agent names (`agent.constraint_names`), fits one **cost critic per constraint** on the same input as the reward critic (global state for MAPPO-Lagrangian), computes cost advantages with GAE, and after each policy update feeds the batch's mean episode costs $\hat J_{C,k}$ to the multiplier controller:

* dual ascent (m5-t3): $\lambda_k\leftarrow[\lambda_k+\eta(\hat J_{C,k}-d_k)]_+$;
* PID (m5-t6): $\lambda_k=[K_p e_k+K_i I_k+K_d\,\Delta e_k]_+$, $I_k\leftarrow[I_k+e_k]_+$, $e_k=\hat J_{C,k}-d_k$.

Policy step: clipped PPO on $A=(A_R-\sum_k\lambda_kA_{C,k})/(1+\sum_k\lambda_k)$. Multipliers are clamped to $[0,20]$ and saved with the checkpoint.

## 5. Sanity properties (tests/test_safety.py)

1. Every cost is 0 when its constraint holds and strictly positive when it does not (unit tests on the functions).
2. Permanent maximal curtailment violates comfort; no control violates nothing on a nominal day (`test_env.py`).
3. Dual ascent is monotone non-decreasing while a constraint is violated; PID responds proportionally and decays once the error disappears.
4. Removing a line raises the maximum loading on a nominal no-control day; restoring it reproduces the intact flows.
5. Constrained agents train, log `lambda_*` and `ep_cost_*`, and survive a checkpoint round-trip including the multiplier state.
