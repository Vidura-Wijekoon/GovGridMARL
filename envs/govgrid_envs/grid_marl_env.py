"""PettingZoo ParallelEnv for residential demand response on IEEE-14.

Implements tasks m2-t7 (agents & spaces), m2-t8 (reward & constraints),
m2-t12 (PettingZoo wrapper) and m2-t13 (reset/step/reward). The formal
model is the Constrained Multi-Objective Markov Game written up in
``docs/design/cmomg_formulation.md``; this module is its executable form.

Agents
------
``household_i`` (i = 0..N-1) are the learning agents. Each represents a
residential *cluster* of ``homes_per_agent`` homes in one income group
(low / mid / high) attached to an IEEE-14 load bus. The aggregator and the
DSO of the proposal are environment-side scripted entities in Phase 1: the
DSO publishes the tariff and a peak-warning signal, the aggregator computes
the network state. Hooks for making them learning agents exist
(``agent_roles``) but are not used before Phase 3.

Observation (per household, Box, dim = 14 + n_groups)
------------------------------------------------------
sin/cos time-of-day, own inflexible load, own flexible load (both kW per
home, normalised), own next-step base load forecast, price now, price next,
PV availability, bus voltage (pu, centred), max branch loading, aggregate
residential load (normalised), own cumulative curtailment burden, burden
minus population mean (fairness signal), previous action, group one-hot.

Action (per household, Discrete(n_levels))
------------------------------------------
Curtail level ``k`` removes ``k / (n_levels-1) * max_curtail`` of the
*flexible* load this step. A fraction ``rebound`` of curtailed energy comes
back over the next ``rebound_steps`` steps (load shifting, not shedding).

Reward (team + individual; averaged into a common reward by default)
--------------------------------------------------------------------
r_i =  w_peak * [(P_base/P_ref)^2 - (P_res/P_ref)^2]   (shared, peak pressure, difference form)
      +w_cost * price * (P_base - P_res) * dt / C_ref   (shared, energy cost saved vs no-control)
      -w_comfort * disutility_i                  (individual, convex in curtailment)
      -w_fair  * F_i(burdens) / T                (fairness term, m4-t3; 0 in Phase 1)
      -w_safety * violation_cost                 (shared, off by default: Phase 2 uses
                                                  a Lagrangian on info["cost"] instead)

Fairness term (task m4-t3). ``fairness_form`` selects the signal and
``fairness_basis`` what is equalised:

* ``"jain"``  (shared): F = 1 - Jain(B_t); every agent is charged the same.
* ``"gap"``   (individual): F_i = max(0, s_i - mean(s)); only agents that carry
  more than their fair share are charged (positive-part deviation).
* basis ``"kwh"`` uses B_i (the evaluation-protocol Jain), ``"fraction"``
  uses B_i / F_i (relative burden, equal share of own flexibility).

The term is divided by the horizon T so that one day at Jain = 0.65 costs
w_fair * 0.35 per agent - the same order as the efficiency return (~1.25) -
which keeps the sweep over w_fair in [0, 8] meaningful (m4-t5).

Objective vector (m4-t4 / m4-t8). ``info[agent]["objectives"]`` carries the
*unweighted* components {efficiency, fairness, comfort, safety} such that
``reward_i = efficiency_i + w_fair * fairness_i + w_comfort * comfort_i +
w_safety * safety_i`` (the efficiency component already includes w_peak and
w_cost). MOPPO consumes this vector; weighted-sum scalarisation is exactly
the scalar reward.

Both shared terms are *difference rewards* against the known no-control
counterfactual (do-nothing scores exactly 0), which removes the large
action-independent constant and the day-to-day demand variance from the
learning signal without changing the optimal policy.

Constraints are always *measured* and exposed in ``info``:
``cost_voltage``, ``cost_loading``, ``cost_comfort`` (>= 0, 0 = satisfied),
plus the boolean ``violation``. Phase 2 (m5) turns these into Lagrangian
terms; Phase 1 only needs them to be observable and correct (m2-t14).
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces

from govgridmarl.data.synthetic import GROUPS, LoadDataset, generate_profiles
from govgridmarl.fairness.jain import jain_index
from govgridmarl.fairness.metrics import burden_fraction, fair_share_deviation, fairness_report
from govgridmarl.safety.constraints import (
    ALL_CONSTRAINTS,
    comfort_cost,
    fairness_cost,
    loading_cost,
    voltage_violation_cost,
)

try:  # PettingZoo is a hard dependency at runtime but keep the import soft for docs
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover
    ParallelEnv = object  # type: ignore[misc,assignment]


DEFAULT_REWARD_WEIGHTS = {"peak": 1.0, "cost": 0.5, "comfort": 0.1, "fairness": 0.0, "safety": 0.0}


class GridMARLEnv(ParallelEnv):
    """Parallel multi-agent demand-response environment over IEEE-14.

    Parameters
    ----------
    n_households:
        Number of household cluster agents (default 14).
    case:
        Only ``"ieee14"`` is implemented; ``"ieee118"`` is a Month-6 task.
    horizon:
        Steps per episode (24 = hourly day).
    seed:
        Base seed for the dataset and the episode RNG.
    homes_per_agent:
        Homes represented by one agent (sets the MW scale seen by the grid).
    n_levels:
        Number of discrete curtailment levels (0 = no curtailment).
    max_curtail:
        Largest fraction of *flexible* load one action can curtail.
    rebound, rebound_steps:
        Share of curtailed energy that returns, and over how many steps.
    reward_weights:
        Dict with keys peak, cost, comfort, fairness, safety.
    v_min, v_max, max_loading:
        Safety constraint bounds (pu, pu, fraction of rating). ``v_max``
        defaults to 1.10 because case14 generator set-points legitimately
        hold bus 8 at 1.09 pu (see m1-t14 log).
    rating_factor:
        Synthetic thermal ratings relative to nominal case14 flows.
    background_factor:
        Scale of the uncontrollable case14 background load.
    use_power_flow:
        If False, skip the AC power flow (fast unit tests / debugging); grid
        observations are then zeros and no voltage/loading costs are produced.
    day_variability:
        Std of the per-episode log-normal demand multiplier (weather / season).
    pf_backend:
        ``"auto"`` | ``"lightsim"`` | ``"pandapower"``.
    renewable_share, wind_fraction, wind_variability:
        Generation-mix scenario ("energy share"). ``renewable_share`` is the
        fraction of the *nominal daily system energy* (background + residential)
        supplied by renewables; ``wind_fraction`` splits it between a
        transmission-connected wind farm at B2 and rooftop PV distributed over
        the 11 load buses in proportion to their load; the remainder comes from
        the conventional units (slack B1 + B2). ``wind_variability`` is the
        log-normal day-to-day spread of the wind profile. Default 0 = the
        Phase-1 baseline without renewables; the sweep 0 → 0.8 is the
        energy-share analysis requested by the supervisor.
    dataset:
        Optional pre-built :class:`LoadDataset` (e.g. from UK-DALE loaders).
    reward_mode:
        ``"difference"`` (default): each agent gets its *difference reward*
        (marginal contribution to the shared efficiency objective, x N) plus
        its own comfort term — same optimum as the team objective, far less
        credit-assignment noise with 14 simultaneous learners.
        ``"team"``: every agent gets the team-mean reward (common-reward
        Dec-POMDP). ``"individual"``: shared peak/cost + private comfort — a
        mixed-motive game with a free-rider equilibrium, kept for the
        incentive analysis in Phase 2. ``info["team_reward"]`` always carries
        the team objective for evaluation regardless of the mode.
    """

    metadata = {"name": "govgrid_marl_v0", "render_modes": ["ansi"], "is_parallelizable": True}

    def __init__(
        self,
        n_households: int = 14,
        case: str = "ieee14",
        horizon: int = 24,
        seed: int = 42,
        homes_per_agent: int = 2000,
        n_levels: int = 5,
        max_curtail: float = 0.6,
        rebound: float = 0.5,
        rebound_steps: int = 2,
        reward_weights: dict[str, float] | None = None,
        v_min: float = 0.95,
        v_max: float = 1.10,
        max_loading: float = 1.0,
        rating_factor: float = 1.5,
        background_factor: float = 1.0,
        use_power_flow: bool = True,
        day_variability: float = 0.10,
        pf_backend: str = "auto",
        renewable_share: float = 0.0,
        wind_fraction: float = 0.5,
        wind_variability: float = 0.25,
        dataset: LoadDataset | None = None,
        reward_mode: str = "difference",
        fairness_form: str = "jain",
        fairness_basis: str = "kwh",
        fairness_target: float = 0.85,
        fairness_target_basis: str = "fraction",
        contingency: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        if case != "ieee14":
            raise NotImplementedError("Only case='ieee14' is implemented (IEEE-118 is task m6-t*)")
        self.n_households = int(n_households)
        self.case = case
        self.horizon = int(horizon)
        self.seed_value = int(seed)
        self.homes_per_agent = int(homes_per_agent)
        self.n_levels = int(n_levels)
        self.max_curtail = float(max_curtail)
        self.rebound = float(rebound)
        self.rebound_steps = int(rebound_steps)
        self.w = dict(DEFAULT_REWARD_WEIGHTS, **(reward_weights or {}))
        self.v_min, self.v_max, self.max_loading = float(v_min), float(v_max), float(max_loading)
        self.rating_factor = float(rating_factor)
        self.background_factor = float(background_factor)
        self.use_power_flow = bool(use_power_flow)
        self.day_variability = float(day_variability)
        self.renewable_share = float(renewable_share)
        self.wind_fraction = float(wind_fraction)
        self.wind_variability = float(wind_variability)
        if not 0.0 <= self.renewable_share < 1.0 or not 0.0 <= self.wind_fraction <= 1.0:
            raise ValueError("renewable_share in [0,1), wind_fraction in [0,1]")
        if reward_mode not in ("difference", "team", "individual"):
            raise ValueError("reward_mode must be 'difference', 'team' or 'individual'")
        self.reward_mode = reward_mode
        if fairness_form not in ("jain", "gap"):
            raise ValueError("fairness_form must be 'jain' or 'gap'")
        if fairness_basis not in ("kwh", "fraction"):
            raise ValueError("fairness_basis must be 'kwh' or 'fraction'")
        self.fairness_form, self.fairness_basis = fairness_form, fairness_basis
        self.fairness_target = float(fairness_target)
        if fairness_target_basis not in ("kwh", "fraction"):
            raise ValueError("fairness_target_basis must be 'kwh' or 'fraction'")
        self.fairness_target_basis = fairness_target_basis  # gate basis (amended 2026-10-20)
        self.render_mode = render_mode

        self.possible_agents = [f"household_{i}" for i in range(self.n_households)]
        self.agents: list[str] = []
        self._idx = {a: i for i, a in enumerate(self.possible_agents)}

        # data ---------------------------------------------------------------
        self.dataset = dataset or generate_profiles(self.n_households, self.horizon, seed=seed)
        if self.dataset.n_households != self.n_households:
            raise ValueError("dataset households must match n_households")
        self.groups = list(self.dataset.groups)
        self.group_ids = np.array([GROUPS.index(g) for g in self.groups])
        self.comfort_tol = np.array([m.get("comfort_tol", 0.4) for m in self.dataset.meta])
        self.vulnerable = np.array(self.dataset.vulnerable, dtype=bool)  # m4-t2
        self.dt = self.dataset.dt_hours
        self.price = self.dataset.price
        self.pv = self.dataset.pv_kw

        # normalisation references (per home, kW) -----------------------------
        self._p_ref_home = float(self.dataset.base_kw.sum(axis=0).max() / self.n_households)
        self._c_ref = float(
            (self.dataset.base_kw.sum(axis=0) * self.price).sum()
            * self.dt
            / self.horizon
            / self.n_households
        )

        # power flow ---------------------------------------------------------
        self._pf = None
        if self.use_power_flow:
            from govgrid_envs.ieee14 import Ieee14PowerFlow, background_shape

            self._pf = Ieee14PowerFlow(self.n_households, self.rating_factor, backend=pf_backend)
            self._pf.set_contingency(contingency)
            self._bg_shape = background_shape(self.horizon)
            self.household_buses = list(self._pf.household_buses)
        else:
            from govgrid_envs.ieee14 import household_bus_map

            self._bg_shape = np.ones(self.horizon)
            self.household_buses = household_bus_map(self.n_households)

        self._rng = np.random.default_rng(seed)
        self._setup_generation_mix()
        self._obs_dim = 14 + len(GROUPS)
        self._t = 0
        self._episode_count = 0

    # ------------------------------------------------------------------ spaces
    @functools.cache
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=-5.0, high=5.0, shape=(self._obs_dim,), dtype=np.float32)

    @functools.cache
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(self.n_levels)

    @property
    def state_dim(self) -> int:
        return self.state().shape[0]

    # ------------------------------------------------------------------- reset
    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.agents = list(self.possible_agents)
        self._t = 0
        self._episode_count += 1
        options = options or {}
        # episode-level demand multiplier (weather / season)
        self._day_scale = float(
            options.get("day_scale", np.exp(self._rng.normal(0.0, self.day_variability)))
        )
        noise = np.exp(self._rng.normal(0.0, 0.05, size=self.dataset.base_kw.shape))
        self._base = self.dataset.base_kw * self._day_scale * noise  # kW/home
        self._flex = np.minimum(self.dataset.flexible_kw * self._day_scale * noise, self._base)
        self._pending = np.zeros((self.n_households, self.horizon + self.rebound_steps + 1))
        self._burden = np.zeros(self.n_households)  # cumulative curtailed kWh/home
        self._flex_energy = np.zeros(self.n_households)  # cumulative flexible kWh/home
        self._prev_action = np.zeros(self.n_households)
        self._res_profile: list[float] = []
        self._baseline_profile: list[float] = []
        self._cost_total = 0.0
        self._violations = 0
        self._violations_by = dict.fromkeys(ALL_CONSTRAINTS, 0)
        self._last_pf = None
        self._last_grid = {"vm_min": 1.0, "vm_max": 1.0, "loading_max": 0.0}
        self._served = np.zeros(self.n_households)  # kW/home served this step
        self._wind_day = float(
            options.get("wind_scale", np.exp(self._rng.normal(0.0, self.wind_variability)))
        )
        self._mix_log: list[dict[str, float]] = []
        self._solve_grid(self._base[:, 0])  # initial grid state for observations
        obs = self._all_obs()
        infos = {a: {} for a in self.agents}
        return obs, infos

    # -------------------------------------------------------------------- step
    def step(self, actions: dict[str, int]):
        if not self.agents:
            raise RuntimeError("step() called on a finished episode; call reset()")
        t = self._t
        n = self.n_households
        act = np.zeros(n)
        for a, k in actions.items():
            act[self._idx[a]] = int(k)
        frac = act / max(1, self.n_levels - 1) * self.max_curtail  # curtailment fraction

        base = self._base[:, t]
        flex = self._flex[:, t]
        curtailed = frac * flex  # kW/home removed this step
        served = base - curtailed + self._pending[:, t]  # rebound from earlier steps
        # schedule rebound of curtailed energy
        if self.rebound > 0 and self.rebound_steps > 0:
            share = self.rebound * curtailed / self.rebound_steps
            for k in range(1, self.rebound_steps + 1):
                self._pending[:, t + k] += share
        self._served = served
        self._burden += curtailed * self.dt
        self._flex_energy += flex * self.dt

        # grid -----------------------------------------------------------
        self._solve_grid(served)
        self._mix_log.append(dict(self._last_mix))
        g = self._last_grid
        p_res_home = float(served.mean())  # kW per home (aggregate / n)
        self._res_profile.append(float(served.sum() * self.homes_per_agent / 1000.0))  # MW
        self._baseline_profile.append(float(base.sum() * self.homes_per_agent / 1000.0))

        # constraint costs (>= 0; 0 = satisfied), govgridmarl.safety (m5-t1/t2) ----
        cost_voltage = voltage_violation_cost([g["vm_min"], g["vm_max"]], self.v_min, self.v_max)
        cost_loading = loading_cost([g["loading_max"]], self.max_loading)
        cost_comfort_i = comfort_cost(self._burden, self._flex_energy, self.comfort_tol)
        violation = (cost_voltage > 0) or (cost_loading > 0) or bool((cost_comfort_i > 0).any())
        self._violations += int(violation)
        self._violations_by["voltage"] += int(cost_voltage > 0)
        self._violations_by["loading"] += int(cost_loading > 0)
        self._violations_by["comfort"] += int((cost_comfort_i > 0).any())

        # rewards ----------------------------------------------------------
        # Shared terms are expressed relative to the known no-control load b
        # (action-independent, so the optimal policy is unchanged, but the
        # large constant and the day-to-day demand variance drop out of the
        # learning signal). Do-nothing scores exactly 0.
        p_base_home = float(base.mean())
        pr, cr = self._p_ref_home, max(self._c_ref, 1e-9)
        price = float(self.price[t])

        def g_team(p_mean: float) -> float:  # shared efficiency objective
            return (
                self.w["peak"] * ((p_base_home / pr) ** 2 - (p_mean / pr) ** 2)
                + self.w["cost"] * price * (p_base_home - p_mean) * self.dt / cr
            )

        team_eff = g_team(p_res_home)
        step_cost = float(price * p_res_home * self.dt)
        self._cost_total += step_cost * self.n_households
        disutility = frac**2 * (flex / np.maximum(base, 1e-9))  # convex, scaled by flex share
        comfort_term = -self.w["comfort"] * disutility * 10.0
        jain = jain_index(self._burden.tolist()) if self._burden.sum() > 0 else 1.0
        jain_gate = jain if self.fairness_target_basis == "kwh" else self._jain_frac()
        cost_fair = fairness_cost(jain_gate, self.fairness_target, self.horizon)
        self._violations_by["fairness"] += int(cost_fair > 0)
        fair_vec = self._fairness_signal(jain)  # (N,) >= 0, unweighted, already / T
        fair_term = -self.w["fairness"] * fair_vec
        safety_vec = cost_voltage * 10.0 + cost_loading + cost_comfort_i  # (N,)
        safety_term = -self.w["safety"] * safety_vec
        if self.reward_mode == "difference":
            # Difference reward (Wolpert & Tumer 2002): agent i is credited with
            # the marginal effect of its own deviation from base load on the
            # shared objective, G(z) - G(z_{-i}), scaled by N so magnitudes
            # match the team reward. Removes other agents' exploration noise
            # from i's credit signal (crucial with N=14 simultaneous learners)
            # while keeping the same optimum as the team objective.
            delta = served - base  # own deviation (kW/home); rebound counts against you
            eff = np.array([n * (team_eff - g_team(p_res_home - d / n)) for d in delta])
            rewards_vec = eff + fair_term + comfort_term + safety_term
        else:
            rewards_vec = team_eff + fair_term + comfort_term + safety_term
            if self.reward_mode == "team":  # fully cooperative common reward
                rewards_vec = np.full(n, float(rewards_vec.mean()))
        self._team_reward = float(
            team_eff + fair_term.mean() + comfort_term.mean() + np.mean(safety_term)
        )
        self._prev_action = frac / max(self.max_curtail, 1e-9)
        # unweighted objective components (m4-t4): reward_i == sum_k w_k * obj_k
        if self.reward_mode == "difference":
            eff_vec = eff
        elif self.reward_mode == "team":
            eff_vec = np.full(n, float(np.mean(rewards_vec)) - 0.0)
            # the team mean already contains everyone's fairness/comfort/safety:
            eff_vec = eff_vec - (fair_term + comfort_term + safety_term)
        else:
            eff_vec = np.full(n, float(team_eff))
        self._objectives = {
            "efficiency": eff_vec,
            "fairness": -fair_vec,
            "comfort": -disutility * 10.0,
            "safety": -safety_vec,
        }

        self._t += 1
        done = self._t >= self.horizon
        obs = (
            self._all_obs()
            if not done
            else {a: self._obs(i) for i, a in enumerate(self.possible_agents)}
        )
        rewards = {a: float(rewards_vec[i]) for i, a in enumerate(self.possible_agents)}
        terminations = {a: False for a in self.possible_agents}
        truncations = {a: done for a in self.possible_agents}
        common = {
            "t": t,
            "team_reward": self._team_reward,
            "p_res_mw": self._res_profile[-1],
            "p_base_mw": self._baseline_profile[-1],
            "vm_min": g["vm_min"],
            "vm_max": g["vm_max"],
            "loading_max": g["loading_max"],
            "cost_voltage": cost_voltage,
            "cost_loading": cost_loading,
            "cost_comfort": float(cost_comfort_i.mean()),
            "cost_fairness": cost_fair,
            "violation": violation,
            "jain": jain,
            "jain_frac": float(self._jain_frac()),
            "pf_converged": bool(g.get("converged", True)),
            "mix": dict(self._last_mix),
        }
        infos = {
            a: dict(
                common,
                curtail_frac=float(frac[i]),
                burden_kwh=float(self._burden[i]),
                cost_comfort_i=float(cost_comfort_i[i]),
                group=self.groups[i],
                vulnerable=bool(self.vulnerable[i]),
                objectives={k: float(v[i]) for k, v in self._objectives.items()},
                cost_vec={
                    "voltage": cost_voltage,
                    "loading": cost_loading,
                    "comfort": float(cost_comfort_i[i]),
                    "fairness": cost_fair,
                },
                # indicator form (m5-t5): 1 per violating step, so an episode cost is a
                # count of violating hours (0..T) - the scale Safety-Gym style budgets use
                cost_ind={
                    "voltage": float(cost_voltage > 0),
                    "loading": float(cost_loading > 0),
                    "comfort": float(cost_comfort_i[i] > 0),
                    "fairness": float(cost_fair > 0),
                },
            )
            for i, a in enumerate(self.possible_agents)
        }
        if done:
            summary = self.episode_summary()
            for a in self.possible_agents:
                infos[a]["episode"] = summary
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    # ------------------------------------------------------------- helpers
    @property
    def contingency(self) -> int | None:
        """Line index out of service (N-1 study, m5-t12) or ``None``."""
        return self._pf.contingency if self._pf is not None else None

    def set_contingency(self, line_idx: int | None) -> None:
        """Apply / clear an N-1 line outage; takes effect from the next ``reset()``."""
        if self._pf is None:
            raise RuntimeError("contingencies need the power flow (use_power_flow=True)")
        self._pf.set_contingency(line_idx)

    def _jain_frac(self) -> float:
        fr = burden_fraction(self._burden, self._flex_energy)
        return float(jain_index(fr.tolist())) if fr.sum() > 0 else 1.0

    def _fairness_signal(self, jain_kwh: float) -> np.ndarray:
        """Per-agent unweighted fairness penalty (>= 0), divided by T (m4-t3)."""
        n = self.n_households
        if self._burden.sum() <= 0:
            return np.zeros(n)
        if self.fairness_form == "jain":
            j = jain_kwh if self.fairness_basis == "kwh" else self._jain_frac()
            return np.full(n, (1.0 - j) / self.horizon)
        cap = self._flex_energy if self.fairness_basis == "fraction" else None
        dev = fair_share_deviation(self._burden, cap)
        if cap is None:  # kWh deviation -> normalise by the mean burden so it is O(1)
            dev = dev / max(float(self._burden.mean()), 1e-9)
        return np.maximum(0.0, dev) / self.horizon

    def _solve_grid(self, served_kw_home: np.ndarray) -> None:
        if self._pf is None:
            self._last_grid = {"vm_min": 1.0, "vm_max": 1.0, "loading_max": 0.0, "converged": True}
            res_mw = float(served_kw_home.sum() * self.homes_per_agent / 1000.0)
            self._last_mix = {
                k: 0.0 for k in ("pv_mw", "wind_mw", "ext_grid_mw", "losses_mw", "surplus_mw")
            }
            self._last_mix.update({"load_mw": res_mw, "res_mw": res_mw, "conv_mw": res_mw})
            return
        p_mw = served_kw_home * self.homes_per_agent / 1000.0
        t = min(self._t, self.horizon - 1)
        bg = self.background_factor * self._bg_shape[t]
        pv_mw, wind_mw = self.renewable_injection(t)
        res = self._pf.solve(p_mw, bg, pv_mw=pv_mw, wind_mw=wind_mw)
        self._last_pf = res
        load_mw = float(p_mw.sum() + self._pf.background_p.sum() * bg)
        self._last_mix = {
            "load_mw": load_mw,
            "res_mw": float(p_mw.sum()),
            "pv_mw": float(pv_mw.sum()),
            "wind_mw": float(wind_mw),
            "conv_mw": float(res.slack_p_mw),
            "ext_grid_mw": float(res.ext_grid_p_mw),
            "losses_mw": float(res.losses_mw),
            "surplus_mw": float(max(0.0, -res.ext_grid_p_mw)),  # export = renewable surplus
        }
        if res.converged:
            self._last_grid = {
                "vm_min": float(res.vm_pu.min()),
                "vm_max": float(res.vm_pu.max()),
                "loading_max": res.max_loading,
                "converged": True,
            }
        else:  # treat divergence as a hard violation (blackout proxy)
            self._last_grid = {"vm_min": 0.0, "vm_max": 1.0, "loading_max": 2.0, "converged": False}

    def _obs(self, i: int) -> np.ndarray:
        t = min(self._t, self.horizon - 1)
        tn = min(self._t + 1, self.horizon - 1)
        g = self._last_grid
        base, flex = self._base[i, t], self._flex[i, t]
        burden_frac = self._burden / np.maximum(self._flex_energy, 1e-9)
        bus = self.household_buses[i]
        vm_bus = (
            float(self._last_pf.vm_pu[bus])
            if (self._last_pf is not None and self._last_pf.converged)
            else 1.0
        )
        agg = (
            float(self._served.mean() / self._p_ref_home)
            if self._t > 0
            else float(self._base[:, 0].mean() / self._p_ref_home)
        )
        vec = [
            np.sin(2 * np.pi * t / self.horizon),
            np.cos(2 * np.pi * t / self.horizon),
            (base - flex) / self._p_ref_home,
            flex / self._p_ref_home,
            self._base[i, tn] / self._p_ref_home,
            self.price[t] / 0.4,
            self.price[tn] / 0.4,
            self.pv[t] / max(self.pv.max(), 1e-9),
            (vm_bus - 1.0) * 20.0,
            g["loading_max"],
            agg,
            burden_frac[i],
            burden_frac[i] - burden_frac.mean(),
            self._prev_action[i],
        ]
        onehot = np.zeros(len(GROUPS))
        onehot[self.group_ids[i]] = 1.0
        return np.clip(np.concatenate([np.asarray(vec, dtype=np.float32), onehot]), -5, 5).astype(
            np.float32
        )

    def _all_obs(self) -> dict[str, np.ndarray]:
        return {a: self._obs(i) for i, a in enumerate(self.possible_agents)}

    def state(self) -> np.ndarray:
        """Global state for centralised critics: concat of all local obs + grid summary."""
        g = self._last_grid
        grid = np.array(
            [g["vm_min"] - 1.0, g["vm_max"] - 1.0, g["loading_max"]], dtype=np.float32
        ) * [20, 20, 1]
        return np.concatenate([self._obs(i) for i in range(self.n_households)] + [grid]).astype(
            np.float32
        )

    def episode_summary(self) -> dict[str, Any]:
        res = np.asarray(self._res_profile)
        base = np.asarray(self._baseline_profile)
        peak_red = float((base.max() - res.max()) / base.max() * 100.0) if base.size else 0.0
        burdens = self._burden.tolist()
        by_group = {}
        for gname in GROUPS:
            m = self.group_ids == GROUPS.index(gname)
            if m.any():
                by_group[gname] = float(self._burden[m].mean())
        fair = fairness_report(burdens, self._flex_energy, self.groups, self.vulnerable.tolist())
        return {
            "peak_reduction_pct": peak_red,
            "res_peak_mw": float(res.max()) if res.size else 0.0,
            "base_peak_mw": float(base.max()) if base.size else 0.0,
            "energy_res_mwh": float(res.sum() * self.dt),
            "energy_base_mwh": float(base.sum() * self.dt),
            "cost_total": float(self._cost_total * self.homes_per_agent),  # currency (all homes)
            "jain_index": float(jain_index(burdens)) if sum(burdens) > 0 else 1.0,
            "violations": int(self._violations),
            "violation_rate": float(self._violations / max(1, self.horizon)),
            "violations_by_constraint": dict(self._violations_by),
            "contingency": self.contingency,
            "burden_by_group_kwh": by_group,
            "jain_frac": fair["jain_frac"],
            "gini_kwh": fair["gini_kwh"],
            "group_gap_ratio": fair["group_gap_ratio"],
            "vulnerable_burden_ratio": fair["vulnerable_burden_ratio"],
            "vulnerable_share_of_burden": fair["vulnerable_share_of_burden"],
            "day_scale": self._day_scale,
            **self.energy_share_summary(),
        }

    # ------------------------------------------------------- generation mix
    def _setup_generation_mix(self) -> None:
        """Pre-compute renewable profiles for the configured energy-share scenario."""
        H = self.horizon
        t = (np.arange(H) + 0.5) * (24.0 / H)
        # nominal daily system energy (MWh): background + residential base
        bg_p = self._pf.background_p if self._pf is not None else np.zeros(1)
        bg_bus_p = np.asarray(bg_p, dtype=float)
        e_bg = float(bg_bus_p.sum() * self._bg_shape.sum() * self.dt)
        e_res = float(self.dataset.base_kw.sum() * self.homes_per_agent / 1000.0 * self.dt)
        self._e_nominal_mwh = e_bg + e_res
        e_ren = self.renewable_share * self._e_nominal_mwh
        e_wind, e_pv = self.wind_fraction * e_ren, (1.0 - self.wind_fraction) * e_ren
        # shapes (mean-normalised to 1)
        pv_shape = np.asarray(self.pv, dtype=float)
        pv_shape = pv_shape / max(pv_shape.mean(), 1e-9)
        wind_shape = 1.0 + 0.35 * np.cos(2 * np.pi * (t - 3.0) / 24.0)  # night-peaking
        wind_shape = wind_shape / wind_shape.mean()
        self._pv_total_mw = e_pv / (H * self.dt) * pv_shape  # MW per step, all buses
        self._wind_mw = e_wind / (H * self.dt) * wind_shape
        # PV split over load buses in proportion to nominal bus load (rooftop share)
        if self._pf is not None and bg_bus_p.size == len(self._pf.load_buses):
            res_by_bus = np.zeros(len(self._pf.load_buses))
            mean_home_kw = self.dataset.base_kw.mean(axis=1) * self.homes_per_agent / 1000.0
            for i, b in enumerate(self.household_buses):
                res_by_bus[self._pf.load_buses.index(b)] += mean_home_kw[i]
            w = bg_bus_p * self._bg_shape.mean() + res_by_bus
            self._pv_bus_weights = w / w.sum()
        else:
            self._pv_bus_weights = np.ones(1)
        self._wind_day = 1.0

    def renewable_injection(self, t: int) -> tuple[np.ndarray, float]:
        """(pv_mw per load bus, wind_mw) at step ``t`` for the current episode."""
        if self.renewable_share <= 0.0:
            return np.zeros(len(self._pv_bus_weights)), 0.0
        return self._pv_total_mw[t] * self._pv_bus_weights, float(self._wind_mw[t] * self._wind_day)

    def energy_share_summary(self) -> dict[str, Any]:
        """Energy-share KPIs for the episode (system level, MWh)."""
        if not getattr(self, "_mix_log", None):
            return {}
        m = self._mix_log
        dt = self.dt
        e_load = sum(x["load_mw"] for x in m) * dt
        e_pv = sum(x["pv_mw"] for x in m) * dt
        e_wind = sum(x["wind_mw"] for x in m) * dt
        e_surplus = sum(x["surplus_mw"] for x in m) * dt
        e_conv = sum(max(0.0, x["conv_mw"]) for x in m) * dt
        net_peak = max(x["ext_grid_mw"] for x in m)
        return {
            "renewable_share_config": self.renewable_share,
            "wind_fraction_config": self.wind_fraction,
            "energy_load_mwh": float(e_load),
            "energy_pv_mwh": float(e_pv),
            "energy_wind_mwh": float(e_wind),
            "energy_conventional_mwh": float(e_conv),
            "renewable_surplus_mwh": float(e_surplus),
            "renewable_share_of_consumption": float(
                (e_pv + e_wind - e_surplus) / max(e_load, 1e-9)
            ),
            "net_import_peak_mw": float(net_peak),
            "system_losses_mwh": float(sum(x["losses_mw"] for x in m) * dt),
        }

    def render(self):
        if self._res_profile:
            return f"t={self._t} P_res={self._res_profile[-1]:.2f} MW grid={self._last_grid}"
        return f"t={self._t} (reset)"

    def close(self):
        pass


def make_env(**kwargs) -> GridMARLEnv:
    """Factory used by Hydra (``_target_: govgrid_envs.make_env``)."""
    return GridMARLEnv(**kwargs)


__all__ = ["GridMARLEnv", "make_env", "DEFAULT_REWARD_WEIGHTS"]
