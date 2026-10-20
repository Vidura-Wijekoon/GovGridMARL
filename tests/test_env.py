"""Environment correctness tests (tasks m2-t13 / m2-t14)."""

import numpy as np
import pytest

from govgrid_envs import GridMARLEnv
from govgridmarl.data import GROUPS, generate_profiles


@pytest.fixture(scope="module")
def env():
    return GridMARLEnv(n_households=8, seed=0, day_variability=0.0)


@pytest.fixture(scope="module")
def fast_env():
    return GridMARLEnv(n_households=6, seed=0, use_power_flow=False, day_variability=0.0)


# --- data ---------------------------------------------------------------------
def test_synthetic_profiles_calibrated():
    ds = generate_profiles(30, 24, seed=1)
    s = ds.summary()
    assert 5.0 < s["daily_kwh_mean"] < 11.0  # TDCV low..high band
    assert 17 <= s["aggregate_peak_hour"] <= 20  # evening peak
    assert 0.35 < s["load_factor"] < 0.65
    assert 0.15 <= s["flex_share_mean"] <= 0.40
    assert set(ds.groups) == set(GROUPS)
    assert (ds.flexible_kw <= ds.base_kw + 1e-12).all()
    assert (ds.base_kw > 0).all()


def test_synthetic_is_deterministic():
    a = generate_profiles(5, 24, seed=7)
    b = generate_profiles(5, 24, seed=7)
    assert np.allclose(a.base_kw, b.base_kw)
    assert a.groups == b.groups


# --- spaces / API -------------------------------------------------------------
def test_spaces(fast_env):
    obs, infos = fast_env.reset(seed=0)
    assert set(obs) == set(fast_env.possible_agents)
    for a, o in obs.items():
        assert fast_env.observation_space(a).contains(o)
        assert fast_env.action_space(a).n == fast_env.n_levels
    assert fast_env.state().shape[0] == fast_env.state_dim


def test_pettingzoo_api(fast_env):
    from pettingzoo.test import parallel_api_test

    parallel_api_test(fast_env, num_cycles=50)


def test_episode_length_and_truncation(fast_env):
    fast_env.reset(seed=3)
    n = 0
    while fast_env.agents:
        _, _, term, trunc, _ = fast_env.step({a: 0 for a in fast_env.agents})
        n += 1
    assert n == fast_env.horizon
    assert all(trunc.values()) and not any(term.values())


# --- dynamics -----------------------------------------------------------------
def test_no_control_is_baseline(fast_env):
    fast_env.reset(seed=5)
    while fast_env.agents:
        _, _, _, _, info = fast_env.step({a: 0 for a in fast_env.agents})
    ep = info["household_0"]["episode"]
    assert ep["peak_reduction_pct"] == pytest.approx(0.0)
    assert ep["jain_index"] == 1.0
    assert ep["energy_res_mwh"] == pytest.approx(ep["energy_base_mwh"])


def test_full_curtailment_reduces_peak_and_energy(fast_env):
    fast_env.reset(seed=5)
    top = fast_env.n_levels - 1
    while fast_env.agents:
        _, _, _, _, info = fast_env.step({a: top for a in fast_env.agents})
    ep = info["household_0"]["episode"]
    assert ep["peak_reduction_pct"] > 5.0
    # rebound < 1 so some energy is genuinely shed
    assert ep["energy_res_mwh"] < ep["energy_base_mwh"]
    # comfort constraint must fire under permanent maximal curtailment
    assert ep["violations"] > 0


def test_rebound_conserves_energy_when_full():
    e = GridMARLEnv(
        n_households=4,
        seed=0,
        use_power_flow=False,
        day_variability=0.0,
        rebound=1.0,
        rebound_steps=1,
    )
    e.reset(seed=1)
    top = e.n_levels - 1
    while e.agents:
        _, _, _, _, info = e.step({a: top for a in e.agents})
    ep = info["household_0"]["episode"]
    # energy shifted past the horizon is lost, so allow a small deficit only
    assert ep["energy_res_mwh"] <= ep["energy_base_mwh"] + 1e-9
    assert ep["energy_res_mwh"] > 0.95 * ep["energy_base_mwh"]


def test_unequal_curtailment_lowers_jain(fast_env):
    fast_env.reset(seed=2)
    agents = fast_env.possible_agents
    top = fast_env.n_levels - 1
    while fast_env.agents:
        acts = {a: (top if i == 0 else 0) for i, a in enumerate(agents)}
        _, _, _, _, info = fast_env.step(acts)
    ep = info["household_0"]["episode"]
    assert ep["jain_index"] == pytest.approx(1.0 / len(agents), rel=1e-6)


def test_reward_prefers_curtailment_at_peak(fast_env):
    """At the evening peak, curtailing must not be worse than doing nothing (peak/cost terms)."""
    fast_env.reset(seed=4)
    for _ in range(18):
        fast_env.step({a: 0 for a in fast_env.agents})
    import copy

    e0 = copy.deepcopy(fast_env)
    _, r0, *_ = e0.step({a: 0 for a in e0.agents})
    e1 = copy.deepcopy(fast_env)
    _, r1, *_ = e1.step({a: 2 for a in e1.agents})
    assert sum(r1.values()) > sum(r0.values())


# --- grid coupling ------------------------------------------------------------
def test_power_flow_signals(env):
    env.reset(seed=1)
    seen_loading = []
    while env.agents:
        _, _, _, _, info = env.step({a: 0 for a in env.agents})
        i = info["household_0"]
        assert i["pf_converged"]
        assert 0.9 < i["vm_min"] <= i["vm_max"] < 1.11
        seen_loading.append(i["loading_max"])
    # loading must vary through the day and be substantial at the evening peak
    assert max(seen_loading) - min(seen_loading) > 0.05
    assert 0.3 < max(seen_loading) < 1.5


def test_constraint_fires_under_stress():
    e = GridMARLEnv(n_households=8, seed=0, day_variability=0.0, background_factor=1.25)
    e.reset(seed=1, options={"day_scale": 1.6})
    costs = []
    while e.agents:
        _, _, _, _, info = e.step({a: 0 for a in e.agents})
        costs.append(info["household_0"]["cost_loading"])
    assert max(costs) > 0.0  # thermal constraint signal fires on a stressed day


def test_backends_agree():
    from govgrid_envs.ieee14 import Ieee14PowerFlow

    a = Ieee14PowerFlow(6, backend="pandapower")
    try:
        b = Ieee14PowerFlow(6, backend="lightsim")
    except Exception:
        pytest.skip("lightsim2grid not available")
    p = np.array([2.0, 3.0, 1.0, 4.0, 2.5, 1.5])
    ra, rb = a.solve(p, 1.0), b.solve(p, 1.0)
    assert ra.converged and rb.converged
    assert np.allclose(ra.vm_pu, rb.vm_pu, atol=1e-6)
    assert np.allclose(ra.trafo_loading, rb.trafo_loading, atol=1e-6)
    assert np.allclose(ra.line_loading, rb.line_loading, atol=2e-2)  # from-side vs max-side current


# --- generation mix / energy share (supervisor request, Sep 2026) ----------------
def test_renewable_share_zero_is_baseline():
    a = GridMARLEnv(n_households=6, seed=0, day_variability=0.0)
    b = GridMARLEnv(n_households=6, seed=0, day_variability=0.0, renewable_share=0.0)
    a.reset(seed=1)
    b.reset(seed=1)
    while a.agents:
        _, ra, _, _, ia = a.step({x: 1 for x in a.agents})
        _, rb, _, _, ib = b.step({x: 1 for x in b.agents})
    assert ra == rb
    assert ia["household_0"]["episode"]["renewable_share_of_consumption"] == 0.0


def test_renewable_share_displaces_conventional_generation():
    e0 = GridMARLEnv(n_households=8, seed=0, day_variability=0.0)
    e4 = GridMARLEnv(n_households=8, seed=0, day_variability=0.0, renewable_share=0.4)
    for e in (e0, e4):
        e.reset(seed=1, options={"day_scale": 1.0, "wind_scale": 1.0})
        while e.agents:
            _, _, _, _, info = e.step({a: 0 for a in e.agents})
        e.ep = info["household_0"]["episode"]
    assert e4.ep["energy_conventional_mwh"] < 0.7 * e0.ep["energy_conventional_mwh"]
    assert (
        0.35 < e4.ep["renewable_share_of_consumption"] <= 0.41
    )  # ~= configured share (no surplus yet)
    assert e4.ep["net_import_peak_mw"] < e0.ep["net_import_peak_mw"]
    # energy balance: load + losses == conventional + renewables - surplus (per step sums)
    m = e4._mix_log
    lhs = sum(x["load_mw"] + x["losses_mw"] for x in m)
    rhs = sum(x["conv_mw"] + x["pv_mw"] + x["wind_mw"] for x in m)
    assert abs(lhs - rhs) < 1e-6 * max(1.0, abs(rhs))


def test_pv_heavy_high_share_creates_midday_surplus_and_stress():
    e = GridMARLEnv(
        n_households=8, seed=0, day_variability=0.0, renewable_share=0.8, wind_fraction=0.2
    )
    e.reset(seed=1, options={"day_scale": 1.0, "wind_scale": 1.0})
    loading = []
    while e.agents:
        _, _, _, _, info = e.step({a: 0 for a in e.agents})
        loading.append(info["household_0"]["loading_max"])
    ep = info["household_0"]["episode"]
    assert ep["renewable_surplus_mwh"] > 0.0  # exported at midday
    assert max(loading) > 1.0  # reverse flows overload a branch -> Phase-2 constraint case
