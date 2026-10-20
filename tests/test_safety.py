"""Safety constraints, Lagrangian controllers and N-1 contingencies (m5-t1 .. m5-t6, m5-t12)."""

import numpy as np
import pytest

from govgrid_envs import GridMARLEnv
from govgridmarl.agents import make_agent
from govgridmarl.agents.lagrangian import LagrangianController
from govgridmarl.safety import (
    ALL_CONSTRAINTS,
    ConstraintSet,
    blackout_cost,
    comfort_cost,
    fairness_cost,
    loading_cost,
    voltage_violation_cost,
)
from govgridmarl.training import Trainer


# --- m5-t1: cost functions -----------------------------------------------------
def test_cost_functions_zero_when_satisfied():
    assert voltage_violation_cost([0.96, 1.0, 1.09]) == 0.0
    assert loading_cost([0.2, 0.99]) == 0.0
    assert np.all(comfort_cost([1.0, 2.0], [10.0, 10.0], 0.3) == 0.0)
    assert blackout_cost(True) == (0.0, 0.0)
    assert fairness_cost(0.9) == 0.0


def test_cost_functions_positive_when_violated():
    assert voltage_violation_cost([0.93, 1.0, 1.12]) == pytest.approx(0.02 + 0.02)
    assert np.allclose(voltage_violation_cost([0.93, 1.0, 1.12], per_bus=True), [0.02, 0.0, 0.02])
    assert loading_cost([0.2, 1.3]) == pytest.approx(0.3)
    assert np.allclose(comfort_cost([5.0, 2.0], [10.0, 10.0], [0.3, 0.4]), [0.2, 0.0])
    assert blackout_cost(False) == (1.0, 2.0)
    assert fairness_cost(0.65, 0.85, 24) == pytest.approx(0.2 / 24)


def test_constraint_set():
    cs = ConstraintSet(["voltage", "fairness"], {"fairness": 0.01})
    assert cs.limits.tolist() == [0.0, 0.01]
    assert not cs.is_safety_only
    assert cs.vector({"voltage": 0.1, "loading": 9, "comfort": 9, "fairness": 0.2}).tolist() == [
        pytest.approx(0.1),
        pytest.approx(0.2),
    ]
    with pytest.raises(ValueError):
        ConstraintSet(["frequency"])


# --- m5-t2: environment cost vector -----------------------------------------------
def test_env_cost_vector_and_violation_accounting():
    e = GridMARLEnv(n_households=6, seed=0, use_power_flow=False, day_variability=0.0)
    e.reset(seed=1)
    top = e.n_levels - 1
    while e.agents:
        _, _, _, _, info = e.step({a: top for a in e.agents})
    i0 = info["household_0"]
    assert set(i0["cost_vec"]) == set(ALL_CONSTRAINTS)
    assert i0["cost_vec"]["comfort"] == pytest.approx(i0["cost_comfort_i"])
    ep = i0["episode"]
    assert ep["violations_by_constraint"]["comfort"] > 0
    assert ep["violations_by_constraint"]["voltage"] == 0  # no power flow -> no grid costs
    assert ep["violations"] >= ep["violations_by_constraint"]["comfort"]
    # fairness is measured but is not a safety violation
    assert "fairness" in ep["violations_by_constraint"]


# --- m5-t3 / m5-t6: multiplier dynamics ------------------------------------------
def test_dual_ascent_grows_then_relaxes():
    c = LagrangianController(["loading"], method="lagrangian", lr=0.5)
    c.update([1.0])
    c.update([1.0])
    assert c.multipliers[0] == pytest.approx(1.0)
    for _ in range(3):
        c.update([0.0])  # satisfied: pure dual ascent cannot go below zero but never decreases
    assert c.multipliers[0] == pytest.approx(1.0)
    c.update([-1.0])  # cost below budget (budget > 0 case) -> decreases
    assert c.multipliers[0] == pytest.approx(0.5)


def test_pid_lagrangian_reacts_and_decays():
    c = LagrangianController(["loading"], method="pid_lagrangian", kp=1.0, ki=0.1, kd=0.5)
    r = c.update([1.0])
    # kp*1 + ki*1 + kd*(1-0)
    assert r["lambda_loading"] == pytest.approx(1.0 + 0.1 + 0.5)
    c.update([0.0])  # proportional and derivative parts drop, integral stays
    assert c.multipliers[0] == pytest.approx(max(0.0, 0.0 + 0.1 * 1.0 + 0.5 * (0.0 - 1.0)))
    d = c.state_dict()
    c2 = LagrangianController(["loading"], method="pid_lagrangian")
    c2.load_state_dict(d)
    assert np.allclose(c2.integral, c.integral)


def test_constrained_agents_train_and_roundtrip(tmp_path):
    def env_fn():
        return GridMARLEnv(n_households=4, seed=0, use_power_flow=False, day_variability=0.0)

    e = env_fn()
    e.reset(seed=0)
    for name in ("mappo_lagrangian", "ppo_lagrangian"):
        ag = make_agent(
            name,
            17,
            e.state_dim,
            4,
            e.n_levels,
            cfg={"hidden": [32, 32], "n_epochs": 2, "constraints": ["comfort", "fairness"]},
        )
        assert ag.n_costs == 2 and ag.constraint_names == ("comfort", "fairness")
        tr = Trainer(
            env_fn,
            ag,
            tmp_path / name,
            n_envs=2,
            n_steps=24,
            total_timesteps=96,
            eval_every=2,
            eval_episodes=1,
            tracker=None,
        )
        s = tr.fit()
        assert s["updates"] == 2
        assert "lambda_comfort" in tr.history[-1]
        assert np.isfinite(tr.history[-1]["ep_cost_comfort"])
        from govgridmarl.agents.ppo_core import PPOFamilyAgent

        ag2 = PPOFamilyAgent.from_checkpoint(tmp_path / name / "checkpoints" / "final.pt")
        assert np.allclose(ag2.multipliers, ag.multipliers)
        obs = np.stack([e._obs(i) for i in range(4)])
        a1, _, _ = ag.act(obs, e.state(), deterministic=True)
        a2, _, _ = ag2.act(obs, e.state(), deterministic=True)
        assert (a1 == a2).all()


# --- m5-t12: N-1 contingency ---------------------------------------------------
def test_n1_contingency_changes_flows():
    base = GridMARLEnv(n_households=8, seed=0, day_variability=0.0)
    n1 = GridMARLEnv(n_households=8, seed=0, day_variability=0.0, contingency=0)
    assert n1.contingency == 0
    for e in (base, n1):
        e.reset(seed=1, options={"day_scale": 1.0})
        lm = []
        while e.agents:
            _, _, _, _, info = e.step({a: 0 for a in e.agents})
            lm.append(info["household_0"]["loading_max"])
        e.lmax = max(lm)
    assert n1.lmax != pytest.approx(base.lmax)
    assert n1.lmax > base.lmax  # losing a branch loads the remaining ones more
    n1.set_contingency(None)
    n1.reset(seed=1, options={"day_scale": 1.0})
    lm = []
    while n1.agents:
        _, _, _, _, info = n1.step({a: 0 for a in n1.agents})
        lm.append(info["household_0"]["loading_max"])
    assert max(lm) == pytest.approx(base.lmax, rel=1e-6)
