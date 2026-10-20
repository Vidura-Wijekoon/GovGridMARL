"""PPO-family agent tests (m3-t1 .. m3-t3) — fast, power flow disabled."""

import numpy as np
import pytest

from govgrid_envs import GridMARLEnv
from govgridmarl.agents import REGISTRY, make_agent
from govgridmarl.agents.heuristic import NoControl, RuleBasedDR
from govgridmarl.agents.ppo_core import compute_gae
from govgridmarl.training import Trainer


def env_fn():
    return GridMARLEnv(n_households=4, seed=0, use_power_flow=False, day_variability=0.0)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_act_shapes_and_update(name, tmp_path):
    e = env_fn()
    e.reset(seed=0)
    ag = make_agent(name, 17, e.state_dim, 4, e.n_levels, cfg={"hidden": [32, 32], "n_epochs": 2})
    assert ag.mode == name
    obs = np.stack([e._obs(i) for i in range(4)])
    a, lp, v = ag.act(obs, e.state())
    assert a.shape == (4,) and lp.shape == (4,) and v.shape[0] == 4  # MOPPO: (4, K)
    assert all(0 <= x < e.n_levels for x in a)
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
    assert (tmp_path / name / "metrics.csv").exists()
    assert (tmp_path / name / "checkpoints" / "final.pt").exists()
    assert np.isfinite(s["final_eval_return"])


def test_checkpoint_roundtrip(tmp_path):
    from govgridmarl.agents.ppo_core import PPOFamilyAgent

    e = env_fn()
    e.reset(seed=0)
    ag = make_agent("mappo", 17, e.state_dim, 4, 5, cfg={"hidden": [16]})
    obs = np.stack([e._obs(i) for i in range(4)])
    ag.save(tmp_path / "a.pt")
    ag2 = PPOFamilyAgent.from_checkpoint(tmp_path / "a.pt")
    a1, _, v1 = ag.act(obs, e.state(), deterministic=True)
    a2, _, v2 = ag2.act(obs, e.state(), deterministic=True)
    assert (a1 == a2).all() and np.allclose(v1, v2)


def test_no_param_sharing_ippo():
    e = env_fn()
    e.reset(seed=0)
    ag = make_agent("ippo", 17, e.state_dim, 4, 5, cfg={"share_params": False, "hidden": [16]})
    assert len(ag.nets) == 4
    obs = np.stack([e._obs(i) for i in range(4)])
    a, lp, v = ag.act(obs, e.state())
    assert a.shape == (4,)


def test_gae_matches_montecarlo_when_lambda_one():
    T, E, N = 6, 1, 2
    r = np.ones((T, E, N), np.float32)
    v = np.zeros((T, E, N), np.float32)
    d = np.zeros((T, E), np.float32)
    d[-1] = 1.0
    adv, ret = compute_gae(r, v, d, np.zeros((E, N), np.float32), gamma=1.0, lam=1.0)
    assert np.allclose(ret[:, 0, 0], np.arange(T, 0, -1))
    # a done at step 2 cuts the bootstrap
    d[2] = 1.0
    adv, ret = compute_gae(r, v, d, np.zeros((E, N), np.float32), gamma=1.0, lam=1.0)
    assert np.allclose(ret[:3, 0, 0], [3, 2, 1])


def test_heuristic_and_none():
    e = env_fn()
    obs, _ = e.reset(seed=0)
    h, n = RuleBasedDR(), NoControl()
    o = np.stack([obs[a] for a in e.possible_agents])
    ah, _, _ = h.act(o)
    an, _, _ = n.act(o)
    assert an.sum() == 0 and ah.shape == (4,)
    # heuristic curtails at the evening price peak
    for _ in range(17):
        obs, *_ = e.step({a: 0 for a in e.agents})
    o = np.stack([obs[a] for a in e.possible_agents])
    ah, _, _ = h.act(o)
    assert ah.max() == e.n_levels - 1
