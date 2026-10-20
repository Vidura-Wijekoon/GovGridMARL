"""Generic MARL training loop (tasks m3-t5 / m3-t6).

Runs ``n_envs`` copies of the PettingZoo ParallelEnv in lockstep (episodes
are short and the power flow is C++-fast, so sequential stepping beats
process pools on a laptop), collects ``n_steps`` joint transitions per env,
computes GAE, calls ``agent.update()`` and logs:

* per update: mean agent return, peak reduction, Jain, violation rate,
  PPO losses, wall-clock, env-steps/s  -> ``<run_dir>/metrics.csv`` and the
  experiment tracker (MLflow by default, task m1-t4);
* every ``eval_every`` updates: a deterministic (greedy) evaluation on the
  fixed evaluation seeds (m3-t7 protocol) -> ``<run_dir>/eval.csv``;
* checkpoints ``<run_dir>/checkpoints/update_<k>.pt`` and ``final.pt``.

Convergence criterion (m3-t6, :func:`convergence_test`): the training curve
is declared converged when the fitted change of the smoothed mean return
over the last ``patience`` updates is within ``tol`` x |mean| *and* the
window standard deviation is below ``std_tol`` x |mean|. The flag is
written to ``summary.json`` and recomputed from ``metrics.csv`` by
``experiments/evaluate.py``.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from govgridmarl.agents.ppo_core import PPOFamilyAgent, compute_gae
from govgridmarl.utils.logging import get_logger

_log = get_logger("govgridmarl.trainer")


def _stack_obs(obs: dict[str, np.ndarray], agents: list[str]) -> np.ndarray:
    return np.stack([obs[a] for a in agents]).astype(np.float32)


class Trainer:
    """Runs rollouts, GAE and PPO updates for a :class:`PPOFamilyAgent`."""

    def __init__(
        self,
        env_fn,
        agent: PPOFamilyAgent,
        run_dir: str | Path,
        n_envs: int = 8,
        n_steps: int = 96,
        total_timesteps: int = 100_000,
        eval_every: int = 10,
        eval_episodes: int = 5,
        eval_seeds: tuple[int, ...] | None = None,
        checkpoint_every: int = 25,
        seed: int = 0,
        tracker=None,
        convergence: dict[str, float] | None = None,
        log_every: int = 1,
    ) -> None:
        self.envs = [env_fn() for _ in range(n_envs)]
        self.agent = agent
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.n_envs, self.n_steps = int(n_envs), int(n_steps)
        self.total_timesteps = int(total_timesteps)
        self.eval_every, self.eval_episodes = int(eval_every), int(eval_episodes)
        self.eval_seeds = (
            tuple(eval_seeds) if eval_seeds else tuple(10_000 + k for k in range(eval_episodes))
        )
        self.checkpoint_every = int(checkpoint_every)
        self.seed = int(seed)
        self.tracker = tracker
        self.log_every = int(log_every)
        self.conv = {"patience": 15, "tol": 0.05, "std_tol": 0.05, **(convergence or {})}
        e = self.envs[0]
        self.agents_list = list(e.possible_agents)
        self.N = len(self.agents_list)
        self.gamma, self.lam = agent.cfg.gamma, agent.cfg.gae_lambda
        # multi-objective (MOPPO, m4-t8) and constrained (Lagrangian, m5-t3) hooks
        self.K = int(getattr(agent, "n_objectives", 1))
        self.mo = self.K > 1
        self.cost_names: tuple[str, ...] = tuple(getattr(agent, "constraint_names", ()))
        self.cost_key = (
            "cost_ind" if getattr(agent, "cost_signal", "magnitude") == "indicator" else "cost_vec"
        )
        self.C = len(self.cost_names)
        self._w = dict(getattr(e, "w", {}))
        self._metrics_path = self.run_dir / "metrics.csv"
        self._eval_path = self.run_dir / "eval.csv"
        self.history: list[dict[str, Any]] = []
        self.eval_history: list[dict[str, Any]] = []

    # ---------------------------------------------------------------- rollout
    def _reset_all(self):
        self._obs, self._state = [], []
        for k, env in enumerate(self.envs):
            obs, _ = env.reset(seed=self.seed * 1000 + k)
            self._obs.append(_stack_obs(obs, self.agents_list))
            self._state.append(env.state())
        self._obs = np.stack(self._obs)
        self._state = np.stack(self._state)
        self._ep_returns = np.zeros((self.n_envs, self.N))
        self._ep_costs = np.zeros((self.n_envs, self.N, max(1, self.C)))
        self._finished: list[dict[str, Any]] = []
        self._lam = self.agent.sample_lambda(self.n_envs) if self.mo else None
        self._pref = self.agent.pref_vector(self._lam) if self.mo else None

    def _reward_vec(self, info: dict, rew: dict) -> np.ndarray:
        """Per-agent reward (N,) or objective vector (N, K) from one env step."""
        if not self.mo:
            r = np.array([rew[a] for a in self.agents_list], np.float32)
            if self.agent.team_reward:
                r[:] = r.mean()
            return r
        return np.stack(
            [self.agent.objective_vector(info[a]["objectives"], self._w) for a in self.agents_list]
        ).astype(np.float32)

    def _cost_vec(self, info: dict) -> np.ndarray:
        """Per-agent constraint costs (N, C) from one env step."""
        return np.array(
            [[float(info[a][self.cost_key][c]) for c in self.cost_names] for a in self.agents_list],
            np.float32,
        )

    def collect(self) -> dict[str, np.ndarray]:
        T, E, N, K, C = self.n_steps, self.n_envs, self.N, self.K, self.C
        vshape = (T, E, N, K) if self.mo else (T, E, N)
        buf = {
            "obs": np.zeros((T, E, N, self.agent.obs_dim), np.float32),
            "state": np.zeros((T, E, self.agent.state_dim), np.float32),
            "actions": np.zeros((T, E, N), np.int64),
            "logp": np.zeros((T, E, N), np.float32),
            "values": np.zeros(vshape, np.float32),
            "rewards": np.zeros(vshape, np.float32),
            "dones": np.zeros((T, E), np.float32),
        }
        if self.mo:
            buf["pref"] = np.zeros((T, E, self.agent.pref_dim), np.float32)
            buf["lam"] = np.zeros((T, E), np.float32)
        if C:
            buf["costs"] = np.zeros((T, E, N, C), np.float32)
            buf["cost_values"] = np.zeros((T, E, N, C), np.float32)
        ep_cost_sums: list[np.ndarray] = []
        for t in range(T):
            actions, logp, values = self.agent.act(self._obs, self._state, pref=self._pref)
            buf["obs"][t], buf["state"][t] = self._obs, self._state
            buf["actions"][t], buf["logp"][t], buf["values"][t] = actions, logp, values
            if self.mo:
                buf["pref"][t], buf["lam"][t] = self._pref, self._lam
            if C:
                buf["cost_values"][t] = self.agent.cost_values(self._obs, self._state, self._pref)
            for k, env in enumerate(self.envs):
                act = {a: int(actions[k, i]) for i, a in enumerate(self.agents_list)}
                obs, rew, term, trunc, info = env.step(act)
                r = self._reward_vec(info, rew)
                buf["rewards"][t, k] = r
                self._ep_returns[k] += self.agent.scalarise(r, self._lam[k]) if self.mo else r
                if C:
                    c = self._cost_vec(info)
                    buf["costs"][t, k] = c
                    self._ep_costs[k] += c
                done = all(trunc.values()) or all(term.values())
                buf["dones"][t, k] = float(done)
                if done:
                    ep = info[self.agents_list[0]]["episode"]
                    rec = {"return": float(self._ep_returns[k].mean()), **ep}
                    if self.mo:
                        rec["lambda"] = float(self._lam[k])
                    if C:
                        ep_cost_sums.append(self._ep_costs[k].mean(axis=0))
                        rec.update(
                            {
                                f"ep_cost_{n}": float(v)
                                for n, v in zip(self.cost_names, ep_cost_sums[-1])
                            }
                        )
                    self._finished.append(rec)
                    self._ep_returns[k] = 0.0
                    self._ep_costs[k] = 0.0
                    obs, _ = env.reset()
                    if self.mo:  # new preference for the new episode
                        self._lam[k] = self.agent.sample_lambda(1)[0]
                        self._pref[k] = self.agent.pref_vector(self._lam[k : k + 1])[0]
                self._obs[k] = _stack_obs(obs, self.agents_list)
                self._state[k] = env.state()
        last_values = self.agent.values(self._obs, self._state, self._pref)
        adv, ret = compute_gae(
            buf["rewards"], buf["values"], buf["dones"], last_values, self.gamma, self.lam
        )
        flat = {k: v.reshape((T * E,) + v.shape[2:]) for k, v in buf.items() if k != "dones"}
        if self.mo:
            adv = self.agent.scalarise(adv, buf["lam"][..., None])  # (T, E, N)
        flat["advantages"] = adv.reshape(T * E, N)
        flat["returns"] = ret.reshape((T * E,) + ret.shape[2:])
        if C:
            last_c = self.agent.cost_values(self._obs, self._state, self._pref)
            cadv, cret = compute_gae(
                buf["costs"], buf["cost_values"], buf["dones"], last_c, self.gamma, self.lam
            )
            flat["cost_advantages"] = cadv.reshape(T * E, N, C)
            flat["cost_returns"] = cret.reshape(T * E, N, C)
            flat["ep_cost_mean"] = (
                np.mean(ep_cost_sums, axis=0)
                if ep_cost_sums
                else buf["costs"].sum(axis=0).mean(axis=(0, 1)) * 0.0
            )
        return flat

    # ------------------------------------------------------------------- eval
    def evaluate(self, deterministic: bool = True, lam: float | None = None) -> dict[str, float]:
        env = self.envs[0]
        rets, eps = [], []
        pref = None
        if self.mo:
            lam = self.agent.eval_lambda if lam is None else float(lam)
            pref = self.agent.pref_vector(np.array([lam]))
        for s in self.eval_seeds:
            obs, _ = env.reset(seed=s, options={"day_scale": 1.0})
            total = np.zeros(self.N)
            while env.agents:
                o = _stack_obs(obs, self.agents_list)
                a, _, _ = self.agent.act(o, env.state(), deterministic=deterministic, pref=pref)
                obs, rew, _, _, info = env.step(
                    {ag: int(a[i]) for i, ag in enumerate(self.agents_list)}
                )
                total += [rew[ag] for ag in self.agents_list]
            rets.append(total.mean())
            eps.append(info[self.agents_list[0]]["episode"])
        # restore training stream for env 0
        obs, _ = env.reset()
        self._obs[0] = _stack_obs(obs, self.agents_list)
        self._state[0] = env.state()
        self._ep_returns[0] = 0.0
        return {
            "eval_return": float(np.mean(rets)),
            "eval_peak_reduction_pct": float(np.mean([e["peak_reduction_pct"] for e in eps])),
            "eval_jain": float(np.mean([e["jain_index"] for e in eps])),
            "eval_violation_rate": float(np.mean([e["violation_rate"] for e in eps])),
            "eval_cost_total": float(np.mean([e["cost_total"] for e in eps])),
            "eval_jain_frac": float(np.mean([e.get("jain_frac", np.nan) for e in eps])),
            "eval_vulnerable_burden_ratio": float(
                np.mean([e.get("vulnerable_burden_ratio", np.nan) for e in eps])
            ),
        }

    # -------------------------------------------------------------------- fit
    def fit(self) -> dict[str, Any]:
        self._reset_all()
        steps_per_update = self.n_steps * self.n_envs
        n_updates = max(1, self.total_timesteps // steps_per_update)
        _log.info(
            "Training %s: %d updates x %d env-steps (%d agents, %d envs)",
            self.agent.mode.upper(),
            n_updates,
            steps_per_update,
            self.N,
            self.n_envs,
        )
        t_start = time.perf_counter()
        best = -np.inf
        for u in range(1, n_updates + 1):
            t0 = time.perf_counter()
            batch = self.collect()
            t_env = time.perf_counter() - t0
            stats = self.agent.update(batch)
            t_upd = time.perf_counter() - t0 - t_env
            fin = self._finished
            self._finished = []
            row = {
                "update": u,
                "timesteps": u * steps_per_update,
                "episodes": len(fin),
                "return_mean": float(np.mean([f["return"] for f in fin])) if fin else np.nan,
                "return_std": float(np.std([f["return"] for f in fin])) if fin else np.nan,
                "peak_reduction_pct": (
                    float(np.mean([f["peak_reduction_pct"] for f in fin])) if fin else np.nan
                ),
                "jain": float(np.mean([f["jain_index"] for f in fin])) if fin else np.nan,
                "violation_rate": (
                    float(np.mean([f["violation_rate"] for f in fin])) if fin else np.nan
                ),
                "cost_total": float(np.mean([f["cost_total"] for f in fin])) if fin else np.nan,
                "jain_frac": (
                    float(np.mean([f.get("jain_frac", np.nan) for f in fin])) if fin else np.nan
                ),
                "vulnerable_burden_ratio": (
                    float(np.mean([f.get("vulnerable_burden_ratio", np.nan) for f in fin]))
                    if fin
                    else np.nan
                ),
                **{
                    f"ep_cost_{n}": (
                        float(np.mean([f.get(f"ep_cost_{n}", np.nan) for f in fin]))
                        if fin
                        else np.nan
                    )
                    for n in self.cost_names
                },
                **stats,
                "env_steps_per_s": steps_per_update / max(t_env, 1e-9),
                "t_env_s": t_env,
                "t_update_s": t_upd,
                "wall_s": time.perf_counter() - t_start,
            }
            self.history.append(row)
            self._append_csv(self._metrics_path, row)
            if self.tracker is not None:
                self.tracker.log_metrics(
                    {
                        k: v
                        for k, v in row.items()
                        if isinstance(v, (int, float))
                        and not (isinstance(v, float) and np.isnan(v))
                    },
                    step=row["timesteps"],
                )
            if u % self.log_every == 0:
                _log.info(
                    "upd %4d/%d | steps %7d | R %.3f | peak %.1f%% | J %.3f | viol %.2f | "
                    "kl %.4f ent %.3f | %.0f steps/s",
                    u,
                    n_updates,
                    row["timesteps"],
                    row["return_mean"],
                    row["peak_reduction_pct"],
                    row["jain"],
                    row["violation_rate"],
                    stats["approx_kl"],
                    stats["entropy"],
                    row["env_steps_per_s"],
                )
            if self.eval_every and (u % self.eval_every == 0 or u == n_updates):
                ev = {"update": u, "timesteps": row["timesteps"], **self.evaluate()}
                self.eval_history.append(ev)
                self._append_csv(self._eval_path, ev)
                if self.tracker is not None:
                    self.tracker.log_metrics(ev, step=ev["timesteps"])
                _log.info(
                    "   eval | R %.3f | peak %.1f%% | J %.3f | viol %.2f",
                    ev["eval_return"],
                    ev["eval_peak_reduction_pct"],
                    ev["eval_jain"],
                    ev["eval_violation_rate"],
                )
                if ev["eval_return"] > best:
                    best = ev["eval_return"]
                    self.agent.save(self.run_dir / "checkpoints" / "best.pt")
            if self.checkpoint_every and u % self.checkpoint_every == 0:
                self.agent.save(self.run_dir / "checkpoints" / f"update_{u}.pt")
        self.agent.save(self.run_dir / "checkpoints" / "final.pt")
        summary = self.summary()
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        if self.tracker is not None:
            self.tracker.log_metrics(
                {k: v for k, v in summary.items() if isinstance(v, (int, float))}
            )
        _log.info(
            "Done in %.1fs. converged=%s final eval R=%.3f peak=%.1f%%",
            summary["wall_s"],
            summary["converged"],
            summary.get("final_eval_return", np.nan),
            summary.get("final_eval_peak_reduction_pct", np.nan),
        )
        return summary

    # ---------------------------------------------------------------- utils
    def converged(self) -> tuple[bool, dict[str, float]]:
        return convergence_test([h["return_mean"] for h in self.history], **self.conv)

    def summary(self) -> dict[str, Any]:
        conv, diag = self.converged()
        last = self.history[-1] if self.history else {}
        first = next((h for h in self.history if not np.isnan(h["return_mean"])), {})
        out: dict[str, Any] = {
            "mode": self.agent.mode,
            "seed": self.seed,
            "updates": len(self.history),
            "timesteps": last.get("timesteps", 0),
            "wall_s": last.get("wall_s", 0.0),
            "first_return": first.get("return_mean", np.nan),
            "final_return": last.get("return_mean", np.nan),
            "final_peak_reduction_pct": last.get("peak_reduction_pct", np.nan),
            "final_jain": last.get("jain", np.nan),
            "final_violation_rate": last.get("violation_rate", np.nan),
            "converged": conv,
            **{f"conv_{k}": v for k, v in diag.items()},
            "mean_env_steps_per_s": (
                float(np.mean([h["env_steps_per_s"] for h in self.history]))
                if self.history
                else 0.0
            ),
        }
        if self.eval_history:
            ev = self.eval_history[-1]
            out.update({f"final_{k}": v for k, v in ev.items() if k.startswith("eval_")})
            out["best_eval_return"] = max(e["eval_return"] for e in self.eval_history)
        return out

    @staticmethod
    def _append_csv(path: Path, row: dict[str, Any]) -> None:
        new = not path.exists()
        with path.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)


def convergence_test(
    returns, patience: int = 15, tol: float = 0.05, std_tol: float = 0.05, smooth: int = 5
) -> tuple[bool, dict[str, float]]:
    """m3-t6 convergence criterion on a per-update mean-return series.

    Converged when, over the last ``patience`` updates, (a) the fitted change of
    the ``smooth``-point moving average is within ``tol`` x |window mean| and
    (b) the raw window standard deviation is below ``std_tol`` x |window mean|.
    The 5 % tolerances reflect the day-to-day demand noise (sigma_d = 0.10)
    which alone induces ~3 % dispersion of per-update returns.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    p = int(patience)
    if r.size < 2 * p:
        return False, {"slope_rel": float("nan"), "std_rel": float("nan")}
    w = r[-p:]
    k = min(int(smooth), p)
    sm = np.convolve(r, np.ones(k) / k, mode="valid")[-p:]
    slope = np.polyfit(np.arange(sm.size), sm, 1)[0] * sm.size  # change over window
    scale = max(abs(float(np.mean(w))), 1e-9)
    slope_rel, std_rel = float(abs(slope) / scale), float(np.std(w) / scale)
    ok = bool(slope_rel < tol and std_rel < std_tol)
    return ok, {"slope_rel": slope_rel, "std_rel": std_rel}


__all__ = ["Trainer", "convergence_test"]
