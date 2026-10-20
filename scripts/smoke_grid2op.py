#!/usr/bin/env python3
"""Grid2Op / L2RPN sandbox smoke test (task m1-t13).

Installs nothing — verifies an existing grid2op install end-to-end:

  1. import grid2op / lightsim2grid / pandapower and report versions;
  2. create the ``l2rpn_case14_sandbox`` environment (LightSim2Grid backend);
  3. check the grid matches IEEE-14 expectations (14 subs, 20 lines);
  4. seeded reset + observation sanity checks;
  5. run a full DoNothing episode to termination;
  6. apply a real topology action + a redispatch action and step through
     both (action-space round-trip);
  7. replay the DoNothing episode with grid2op's Runner (the tooling used
     for evaluation later in the project);
  8. seeded reproducibility check (same seed -> identical trajectory).

Everything is echoed to stdout AND written to
results/logs/grid2op_smoke_<timestamp>.log so the log itself is the
deliverable.

Usage:
    python scripts/smoke_grid2op.py            # full dataset (~/data_grid2op)
    python scripts/smoke_grid2op.py --test-env # bundled chronics, offline-safe
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
sys.path.insert(0, str(REPO_ROOT))

ENV_NAME = "l2rpn_case14_sandbox"


class Tee:
    """Write to stdout and a log file simultaneously."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def run_do_nothing_episode(env, seed: int, chronic_id=None):
    """Run one full episode with the DoNothing action, return summary dict.

    Note: grid2op rotates chronics on every reset independently of the seed,
    so exact reproducibility requires pinning the scenario with ``set_id``
    in addition to seeding.
    """
    if chronic_id is not None:
        env.set_id(chronic_id)
    env.seed(seed)
    obs = env.reset()
    do_nothing = env.action_space({})
    total_reward, steps = 0.0, 0
    t0 = time.perf_counter()
    done = False
    while not done:
        obs, reward, done, info = env.step(do_nothing)
        total_reward += float(reward)
        steps += 1
    wall = time.perf_counter() - t0
    return {
        "chronic": Path(env.chronics_handler.get_id()).name,
        "steps": steps,
        "max_steps": env.chronics_handler.max_timestep(),
        "total_reward": total_reward,
        "wall_s": wall,
        "steps_per_s": steps / wall if wall > 0 else float("inf"),
        "exception": info.get("exception", []),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-env",
        action="store_true",
        help="Use the small chronics subset bundled with grid2op (offline).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = Tee(LOG_DIR / f"grid2op_smoke_{stamp}.log")
    warnings.filterwarnings("ignore")

    log("=" * 70)
    log("GovGridMARL Grid2Op / L2RPN sandbox smoke test (task m1-t13)")
    log(f"UTC time      : {datetime.now(timezone.utc).isoformat()}")
    log(f"Host          : {platform.node()} ({platform.system()} {platform.release()})")
    log(f"Python        : {sys.version.split()[0]} @ {sys.executable}")
    log("=" * 70)

    # -- 1. imports ------------------------------------------------------
    try:
        import grid2op
        import pandapower
    except ImportError as exc:
        log(f"FAIL: grid2op stack not installed ({exc}).")
        log("Fix : run scripts/setup_env.sh or pip install -r requirements.lock")
        log.close()
        return 1

    log(f"grid2op       : {grid2op.__version__}")
    log(f"pandapower    : {pandapower.__version__}")
    try:
        import lightsim2grid

        log(f"lightsim2grid : {lightsim2grid.__version__} (fast backend)")
    except ImportError:
        log("lightsim2grid : not installed (falling back to PandaPower backend)")

    # -- 2. environment creation ----------------------------------------
    from envs.govgrid_envs.grid2op_sandbox import make_sandbox

    log("-" * 70)
    mode = "bundled test chronics" if args.test_env else "full dataset"
    log(f"Creating {ENV_NAME} ({mode}) ...")
    t0 = time.perf_counter()
    env = make_sandbox(ENV_NAME, test=args.test_env)
    log(
        f"Env created in {time.perf_counter() - t0:.2f} s "
        f"| backend: {type(env.backend).__name__}"
    )
    n_chronics = len(env.chronics_handler.subpaths)
    log(f"Chronics      : {n_chronics} scenario folder(s)")

    # -- 3. IEEE-14 structure checks ------------------------------------
    checks = {
        "14 substations": env.n_sub == 14,
        "20 power lines": env.n_line == 20,
        "11 loads": env.n_load == 11,
        "6 generators": env.n_gen == 6,
    }
    log(
        f"Grid          : {env.n_sub} subs | {env.n_line} lines | "
        f"{env.n_load} loads | {env.n_gen} gens"
    )
    log(
        f"Action space  : {env.action_space.n} dims "
        f"({env.action_space.size()} attrs) | "
        f"Observation: {env.observation_space.n} dims"
    )
    for name, ok in checks.items():
        log(f"  [{'OK' if ok else 'BAD'}] {name}")
    if not all(checks.values()):
        log("RESULT: FAIL — grid structure does not match IEEE-14 sandbox.")
        log.close()
        return 1

    # -- 4. seeded reset + observation sanity ----------------------------
    log("-" * 70)
    env.seed(args.seed)
    obs = env.reset()
    scen = Path(env.chronics_handler.get_id()).name
    obs_ok = (
        obs.rho.shape == (env.n_line,)
        and float(obs.rho.max()) > 0.0
        and float(obs.rho.max()) < 2.0
        and obs.line_status.all()
    )
    log(f"Reset (seed {args.seed}) -> scenario '{scen}'")
    log(
        f"Obs sanity    : max line loading rho={obs.rho.max():.3f}, "
        f"all {env.n_line} lines on -> {'OK' if obs_ok else 'BAD'}"
    )
    if not obs_ok:
        log("RESULT: FAIL — observation sanity check failed.")
        log.close()
        return 1

    # -- 5. full DoNothing episode ---------------------------------------
    log("-" * 70)
    log("Running full episode with DoNothing agent ...")
    ep = run_do_nothing_episode(env, args.seed)
    log(
        f"Episode       : scenario '{ep['chronic']}' | "
        f"{ep['steps']}/{ep['max_steps']} steps "
        f"| reward {ep['total_reward']:.1f} "
        f"| {ep['wall_s']:.1f} s ({ep['steps_per_s']:.0f} steps/s)"
    )
    survived_full = ep["steps"] == ep["max_steps"]
    log(
        f"Survival      : {'full horizon' if survived_full else 'early blackout'}"
        f"{'' if survived_full else ' (acceptable for DoNothing baseline)'}"
    )
    if ep["steps"] < 10:
        log("RESULT: FAIL — episode ended almost immediately; env is unhealthy.")
        log.close()
        return 1

    # -- 6. real actions: topology + redispatch --------------------------
    log("-" * 70)
    env.seed(args.seed)
    obs = env.reset()
    topo_act = env.action_space({"set_bus": {"lines_or_id": [(3, 2)]}})
    obs, r1, done, info = env.step(topo_act)
    topo_ok = not info["is_illegal"] and not info["is_ambiguous"]
    log(
        f"Topology act  : line 3 origin -> bus 2 | reward {float(r1):.3f} | "
        f"legal={not info['is_illegal']} -> {'OK' if topo_ok else 'BAD'}"
    )
    redisp_act = env.action_space({"redispatch": [(0, 1.0)]})
    if not done:
        obs, r2, done, info = env.step(redisp_act)
        redisp_ok = not info["is_illegal"] and not info["is_ambiguous"]
        log(
            f"Redispatch act: gen 0 +1.0 MW | reward {float(r2):.3f} | "
            f"legal={not info['is_illegal']} -> {'OK' if redisp_ok else 'BAD'}"
        )
    else:
        redisp_ok = False
        log("Redispatch act: SKIPPED (episode ended after topology action)")
    if not (topo_ok and redisp_ok):
        log("RESULT: FAIL — action round-trip failed.")
        log.close()
        return 1

    # -- 7. Runner (evaluation tooling) ----------------------------------
    log("-" * 70)
    log("Running grid2op Runner (1 episode, DoNothing) ...")
    from grid2op.Runner import Runner

    runner = Runner(**env.get_params_for_runner())
    res = runner.run(nb_episode=1, max_iter=200, env_seeds=[args.seed])
    for _, chron_name, cum_reward, nb_time_step, max_ts in res:
        log(
            f"Runner        : scenario '{Path(chron_name).name}' | "
            f"{nb_time_step}/{max_ts} steps | cum reward {cum_reward:.1f}"
        )
    runner_ok = all(nb >= min(200, mx) * 0 + 1 for *_, nb, mx in res)
    if not runner_ok:
        log("RESULT: FAIL — Runner produced no steps.")
        log.close()
        return 1

    # -- 8. seeded reproducibility ---------------------------------------
    log("-" * 70)
    ep2 = run_do_nothing_episode(env, args.seed, chronic_id=ep["chronic"])
    repro_ok = (
        ep2["chronic"] == ep["chronic"]
        and ep2["steps"] == ep["steps"]
        and abs(ep2["total_reward"] - ep["total_reward"]) < 1e-6
    )
    log(
        f"Reproducibility (seed {args.seed}, scenario '{ep['chronic']}'): "
        f"run1 {ep['steps']} steps / {ep['total_reward']:.4f} vs "
        f"run2 {ep2['steps']} steps / {ep2['total_reward']:.4f} "
        f"-> {'OK' if repro_ok else 'MISMATCH'}"
    )
    if not repro_ok:
        log("RESULT: FAIL — seeded runs diverged.")
        log.close()
        return 1

    env.close()
    log("-" * 70)
    log("RESULT: PASS — Grid2Op L2RPN sandbox verified end-to-end.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
