#!/usr/bin/env python3
"""Feasibility bound for the kWh-basis Jain target (task m4-t7 / m4-t10 evidence).

Scripted (non-learned) policies on the 20 protocol days show what *any*
controller can reach on this population:

* ``equal_fraction(k)``  every household curtails level k at the evening peak
  hours (16-20 h): equal *relative* burden, kWh burden proportional to
  flexibility (Jain_frac = 1 by construction);
* ``equal_kwh``         every household curtails the same kW at those hours,
  capped by the least flexible household (Jain_kWh = 1 by construction);
* ``kwh_blend(alpha)``  convex blend of the two curtailment vectors.

Writes ``results/tables/fairness_feasibility.csv`` and prints the best peak
reduction attainable at Jain_kWh >= 0.85.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "envs"):
    sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from govgrid_envs import GridMARLEnv  # noqa: E402

PEAK_HOURS = range(16, 21)


def run(env, alpha: float, level_frac: float, days):
    """alpha = 0: equal fraction; alpha = 1: equal kWh (continuous curtailment, bypasses levels)."""
    out = []
    for d in days:
        env.reset(seed=d, options={"day_scale": 1.0})
        while env.agents:
            t = env._t
            if t in PEAK_HOURS:
                flex = env._flex[:, t]
                frac_vec = np.full(env.n_households, level_frac)  # equal fraction
                kw_equal = np.full(env.n_households, (level_frac * flex).min())  # equal kW
                target_kw = (1 - alpha) * frac_vec * flex + alpha * kw_equal
                frac = np.clip(target_kw / np.maximum(flex, 1e-9), 0, env.max_curtail)
            else:
                frac = np.zeros(env.n_households)
            # map to the nearest discrete level per household
            lv = np.rint(frac / env.max_curtail * (env.n_levels - 1)).astype(int)
            _, _, _, _, info = env.step({a: int(lv[i]) for i, a in enumerate(env.possible_agents)})
        ep = info[env.possible_agents[0]]["episode"]
        out.append(
            (ep["peak_reduction_pct"], ep["jain_index"], ep["jain_frac"], ep["violation_rate"])
        )
    return np.mean(out, axis=0)


def main() -> int:
    env = GridMARLEnv(n_households=14, seed=42)
    days = [10_000 + k for k in range(20)]
    rows = []
    for level_frac in (0.3, 0.45, 0.6):
        for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
            pk, j, jf, v = run(env, alpha, level_frac, days)
            rows.append(
                {
                    "peak_level_frac": level_frac,
                    "alpha_equal_kwh": alpha,
                    "peak_reduction_pct": pk,
                    "jain_kwh": j,
                    "jain_frac": jf,
                    "violation_rate": v,
                }
            )
            print(
                f"frac {level_frac:.2f} alpha {alpha:.2f} | peak {pk:5.1f}% | Jain kWh {j:.3f} frac {jf:.3f} | viol {v:.2f}"
            )
    df = pd.DataFrame(rows)
    (REPO / "results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv(REPO / "results/tables/fairness_feasibility.csv", index=False)
    ok = df[df.jain_kwh >= 0.85]
    print(
        "best peak reduction with Jain_kWh >= 0.85:",
        f"{ok.peak_reduction_pct.max():.1f} %" if len(ok) else "none",
    )
    ds = env.dataset
    fe = ds.flexible_kw.sum(axis=1) * ds.dt_hours
    print("flexible kWh/home/day by household:", np.round(fe, 2).tolist())
    print("groups:", ds.groups)
    print("max/min flexible energy ratio:", round(fe.max() / fe.min(), 2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
