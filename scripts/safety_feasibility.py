#!/usr/bin/env python3
"""Feasibility bound for "zero comfort violations" (task m5-t8 / m5-t11 evidence).

Scripted policies that respect every household's comfort bound *by
construction*: each household curtails at the peak hours (16-20 h) at the
largest level whose daily curtailed share of flexible energy stays below its
group tolerance theta_g (0.30 / 0.40 / 0.50), optionally scaled by ``safety``
(< 1 leaves a margin for day-to-day noise). Reports the peak reduction and
violation rate such a policy achieves on the 20 protocol days, i.e. the most
any controller can shave without violating comfort.

Writes ``results/tables/safety_feasibility.csv``.
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

PEAK_HOURS = list(range(16, 21))


def run(env, hours, safety: float, days):
    out = []
    n = env.n_households
    for d in days:
        env.reset(seed=d, options={"day_scale": 1.0})
        flex_day = env._flex.sum(axis=1) * env.dt  # kWh/home flexible energy today
        flex_peak = env._flex[:, hours].sum(axis=1) * env.dt
        # curtailment fraction kappa at the peak hours with kappa * flex_peak <= safety*theta*flex_day
        kappa = np.minimum(
            env.max_curtail, safety * env.comfort_tol * flex_day / np.maximum(flex_peak, 1e-9)
        )
        while env.agents:
            t = env._t
            frac = kappa if t in hours else np.zeros(n)
            lv = np.floor(frac / env.max_curtail * (env.n_levels - 1) + 1e-9).astype(int)
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
    for hours, label in ((PEAK_HOURS, "16-20h"), ([17, 18, 19], "17-19h"), ([18, 19], "18-19h")):
        for safety in (1.0, 0.9, 0.8):
            pk, j, jf, v = run(env, hours, safety, days)
            rows.append(
                {
                    "hours": label,
                    "safety_margin": safety,
                    "peak_reduction_pct": pk,
                    "jain_kwh": j,
                    "jain_frac": jf,
                    "violation_rate": v,
                }
            )
            print(
                f"{label} margin {safety:.1f} | peak {pk:5.1f}% | Jain kWh {j:.3f} frac {jf:.3f} | viol {v:.3f}"
            )
    df = pd.DataFrame(rows)
    df.to_csv(REPO / "results/tables/safety_feasibility.csv", index=False)
    ok = df[df.violation_rate == 0.0]
    print(
        "best peak reduction with zero violations:",
        f"{ok.peak_reduction_pct.max():.1f} %" if len(ok) else "none",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
