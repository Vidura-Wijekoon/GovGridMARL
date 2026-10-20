#!/usr/bin/env python3
"""One-click showcase: replay a day on IEEE-14 and open the interactive status page.

What it does (in order):
  1. checks the Python stack (torch, pettingzoo, pandapower, lightsim2grid);
  2. optionally runs the 14-check environment validation (``--validate``);
  3. optionally runs a short MAPPO training smoke (``--train``, ~2 min CPU) when no
     trained checkpoint exists yet;
  4. replays one held-out evaluation day under three policies — no control,
     rule-based DR, and the newest trained MAPPO checkpoint (falls back to the
     rule-based policy if none is found) — using the real AC power flow;
  5. writes ``docs/status/govgridmarl_status.html`` (self-contained, data inlined)
     and opens it in the default browser.

Usage:
    python scripts/showcase.py                  # replay + open page
    python scripts/showcase.py --validate       # also run validate_env first
    python scripts/showcase.py --train          # train a quick MAPPO if no checkpoint
    python scripts/showcase.py --no-browser     # just write the page
    python scripts/showcase.py --day 10003      # another evaluation day

Windows users: ``Start_Show.bat`` in the repo root does all of this.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import warnings
import webbrowser
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "envs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

TEMPLATE = REPO / "scripts" / "templates" / "status_page.html"
OUT_DIR = REPO / "docs" / "status"


def banner(msg: str) -> None:
    print("\n" + "=" * 72 + f"\n{msg}\n" + "=" * 72)


def check_stack() -> bool:
    banner("1/4  Checking the Python stack")
    ok = True
    for mod in ("numpy", "torch", "pettingzoo", "gymnasium", "pandapower", "lightsim2grid"):
        try:
            m = __import__(mod)
            print(f"  [ok]   {mod:14s} {getattr(m, '__version__', '')}")
        except Exception as exc:  # noqa: BLE001
            level = "warn" if mod == "lightsim2grid" else "FAIL"
            print(f"  [{level}] {mod:14s} {exc}")
            ok &= mod == "lightsim2grid"
    if not ok:
        print(
            "\n  Install the stack first:  pip install -r requirements-show.txt && pip install -e .   (Windows; or scripts/setup_env.sh)"
        )
    return ok


def newest_checkpoint() -> Path | None:
    cands = [
        Path(p)
        for p in glob.glob(
            str(REPO / "results" / "runs" / "mappo" / "*" / "checkpoints" / "final.pt")
        )
    ]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def quick_train() -> None:
    banner("2b/4  No MAPPO checkpoint found — running a 2-minute training smoke")
    cmd = [
        sys.executable,
        str(REPO / "experiments" / "train.py"),
        "+experiment=baseline_cloud",
        "agent=mappo",
        "seed=0",
        "train.total_timesteps=30720",
        "train.eval_every=10",
        "tracking.backend=none",
        "run_tag=showcase",
    ]
    subprocess.run(cmd, cwd=REPO, check=False)


def _rollout(env, ag, day: int, detailed: bool = True) -> dict:
    """One evaluation day; returns per-step records (+ mix) and the episode summary."""
    import numpy as np

    A = env.possible_agents
    obs, _ = env.reset(seed=day, options={"day_scale": 1.0, "wind_scale": 1.0})
    steps, mix, lmax = [], [], 0.0
    pp_pf = None
    if detailed:  # pandapower twin for the Newton-Raphson iteration count
        from govgrid_envs.ieee14 import Ieee14PowerFlow

        pp_pf = Ieee14PowerFlow(env.n_households, env.rating_factor, backend="pandapower")
    while env.agents:
        o = np.stack([obs[a] for a in A])
        a, _, _ = ag.act(o, env.state(), deterministic=True)
        obs, r, _, _, info = env.step({ag_: int(a[i]) for i, ag_ in enumerate(A)})
        pf, i0, t = env._last_pf, info[A[0]], info[A[0]]["t"]
        lmax = max(lmax, i0["loading_max"])
        mix.append(i0["mix"])
        if not detailed:
            continue
        n_iter = -1
        if pp_pf is not None:
            pv_mw, wind_mw = env.renewable_injection(t)
            bg = env.background_factor * env._bg_shape[t]
            n_iter = pp_pf.solve(
                env._served * env.homes_per_agent / 1000.0, bg, pv_mw=pv_mw, wind_mw=wind_mw
            ).n_iter
        steps.append(
            {
                "t": t,
                "actions": [int(x) for x in a],
                "served_kw": [round(float(x), 3) for x in env._served],
                "base_kw": [round(float(x), 3) for x in env._base[:, t]],
                "vm": [round(float(x), 4) for x in pf.vm_pu],
                "line_loading": [round(float(x), 3) for x in pf.line_loading],
                "trafo_loading": [round(float(x), 3) for x in pf.trafo_loading],
                "p_res_mw": round(i0["p_res_mw"], 3),
                "p_base_mw": round(i0["p_base_mw"], 3),
                "team_reward": round(i0["team_reward"], 4),
                "burden": [round(float(x), 3) for x in env._burden],
                "violation": bool(i0["violation"]),
                "cost_comfort": round(i0["cost_comfort"], 4),
                "price": float(env.price[t]),
                "mix": {k: round(float(v), 3) for k, v in i0["mix"].items()},
                "nr_iter": int(n_iter),
                "gen_p_mw": [round(float(x), 2) for x in pf.gen_p_mw],
            }
        )
    ep = info[A[0]]["episode"]
    ep = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in ep.items()}
    return {
        "steps": steps,
        "episode": ep,
        "mix": [{k: round(float(v), 3) for k, v in m.items()} for m in mix],
        "loading_max": round(lmax, 3),
    }


# (label, renewable_share, wind_fraction, group). "share" sweeps the renewable share
# at a 50/50 wind/PV split; "wind" sweeps the wind fraction at 80 % share
# (supervisor: "e.g. 70-80 % wind").
ENERGY_SCENARIOS = [
    ("0 % (baseline)", 0.0, 0.5, "share"),
    ("20 %", 0.2, 0.5, "share"),
    ("40 %", 0.4, 0.5, "share"),
    ("60 %", 0.6, 0.5, "share"),
    ("80 %", 0.8, 0.5, "share"),
    ("PV-heavy · 20 % wind", 0.8, 0.2, "wind"),
    ("70 % wind", 0.8, 0.7, "wind"),
    ("80 % wind", 0.8, 0.8, "wind"),
    ("90 % wind", 0.8, 0.9, "wind"),
]


def replay(day: int, ckpt: Path | None) -> dict:
    from govgrid_envs import GridMARLEnv
    from govgridmarl.agents.heuristic import NoControl, RuleBasedDR

    env = GridMARLEnv(n_households=14, seed=42)
    net = env._pf.net
    grid = {
        "buses": [{"id": int(i), "vn_kv": float(v)} for i, v in zip(net.bus.index, net.bus.vn_kv)],
        "lines": [
            {"from": int(a), "to": int(b)} for a, b in zip(net.line.from_bus, net.line.to_bus)
        ],
        "trafos": [
            {"from": int(a), "to": int(b)} for a, b in zip(net.trafo.hv_bus, net.trafo.lv_bus)
        ],
        "gens": sorted(
            set(net.gen.bus.astype(int).tolist() + net.ext_grid.bus.astype(int).tolist())
        ),
        "bg_loads": [
            {"bus": int(b), "p_mw": float(p)}
            for b, p in zip(net.load.loc[env._pf.background_ids].bus, env._pf.background_p)
        ],
        "n_bus": 14,
    }
    hh = [
        {
            "id": i,
            "bus": int(env.household_buses[i]),
            "group": env.groups[i],
            "daily_kwh": round(float(env.dataset.meta[i]["daily_kwh"]), 2),
            "comfort_tol": float(env.comfort_tol[i]),
        }
        for i in range(14)
    ]
    pols = {"none": NoControl(), "heuristic": RuleBasedDR()}
    ckpt_label = "no trained checkpoint — MAPPO slot shows the rule-based policy"
    if ckpt is not None:
        from govgridmarl.agents.ppo_core import PPOFamilyAgent

        pols["mappo"] = PPOFamilyAgent.from_checkpoint(ckpt)
        ckpt_label = f"MAPPO {ckpt.parents[1].name} ({ckpt.parents[2].name})"
    else:
        pols["mappo"] = RuleBasedDR()
    runs = {}
    for name, ag in pols.items():
        runs[name] = _rollout(env, ag, day, detailed=True)
        ep = runs[name]["episode"]
        print(
            f"  {name:9s} peak reduction {ep['peak_reduction_pct']:5.1f} %  Jain {ep['jain_index']:.3f}  "
            f"violations {ep['violation_rate']:.2f}  cost {ep['cost_total']:,.0f}"
        )
    # energy-share scenarios (generation mix), all three policies
    scenarios = []
    for label, rs, wf, group in ENERGY_SCENARIOS:
        e = GridMARLEnv(n_households=14, seed=42, renewable_share=rs, wind_fraction=wf)
        sc = {
            "key": f"rs{int(rs * 100)}_w{int(wf * 100)}",
            "label": label,
            "renewable_share": rs,
            "wind_fraction": wf,
            "group": group,
            "runs": {},
        }
        for name, ag in pols.items():
            r = _rollout(e, ag, day, detailed=False)
            sc["runs"][name] = {
                "episode": r["episode"],
                "mix": r["mix"],
                "loading_max": r["loading_max"],
            }
        m = sc["runs"]["mappo"]["episode"]
        print(
            f"  energy share {label:22s} MAPPO: consumed {m['renewable_share_of_consumption'] * 100:4.0f} %  surplus {m['renewable_surplus_mwh']:7.1f} MWh  "
            f"net-import peak {m['net_import_peak_mw']:6.1f} MW  peak red. {m['peak_reduction_pct']:4.1f} %  max loading {sc['runs']['mappo']['loading_max']:.2f}"
        )
        scenarios.append(sc)
    curves = {}
    for f in sorted(glob.glob(str(REPO / "results" / "runs" / "*" / "seed*" / "metrics.csv"))):
        import pandas as pd

        d = pd.read_csv(f)
        m, sd = Path(f).parents[1].name, Path(f).parent.name
        curves.setdefault(m, {})[sd] = {
            "timesteps": d.timesteps.tolist(),
            "return": d.return_mean.round(3).tolist(),
            "peak": d.peak_reduction_pct.round(2).tolist(),
            "jain": d.jain.round(3).tolist(),
            "viol": d.violation_rate.round(3).tolist(),
            "entropy": d.entropy.round(3).tolist(),
        }
    ga_path = REPO / "results" / "tables" / "grid_analysis.json"
    if not ga_path.exists():
        print("  grid_analysis.json missing — running scripts/grid_analysis.py")
        subprocess.run(
            [sys.executable, str(REPO / "scripts" / "grid_analysis.py")], cwd=REPO, check=False
        )
    grid_analysis = json.loads(ga_path.read_text()) if ga_path.exists() else None
    protocol = _protocol_results()
    return {
        "grid": grid,
        "households": hh,
        "price": env.price.tolist(),
        "pv": env.pv.round(3).tolist(),
        "runs": runs,
        "curves": curves,
        "energy_share": {"scenarios": scenarios},
        "grid_analysis": grid_analysis,
        "protocol": protocol,
        "meta": {
            "date": date.today().isoformat(),
            "dataset": env.dataset.summary(),
            "pf_backend": env._pf.backend,
            "ckpt": ckpt_label,
            "day": day,
        },
    }


def _protocol_results() -> dict | None:
    """20-day benchmark table + gate verdict from experiments/evaluate.py outputs, if present."""
    bench = REPO / "results" / "tables" / "benchmark.csv"
    if not bench.exists():
        return None
    import pandas as pd

    df = pd.read_csv(bench)
    prof_path = REPO / "results" / "tables" / "compute_profile.json"
    prof = json.loads(prof_path.read_text()) if prof_path.exists() else {}
    gate = False
    if "mappo" in set(df.policy):
        m = df[df.policy == "mappo"].iloc[0]
        h = df[df.policy == "heuristic"].iloc[0]
        conv = prof.get("mappo", {}).get("converged", "0/1")
        k, n = (int(x) for x in conv.split("/"))
        gate = bool(m.peak_reduction_pct_mean > h.peak_reduction_pct_mean and k >= (n + 1) // 2)
    order = ["none", "heuristic", "ppo", "ippo", "mappo"]
    df["o"] = df.policy.map({p: i for i, p in enumerate(order)})
    rows = df.sort_values("o").drop(columns="o").to_dict(orient="records")
    return {
        "benchmark": [
            {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in r.items()}
            for r in rows
        ],
        "compute": {k: v for k, v in prof.items() if isinstance(v, dict)},
        "gate": gate,
    }


def write_page(data: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = TEMPLATE.read_text(encoding="utf-8")
    html = (
        html.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
        .replace("__DATE__", data["meta"]["date"])
        .replace("__CKPT__", data["meta"]["ckpt"])
        .replace(
            "__BACKEND__",
            "LightSim2Grid" if data["meta"]["pf_backend"] == "lightsim" else "pandapower",
        )
    )
    head, body = html.split("</style>", 1)
    html = (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        + head
        + "</style>\n</head>\n<body>"
        + body
        + "\n</body>\n</html>\n"
    )
    out = OUT_DIR / "govgridmarl_status.html"
    out.write_text(html, encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--validate", action="store_true", help="run scripts/validate_env.py first")
    ap.add_argument(
        "--train", action="store_true", help="train a quick MAPPO if no checkpoint exists"
    )
    ap.add_argument("--day", type=int, default=10_000, help="evaluation-day seed (default 10000)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if not check_stack():
        return 1
    if args.validate:
        banner("2/4  Environment validation (m2-t14)")
        rc = subprocess.run(
            [sys.executable, str(REPO / "scripts" / "validate_env.py")], cwd=REPO
        ).returncode
        if rc:
            print("  validation FAILED — see results/logs/validate_env_*.log")
            return rc
    ckpt = newest_checkpoint()
    if ckpt is None and args.train:
        quick_train()
        ckpt = newest_checkpoint()
    banner(
        f"3/4  Replaying evaluation day {args.day} on IEEE-14  ({'checkpoint: ' + str(ckpt.relative_to(REPO)) if ckpt else 'no checkpoint yet'})"
    )
    data = replay(args.day, ckpt)
    banner("4/4  Writing the status page")
    out = write_page(data)
    print(f"  {out}")
    if not args.no_browser:
        webbrowser.open(out.as_uri())
    print(
        "\nDone. Also available: results/figures/*.png, results/tables/benchmark.md, "
        "docs/reports/phase1_milestone_report.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
