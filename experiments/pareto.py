"""Efficiency-fairness Pareto analysis (tasks m4-t5, m4-t7, m4-t9, m4-t11 .. m4-t13).

Reads the Month-4 runs (``scripts/run_phase2_month4.sh`` -> ``results/runs_m4``):

* ``mappo/seed<k>_wf<lambda>``       weighted-sum sweep (m4-t5 / m4-t6)
* ``mappo/seed<k>_gapfrac2.0``, ``mappo/seed<k>_jainfrac2.0``   fairness-form ablation (m4-t3)
* ``moppo/seed<k>``                  preference-conditioned MOPPO (m4-t8)

and evaluates every checkpoint on the 20 held-out protocol days (deterministic
policies). MOPPO is evaluated once per lambda of ``LAMBDA_GRID`` with one
trained model. Outputs (``results/tables`` / ``results/figures``):

* ``pareto_sweep.csv``      per (method, lambda, seed): all KPIs on the 20 days
* ``pareto_summary.csv``    per (method, lambda): mean +- 95 % CI
* ``pareto_comparison.csv`` front size, hypervolume and C-metric, weighted-sum vs MOPPO (m4-t9)
* ``fairness_significance.csv``  Welch tests vs the lambda = 0 baseline (m4-t12)
* ``fairness_operating_point.json``  the smallest lambda meeting both targets (m4-t11)
* ``pareto_front.png``, ``fairness_tradeoff.png``, ``burden_by_group_fair.png`` (m4-t13)

Ends with ``RESULT: PASS/FAIL`` on the Month-4 target (Jain >= 0.85 with
peak reduction >= 15 % for at least one configuration, mean over seeds).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "envs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")

from govgridmarl.agents.scalarization import LAMBDA_GRID  # noqa: E402
from govgridmarl.evaluation import (  # noqa: E402
    PHASE2_TARGETS,
    coverage,
    hypervolume_2d,
    in_acceptable_region,
    pareto_front,
)

COLORS = {
    "weighted_sum": "#2a78d6",
    "moppo": "#eb6834",
    "gap_fraction": "#1baf7a",
    "jain_fraction": "#8a5cd6",
    "heuristic": "#eda100",
}
GROUP_COLORS = {"low": "#2a78d6", "mid": "#eb6834", "high": "#1baf7a"}
KPIS = [
    "return",
    "peak_reduction_pct",
    "cost_total",
    "jain_index",
    "jain_frac",
    "gini_kwh",
    "violation_rate",
    "group_gap_ratio",
    "vulnerable_burden_ratio",
    "energy_shed_mwh",
]


def make_env(seed: int = 42, **kw):
    from govgrid_envs import GridMARLEnv

    return GridMARLEnv(n_households=14, seed=seed, **kw)


def rollout(env, agent, seed: int, lam: float | None = None):
    agents = env.possible_agents
    obs, _ = env.reset(seed=seed, options={"day_scale": 1.0})
    pref = agent.pref_vector(np.array([lam])) if lam is not None else None
    team = 0.0
    while env.agents:
        o = np.stack([obs[a] for a in agents])
        a, _, _ = agent.act(o, env.state(), deterministic=True, pref=pref)
        obs, _, _, _, info = env.step({ag: int(a[i]) for i, ag in enumerate(agents)})
        team += info[agents[0]]["team_reward"]
    ep = info[agents[0]]["episode"]
    return {
        "return": team,
        **{k: ep[k] for k in KPIS if k in ep},
        "energy_shed_mwh": ep["energy_base_mwh"] - ep["energy_res_mwh"],
        "burden": env._burden.copy(),
        "groups": list(env.groups),
    }


def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return 0.0
    from scipy import stats

    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="results/runs_m4")
    ap.add_argument("--n-days", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--gate-basis",
        default="fraction",
        choices=["kwh", "fraction"],
        help="Jain basis of the acceptance region (amended to 'fraction' on 2026-10-20)",
    )
    args = ap.parse_args()
    gate = "jain_index" if args.gate_basis == "kwh" else "jain_frac"
    run_root = REPO_ROOT / args.run_root
    out = REPO_ROOT / args.out
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

    from govgridmarl.agents.heuristic import RuleBasedDR
    from govgridmarl.agents.ppo_core import PPOFamilyAgent

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#e6e5e1",
            "grid.linewidth": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "legend.frameon": False,
        }
    )

    env = make_env()
    days = [10_000 + k for k in range(args.n_days)]

    # ---- collect checkpoints ---------------------------------------------------
    jobs = []  # (method, lambda, seed, agent, env_kwargs)
    for ck in sorted(run_root.glob("mappo/seed*/checkpoints/final.pt")):
        run = ck.parents[1].name  # seed0_wf2.0
        m = re.match(r"seed(\d+)_(wf|gapfrac|jainfrac)([\d.]+)$", run)
        if not m:
            continue
        seed, kind, lam = int(m.group(1)), m.group(2), float(m.group(3))
        method = {"wf": "weighted_sum", "gapfrac": "gap_fraction", "jainfrac": "jain_fraction"}[
            kind
        ]
        jobs.append((method, lam, seed, PPOFamilyAgent.from_checkpoint(ck), None))
    for ck in sorted(run_root.glob("moppo/seed*/checkpoints/final.pt")):
        seed = int(ck.parents[1].name.replace("seed", ""))
        agent = PPOFamilyAgent.from_checkpoint(ck)
        for lam in LAMBDA_GRID:
            jobs.append(("moppo", float(lam), seed, agent, lam))
    print(f"{len(jobs)} (method, lambda, seed) evaluations x {len(days)} days")

    rows, burdens = [], {}
    for method, lam, seed, agent, pref_lam in jobs:
        res = [rollout(env, agent, d, pref_lam) for d in days]
        row = {"method": method, "lambda": lam, "seed": seed}
        row.update({k: float(np.mean([r[k] for r in res])) for k in KPIS})
        rows.append(row)
        burdens.setdefault((method, lam), []).append(
            (np.mean([r["burden"] for r in res], axis=0), res[0]["groups"])
        )
        print(
            f"  {method:13s} lam {lam:4.1f} seed {seed} | peak {row['peak_reduction_pct']:5.1f}% "
            f"| J {row['jain_index']:.3f} (frac {row['jain_frac']:.3f}) | viol {row['violation_rate']:.2f} "
            f"| vuln {row['vulnerable_burden_ratio']:.2f}"
        )
    h = RuleBasedDR()
    hres = [rollout(env, h, d) for d in days]
    hrow = {"method": "heuristic", "lambda": np.nan, "seed": 0}
    hrow.update({k: float(np.mean([r[k] for r in hres])) for k in KPIS})
    rows.append(hrow)
    sweep = pd.DataFrame(rows)
    sweep.to_csv(tab_dir / "pareto_sweep.csv", index=False)

    # ---- summary (mean +- CI over seeds) --------------------------------------
    summ = []
    for (method, lam), d in sweep.groupby(["method", "lambda"], dropna=False):
        r = {"method": method, "lambda": lam, "n_seeds": len(d)}
        for k in KPIS:
            r[f"{k}_mean"], r[f"{k}_ci95"] = float(d[k].mean()), ci95(d[k])
        r["in_region"] = in_acceptable_region(r["peak_reduction_pct_mean"], r[f"{gate}_mean"])
        r["in_region_kwh"] = in_acceptable_region(
            r["peak_reduction_pct_mean"], r["jain_index_mean"]
        )
        summ.append(r)
    summary = pd.DataFrame(summ).sort_values(["method", "lambda"])
    summary.to_csv(tab_dir / "pareto_summary.csv", index=False)
    cols = [
        "method",
        "lambda",
        "peak_reduction_pct_mean",
        "jain_index_mean",
        "jain_frac_mean",
        "violation_rate_mean",
        "group_gap_ratio_mean",
        "vulnerable_burden_ratio_mean",
        "in_region",
        "in_region_kwh",
    ]
    print(summary[cols].round(3).to_string(index=False))

    # ---- Pareto comparison: weighted-sum vs MOPPO (m4-t9) ---------------------
    ref = (0.0, 0.5)
    comp = []
    fronts = {}
    for method in ("weighted_sum", "moppo", "gap_fraction", "jain_fraction"):
        d = summary[summary.method == method]
        if d.empty:
            continue
        pts = d[["peak_reduction_pct_mean", f"{gate}_mean"]].values
        fr = pareto_front(pts)
        fronts[method] = fr
        comp.append(
            {
                "method": method,
                "n_points": len(pts),
                "n_nondominated": len(fr),
                "hypervolume": hypervolume_2d(pts, ref),
                "trained_models": int(sweep[sweep.method == method].seed.nunique())
                * (1 if method == "moppo" else int(d["lambda"].nunique())),
                "best_jain": float(d.jain_index_mean.max()),
                "best_peak": float(d.peak_reduction_pct_mean.max()),
                "n_in_region": int(d.in_region.sum()),
            }
        )
    if "weighted_sum" in fronts and "moppo" in fronts:
        c_ws_mo = coverage(fronts["weighted_sum"], fronts["moppo"])
        c_mo_ws = coverage(fronts["moppo"], fronts["weighted_sum"])
        for r in comp:
            if r["method"] == "weighted_sum":
                r["covers_other"] = c_ws_mo
            if r["method"] == "moppo":
                r["covers_other"] = c_mo_ws
    comp = pd.DataFrame(comp)
    comp.to_csv(tab_dir / "pareto_comparison.csv", index=False)
    print(comp.round(3).to_string(index=False))

    # ---- significance vs lambda = 0 (m4-t12) ----------------------------------
    from scipy import stats

    sig = []
    base = sweep[(sweep.method == "weighted_sum") & (sweep["lambda"] == 0.0)]
    for (method, lam), d in sweep.groupby(["method", "lambda"], dropna=False):
        if method == "heuristic" or (method == "weighted_sum" and lam == 0.0):
            continue
        for k in ("peak_reduction_pct", "jain_index", "jain_frac", "violation_rate", "return"):
            a, b = d[k].values, base[k].values
            if a.size > 1 and b.size > 1:
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                delta = float(np.mean([np.sign(x - y) for x in a for y in b]))
            else:
                p, delta = np.nan, np.nan
            sig.append(
                {
                    "method": method,
                    "lambda": lam,
                    "metric": k,
                    "mean": a.mean(),
                    "mean_lambda0": b.mean(),
                    "welch_p": p,
                    "cliffs_delta": delta,
                }
            )
    pd.DataFrame(sig).to_csv(tab_dir / "fairness_significance.csv", index=False)

    # ---- operating point (m4-t11) -----------------------------------------------
    cand = summary[summary.in_region].sort_values(
        ["lambda", "peak_reduction_pct_mean"], ascending=[True, False]
    )
    op = None
    if not cand.empty:
        r = cand.iloc[0]
        op = {
            "method": r.method,
            "lambda": float(r["lambda"]),
            "peak_reduction_pct": float(r.peak_reduction_pct_mean),
            "jain_index": float(r.jain_index_mean),
            "jain_frac": float(r.jain_frac_mean),
            "violation_rate": float(r.violation_rate_mean),
            "targets": PHASE2_TARGETS,
            "gate_basis": args.gate_basis,
        }
    (tab_dir / "fairness_operating_point.json").write_text(json.dumps(op, indent=2))

    # ---- figures (m4-t13) -------------------------------------------------------
    fig, axes2 = plt.subplots(1, 2, figsize=(10.5, 4.0))
    labels = {
        "weighted_sum": "weighted-sum MAPPO (6 models)",
        "moppo": "MOPPO (1 model, 6 preferences)",
        "gap_fraction": "gap / fraction form",
        "jain_fraction": "Jain / fraction form",
    }
    feas_path = tab_dir / "fairness_feasibility.csv"
    feas = pd.read_csv(feas_path) if feas_path.exists() else None
    for ax, basis, lab in zip(
        axes2,
        ("jain_index", "jain_frac"),
        ("Jain index (kWh burden)", "Jain index (relative burden B/F)"),
    ):
        for method in ("weighted_sum", "moppo", "gap_fraction", "jain_fraction"):
            d = summary[summary.method == method].sort_values("lambda")
            if d.empty:
                continue
            ax.errorbar(
                d.peak_reduction_pct_mean,
                d[f"{basis}_mean"],
                xerr=d.peak_reduction_pct_ci95,
                yerr=d[f"{basis}_ci95"],
                fmt="o-" if method in ("weighted_sum", "moppo") else "s",
                color=COLORS[method],
                lw=1.5,
                ms=5,
                capsize=2,
                label=labels[method],
            )
            if method == "weighted_sum":
                for _, r in d.iterrows():
                    ax.annotate(
                        f"λ={r['lambda']:g}",
                        (r.peak_reduction_pct_mean, r[f"{basis}_mean"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=7,
                        color=COLORS[method],
                    )
        ax.plot(
            hrow["peak_reduction_pct"],
            hrow[basis],
            "D",
            color=COLORS["heuristic"],
            label="rule-based DR",
        )
        if basis == "jain_index" and feas is not None:
            ax.plot(
                feas.peak_reduction_pct,
                feas.jain_kwh,
                "x",
                color="#52514e",
                ms=4,
                label="scripted equal-fraction/equal-kWh blends",
            )
        ax.axvline(15, color="#52514e", lw=0.8, ls=":")
        ax.axhline(0.85, color="#52514e", lw=0.8, ls=":")
        ax.fill_between([15, 25], 0.85, 1.02, color="#1baf7a", alpha=0.07, lw=0)
        ax.set_xlim(0, 20)
        ax.set_xlabel("peak reduction (%)")
        ax.set_ylabel(lab)
        ax.set_title(("gate — " if basis == gate else "reported — ") + lab, loc="left", fontsize=9)
    axes2[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(
        "Efficiency-fairness front on IEEE-14 (20 held-out days, mean ± 95 % CI over 3 seeds)",
        x=0.02,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "pareto_front.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3.0))
    ws = summary[summary.method == "weighted_sum"].sort_values("lambda")
    mo = summary[summary.method == "moppo"].sort_values("lambda")
    for ax, k, lab in zip(
        axes,
        ["peak_reduction_pct", "jain_frac", "jain_index", "vulnerable_burden_ratio"],
        [
            "peak reduction (%)",
            "Jain (relative burden, gate)",
            "Jain (kWh burden)",
            "vulnerable / other burden",
        ],
    ):
        for d, method in ((ws, "weighted_sum"), (mo, "moppo")):
            if d.empty:
                continue
            ax.errorbar(
                d["lambda"],
                d[f"{k}_mean"],
                yerr=d[f"{k}_ci95"],
                fmt="o-",
                color=COLORS[method],
                lw=1.5,
                ms=4,
                capsize=2,
                label=method.replace("_", "-"),
            )
        ax.set_xlabel("fairness weight λ")
        ax.set_title(lab, loc="left")
    axes[0].axhline(15, color="#52514e", lw=0.8, ls=":")
    axes[1].axhline(0.85, color="#52514e", lw=0.8, ls=":")
    axes[2].axhline(0.85, color="#52514e", lw=0.8, ls=":")
    axes[0].legend(fontsize=7.5)
    fig.suptitle(
        "Trade-off along the fairness weight (mean ± 95 % CI over 3 seeds)", x=0.02, ha="left"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "fairness_tradeoff.png")
    plt.close(fig)

    show = [("weighted_sum", 0.0), ("weighted_sum", 2.0), ("weighted_sum", 8.0)]
    show = [s for s in show if s in burdens]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, key in zip(axes, show):
            b, groups = burdens[key][0]
            order = np.argsort([("low", "mid", "high").index(g) for g in groups])
            ax.bar(
                range(len(b)), b[order], color=[GROUP_COLORS[groups[i]] for i in order], width=0.7
            )
            j = summary[
                (summary.method == key[0]) & (summary["lambda"] == key[1])
            ].jain_index_mean.iloc[0]
            ax.set_title(f"λ = {key[1]:g} — Jain {j:.2f}", loc="left")
            ax.set_xticks([])
            ax.set_xlabel("household (sorted by income group)")
            ax.grid(axis="x", visible=False)
        axes[0].set_ylabel("curtailment burden (kWh/home/day)")
        from matplotlib.patches import Patch

        axes[-1].legend(
            handles=[Patch(color=c, label=g) for g, c in GROUP_COLORS.items()],
            title="income group",
            fontsize=8,
            title_fontsize=8,
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "burden_by_group_fair.png")
        plt.close(fig)

    ok = op is not None
    print("operating point:", json.dumps(op))
    print(
        f"RESULT: {'PASS' if ok else 'FAIL'} — Month-4 target (Jain >= 0.85 on the "
        f"{args.gate_basis} basis with peak reduction >= 15 %)"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
