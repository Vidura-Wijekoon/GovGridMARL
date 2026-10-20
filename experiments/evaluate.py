"""Evaluation, benchmarking and figure generation (tasks m3-t7, m3-t8, m3-t10).

Implements ``docs/evaluation_protocol.md``:

* loads every ``<run_root>/<agent>/seed*/checkpoints/final.pt``;
* evaluates each seed on the 20 held-out evaluation days (deterministic policy)
  and on the stress set; adds the ``none`` and ``heuristic`` baselines;
* writes ``results/tables/benchmark.csv`` / ``.md`` (mean +- 95 % CI over
  training seeds), ``results/tables/per_seed.csv``, ``significance.csv``;
* draws ``results/figures/`` — training curves, benchmark bars, daily profile,
  burden distribution by income group;
* prints a compact summary and ends with ``RESULT: PASS/FAIL`` on the Phase-1
  gate ("baseline MAPPO converges on IEEE-14").

Usage::

    python experiments/evaluate.py                     # defaults
    python experiments/evaluate.py --run-root results/runs --n-days 20
"""

from __future__ import annotations

import argparse
import json
import platform
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

# Fixed categorical assignment (validated palette; never cycled)
COLORS = {
    "mappo": "#2a78d6",
    "ippo": "#eb6834",
    "ppo": "#1baf7a",
    "heuristic": "#eda100",
    "none": "#8a8985",
}
GROUP_COLORS = {"low": "#2a78d6", "mid": "#eb6834", "high": "#1baf7a"}
ORDER = ["none", "heuristic", "ppo", "ippo", "mappo"]
METRICS = [
    "return",
    "peak_reduction_pct",
    "cost_total",
    "jain_index",
    "violation_rate",
    "energy_shed_mwh",
]


# --------------------------------------------------------------------------- #
def make_env(seed: int = 42, **kw):
    from govgrid_envs import GridMARLEnv

    return GridMARLEnv(n_households=14, seed=seed, **kw)


def rollout(env, agent, seed: int, day_scale: float = 1.0):
    agents = env.possible_agents
    obs, _ = env.reset(seed=seed, options={"day_scale": day_scale})
    team, profile, base = 0.0, [], []
    while env.agents:
        o = np.stack([obs[a] for a in agents])
        a, _, _ = agent.act(o, env.state(), deterministic=True)
        obs, _, _, _, info = env.step({ag: int(a[i]) for i, ag in enumerate(agents)})
        i0 = info[agents[0]]
        team += i0["team_reward"]
        profile.append(i0["p_res_mw"])
        base.append(i0["p_base_mw"])
    ep = info[agents[0]]["episode"]
    return {
        "return": team,
        "peak_reduction_pct": ep["peak_reduction_pct"],
        "cost_total": ep["cost_total"],
        "jain_index": ep["jain_index"],
        "violation_rate": ep["violation_rate"],
        "energy_shed_mwh": ep["energy_base_mwh"] - ep["energy_res_mwh"],
        "burden": env._burden.copy(),
        "groups": list(env.groups),
        "profile": np.array(profile),
        "base": np.array(base),
        "burden_by_group": ep["burden_by_group_kwh"],
    }


def load_policies(run_root: Path):
    from govgridmarl.agents.heuristic import NoControl, RuleBasedDR
    from govgridmarl.agents.ppo_core import PPOFamilyAgent

    pols = {"none": {"fixed": NoControl()}, "heuristic": {"fixed": RuleBasedDR()}}
    for ck in sorted(run_root.glob("*/seed*/checkpoints/final.pt")):
        name, seed = ck.parents[2].name, ck.parents[1].name
        pols.setdefault(name, {})[seed] = PPOFamilyAgent.from_checkpoint(ck)
    return pols


def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return 0.0
    from scipy import stats

    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def welch(a, b):
    from scipy import stats

    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or b.size < 2:
        return np.nan, np.nan
    t, p = stats.ttest_ind(a, b, equal_var=False)
    # Cliff's delta
    d = np.mean([np.sign(x - y) for x in a for y in b])
    return float(p), float(d)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="results/runs")
    ap.add_argument("--n-days", type=int, default=20)
    ap.add_argument("--out", default="results")
    ap.add_argument(
        "--energy-share", action="store_true", help="also sweep renewable share 0..0.8 (slower)"
    )
    ap.add_argument(
        "--es-days", type=int, default=5, help="evaluation days per energy-share scenario"
    )
    args = ap.parse_args()
    run_root = REPO_ROOT / args.run_root
    out = REPO_ROOT / args.out
    fig_dir, tab_dir = out / "figures", out / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib

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
            "axes.edgecolor": "#c3c2b7",
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.titleweight": "normal",
            "legend.frameon": False,
        }
    )

    pols = load_policies(run_root)
    print(f"policies: { {k: len(v) for k, v in pols.items()} }")
    env = make_env()
    stress = make_env(background_factor=1.25)
    eval_seeds = [10_000 + k for k in range(args.n_days)]

    # ---- per-seed evaluation -------------------------------------------------
    rows, stress_rows, profiles, burdens = [], [], {}, {}
    for name in ORDER:
        if name not in pols:
            continue
        for seed, agent in pols[name].items():
            res = [rollout(env, agent, s) for s in eval_seeds]
            row = {"policy": name, "train_seed": seed}
            row.update({m: float(np.mean([r[m] for r in res])) for m in METRICS})
            rows.append(row)
            sres = [rollout(stress, agent, s, day_scale=1.3) for s in eval_seeds]
            srow = {"policy": name, "train_seed": seed}
            srow.update({m: float(np.mean([r[m] for r in sres])) for m in METRICS})
            stress_rows.append(srow)
            if name not in profiles:  # first seed: keep day 0 profile + burdens
                profiles[name] = res[0]
                burdens[name] = (np.mean([r["burden"] for r in res], axis=0), res[0]["groups"])
            print(
                f"  {name:9s} {seed:7s} | R {row['return']:.3f} | peak {row['peak_reduction_pct']:.1f}% "
                f"| J {row['jain_index']:.3f} | viol {row['violation_rate']:.2f} | cost {row['cost_total']:.0f}"
            )
    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(tab_dir / "per_seed.csv", index=False)
    pd.DataFrame(stress_rows).to_csv(tab_dir / "per_seed_stress.csv", index=False)

    # ---- benchmark table -----------------------------------------------------
    bench = []
    for name in ORDER:
        d = per_seed[per_seed.policy == name]
        if d.empty:
            continue
        r = {"policy": name, "n_seeds": len(d)}
        for m in METRICS:
            r[f"{m}_mean"], r[f"{m}_ci95"] = float(d[m].mean()), ci95(d[m])
        bench.append(r)
    bench = pd.DataFrame(bench)
    bench.to_csv(tab_dir / "benchmark.csv", index=False)
    md = [
        "| policy | seeds | return | peak reduction % | cost | Jain | violation rate | energy shed MWh |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in bench.iterrows():
        f = lambda m, p=2: (
            f"{r[f'{m}_mean']:.{p}f} ± {r[f'{m}_ci95']:.{p}f}"
            if r["n_seeds"] > 1
            else f"{r[f'{m}_mean']:.{p}f}"
        )  # noqa: E731
        md.append(
            f"| {r.policy} | {int(r.n_seeds)} | {f('return',3)} | {f('peak_reduction_pct',1)} | "
            f"{f('cost_total',0)} | {f('jain_index',3)} | {f('violation_rate',2)} | {f('energy_shed_mwh',1)} |"
        )
    (tab_dir / "benchmark.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- significance --------------------------------------------------------
    sig = []
    learners = [n for n in ("ppo", "ippo", "mappo") if n in pols]
    for a in learners:
        for b in learners + ["heuristic"]:
            if a == b:
                continue
            for m in ("return", "peak_reduction_pct", "jain_index", "violation_rate"):
                xa = per_seed[per_seed.policy == a][m].values
                xb = per_seed[per_seed.policy == b][m].values
                if b == "heuristic":  # single deterministic value -> one-sample test
                    from scipy import stats

                    p = float(stats.ttest_1samp(xa, xb[0]).pvalue) if xa.size > 1 else np.nan
                    d = float(np.mean(np.sign(xa - xb[0])))
                else:
                    p, d = welch(xa, xb)
                sig.append(
                    {
                        "a": a,
                        "b": b,
                        "metric": m,
                        "mean_a": xa.mean(),
                        "mean_b": xb.mean(),
                        "welch_p": p,
                        "cliffs_delta": d,
                    }
                )
    if sig:
        pd.DataFrame(sig).to_csv(tab_dir / "significance.csv", index=False)

    # ---- training curves -----------------------------------------------------
    curves = {}
    for name in learners:
        dfs = []
        for f in sorted(run_root.glob(f"{name}/seed*/metrics.csv")):
            d = pd.read_csv(f)
            d["seed"] = f.parent.name
            dfs.append(d)
        if dfs:
            curves[name] = pd.concat(dfs)
    if curves:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        for name, d in curves.items():
            g = d.groupby("timesteps")
            for ax, m, lab in (
                (axes[0], "return_mean", "learning return (per agent)"),
                (axes[1], "peak_reduction_pct", "peak reduction (%)"),
            ):
                mu, sd = g[m].mean(), g[m].std().fillna(0)
                k = 5
                mu_s = mu.rolling(k, min_periods=1, center=True).mean()
                ax.plot(mu.index, mu_s, color=COLORS[name], lw=2, label=name.upper())
                ax.fill_between(
                    mu.index,
                    (mu - sd).rolling(k, min_periods=1, center=True).mean(),
                    (mu + sd).rolling(k, min_periods=1, center=True).mean(),
                    color=COLORS[name],
                    alpha=0.15,
                    lw=0,
                )
                ax.set_ylabel(lab)
                ax.set_xlabel("environment steps")
        h = per_seed[per_seed.policy == "heuristic"]
        if not h.empty:
            axes[1].axhline(
                h.peak_reduction_pct.iloc[0], color=COLORS["heuristic"], lw=1.2, ls="--"
            )
            axes[1].text(
                axes[1].get_xlim()[1],
                h.peak_reduction_pct.iloc[0],
                " rule-based DR",
                va="center",
                ha="left",
                fontsize=8,
                color="#52514e",
            )
        axes[0].legend(loc="lower right")
        fig.suptitle(
            "Phase-1 baselines on IEEE-14 — mean ± std over training seeds", x=0.02, ha="left"
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "training_curves.png")
        plt.close(fig)

    # ---- benchmark bars ------------------------------------------------------
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.9))
    for ax, m, lab in zip(
        axes,
        ["peak_reduction_pct", "jain_index", "violation_rate", "cost_total"],
        ["peak reduction (%)", "Jain index", "violation rate", "daily cost"],
        strict=False,
    ):
        names = [n for n in ORDER if n in set(bench.policy)]
        vals = [bench.set_index("policy").loc[n, f"{m}_mean"] for n in names]
        errs = [bench.set_index("policy").loc[n, f"{m}_ci95"] for n in names]
        ax.bar(
            names,
            vals,
            yerr=errs,
            color=[COLORS[n] for n in names],
            width=0.6,
            capsize=2,
            error_kw={"lw": 0.8, "color": "#52514e"},
        )
        for i, v in enumerate(vals):
            ax.text(
                i,
                v,
                f"{v:.2f}" if m != "cost_total" else f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#0b0b0b",
            )
        ax.set_title(lab, loc="left")
        ax.tick_params(axis="x", labelrotation=30)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(15, color="#52514e", lw=0.8, ls=":")
    axes[0].text(-0.4, 15.3, "target ≥ 15 %", fontsize=7.5, color="#52514e")
    axes[1].axhline(0.85, color="#52514e", lw=0.8, ls=":")
    axes[1].text(-0.4, 0.86, "target ≥ 0.85 (Phase 2)", fontsize=7.5, color="#52514e")
    fig.suptitle(
        "Benchmark on 20 held-out days (mean ± 95 % CI over training seeds)", x=0.02, ha="left"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "benchmark.png")
    plt.close(fig)

    # ---- daily profile -------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    hours = np.arange(24)
    base = profiles["none"]["base"]
    ax.plot(hours, base, color=COLORS["none"], lw=2, label="no control")
    for name in ORDER:
        if name in profiles and name != "none":
            ax.plot(
                hours,
                profiles[name]["profile"],
                color=COLORS[name],
                lw=1.8,
                label=name.upper() if name != "heuristic" else "rule-based DR",
            )
    ax.set_xlabel("hour of day")
    ax.set_ylabel("residential aggregate load (MW)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax.set_title("Evaluation day 10000 — load profile under each policy", loc="left")
    fig.tight_layout()
    fig.savefig(fig_dir / "daily_profile.png")
    plt.close(fig)

    # ---- burden distribution -------------------------------------------------
    show = [n for n in ("heuristic", "mappo", "ippo") if n in burdens]
    if show:
        fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.0), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, name in zip(axes, show, strict=False):
            b, groups = burdens[name]
            order = np.argsort([("low", "mid", "high").index(g) for g in groups])
            ax.bar(
                range(len(b)), b[order], color=[GROUP_COLORS[groups[i]] for i in order], width=0.7
            )
            ax.set_title(
                f"{name if name != 'heuristic' else 'rule-based DR'} — Jain "
                f"{per_seed[per_seed.policy == name].jain_index.mean():.2f}",
                loc="left",
            )
            ax.set_xlabel("household (sorted by income group)")
            ax.set_xticks([])
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
        fig.savefig(fig_dir / "burden_by_group.png")
        plt.close(fig)

    # ---- energy-share sweep (generation mix) --------------------------------
    if args.energy_share:
        scen = [(0.0, 0.5), (0.2, 0.5), (0.4, 0.5), (0.6, 0.5), (0.8, 0.5), (0.8, 0.8), (0.8, 0.2)]
        es_rows = []
        for rs, wf in scen:
            e = make_env(renewable_share=rs, wind_fraction=wf)
            for name in ORDER:
                if name not in pols:
                    continue
                for seed, agent in pols[name].items():
                    res = []
                    for s_ in eval_seeds[: args.es_days]:
                        agents = e.possible_agents
                        obs, _ = e.reset(seed=s_, options={"day_scale": 1.0, "wind_scale": 1.0})
                        lmax = 0.0
                        while e.agents:
                            o = np.stack([obs[a] for a in agents])
                            a, _, _ = agent.act(o, e.state(), deterministic=True)
                            obs, _, _, _, info = e.step(
                                {ag: int(a[i]) for i, ag in enumerate(agents)}
                            )
                            lmax = max(lmax, info[agents[0]]["loading_max"])
                        ep = info[agents[0]]["episode"]
                        res.append({**ep, "loading_max": lmax})
                    es_rows.append(
                        {
                            "renewable_share": rs,
                            "wind_fraction": wf,
                            "policy": name,
                            "train_seed": seed,
                            **{
                                k: float(np.mean([r[k] for r in res]))
                                for k in (
                                    "peak_reduction_pct",
                                    "jain_index",
                                    "violation_rate",
                                    "renewable_share_of_consumption",
                                    "renewable_surplus_mwh",
                                    "net_import_peak_mw",
                                    "system_losses_mwh",
                                    "loading_max",
                                )
                            },
                        }
                    )
        es = pd.DataFrame(es_rows)
        es.to_csv(tab_dir / "energy_share_sweep.csv", index=False)
        agg = (
            es.groupby(["renewable_share", "wind_fraction", "policy"])
            .mean(numeric_only=True)
            .reset_index()
        )
        agg.to_csv(tab_dir / "energy_share_summary.csv", index=False)
        print(
            agg[
                [
                    "renewable_share",
                    "wind_fraction",
                    "policy",
                    "renewable_share_of_consumption",
                    "renewable_surplus_mwh",
                    "net_import_peak_mw",
                    "loading_max",
                    "peak_reduction_pct",
                    "jain_index",
                ]
            ]
            .round(3)
            .to_string()
        )
        fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
        base5 = agg[agg.wind_fraction == 0.5]
        for name in ORDER:
            d = base5[base5.policy == name]
            if d.empty:
                continue
            axes[0].plot(
                d.renewable_share * 100,
                d.renewable_share_of_consumption * 100,
                "-o",
                color=COLORS[name],
                lw=2,
                ms=4,
                label=name,
            )
            axes[1].plot(
                d.renewable_share * 100,
                d.renewable_surplus_mwh,
                "-o",
                color=COLORS[name],
                lw=2,
                ms=4,
            )
            axes[2].plot(
                d.renewable_share * 100, d.net_import_peak_mw, "-o", color=COLORS[name], lw=2, ms=4
            )
        axes[0].plot([0, 80], [0, 80], ls=":", color="#52514e", lw=0.8)
        axes[0].set_ylabel("renewables consumed locally (%)")
        axes[1].set_ylabel("surplus exported (MWh/day)")
        axes[2].set_ylabel("net-import peak (MW)")
        for ax in axes:
            ax.set_xlabel("renewable share of daily energy (%) · wind/PV 50/50")
        axes[0].legend(fontsize=8)
        fig.suptitle(
            "Energy-share sweep — Phase-1 policies do not yet adapt to the generation mix",
            x=0.02,
            ha="left",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "energy_share_sweep.png")
        plt.close(fig)

    # ---- compute profile + gate ---------------------------------------------
    from govgridmarl.training import convergence_test

    summaries = {}
    for f in sorted(run_root.glob("*/seed*/summary.json")):
        s = json.loads(f.read_text())
        hist = pd.read_csv(f.parent / "metrics.csv")
        s["converged"], diag = convergence_test(hist["return_mean"].values)  # protocol criterion
        s.update({f"conv_{k}": v for k, v in diag.items()})
        summaries.setdefault(s["mode"], []).append(s)
    prof = {
        "hardware": platform.processor() or platform.machine(),
        "python": sys.version.split()[0],
    }
    for mode, ss in summaries.items():
        prof[mode] = {
            "seeds": len(ss),
            "timesteps": ss[0]["timesteps"],
            "wall_s_mean": float(np.mean([s["wall_s"] for s in ss])),
            "env_steps_per_s": float(np.mean([s["mean_env_steps_per_s"] for s in ss])),
            "converged": f"{sum(s['converged'] for s in ss)}/{len(ss)}",
        }
    (tab_dir / "compute_profile.json").write_text(json.dumps(prof, indent=2))
    print(json.dumps(prof, indent=2))

    gate = False
    if "mappo" in summaries:
        ss = summaries["mappo"]
        m = per_seed[per_seed.policy == "mappo"]
        h = per_seed[per_seed.policy == "heuristic"].peak_reduction_pct.iloc[0]
        improved = all(s["final_return"] > max(s["first_return"], 0) for s in ss)
        gate = (
            improved
            and (m.peak_reduction_pct.mean() > h)
            and sum(s["converged"] for s in ss) >= len(ss) // 2 + (len(ss) % 2)
        )
        print(
            f"gate: mappo improved={improved}, peak {m.peak_reduction_pct.mean():.1f}% vs heuristic {h:.1f}%, "
            f"converged {prof['mappo']['converged']}"
        )
    print(
        f"RESULT: {'PASS' if gate else 'FAIL'} — Phase-1 gate (baseline MAPPO converges on IEEE-14)"
    )
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
