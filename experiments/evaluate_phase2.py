"""Phase-2 safety evaluation (tasks m5-t5, m5-t8, m5-t9, m5-t11, m5-t12, m5-t13).

Reads the Month-5 runs (``scripts/run_phase2_month5.sh`` -> ``results/runs_m5``):

====================================  ==================  ====================================
run directory                         label               role
====================================  ==================  ====================================
mappo/seed<k>_unc                     unconstrained       same reward, no constraints (ablation)
mappo_lagrangian/seed<k>_lag          lagrangian          dual ascent, safety set (m5-t3/t4)
mappo_lagrangian/seed<k>_pid          pid_lagrangian      PID multipliers, safety set (m5-t6)
mappo_lagrangian/seed<k>_joint        joint               PID, safety + fairness constraints (m5-t10)
ppo_lagrangian/seed<k>_lag            ppo_lagrangian      centralised PPO-Lagrangian reference
mappo_lagrangian/seed0_lag_lr<eta>    lagrangian_lr<eta>  multiplier learning-rate sweep (m5-t5)
====================================  ==================  ====================================

Every policy is evaluated deterministically on (a) the 20 nominal protocol
days, (b) the stress set (demand x 1.3, background x 1.25) and (c) an N-1
contingency set: the three single-line outages that load the intact ratings
most under no control (screened here, reported in ``n1_screening.csv``).

Outputs in ``results/tables`` / ``results/figures``:

* ``safety_ablation.csv`` / ``.md``   nominal KPIs, mean +- 95 % CI (m5-t9)
* ``phase2_targets.csv``              Jain >= 0.85, peak >= 15 %, zero violations per method (m5-t11)
* ``stress_tests.csv``                per scenario x method: peak, violations, max loading (m5-t12)
* ``multiplier_lr_sweep.csv``         final multipliers, oscillation, violations per eta (m5-t5)
* ``safety_significance.csv``         Welch tests vs the unconstrained learner
* ``violation_curves.png``, ``multiplier_dynamics.png``, ``stress_violations.png``,
  ``safety_benchmark.png`` (m5-t13)

Ends with ``RESULT: PASS/FAIL`` on the joint Phase-2 targets (m5-t11).
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

from govgridmarl.evaluation import check_targets  # noqa: E402

LABELS = {
    ("mappo", "unc"): "unconstrained",
    # magnitude-cost runs (first batch, untuned multipliers - m5-t5 evidence)
    ("mappo_lagrangian", "lag"): "lagrangian_mag",
    ("mappo_lagrangian", "pid"): "pid_lagrangian_mag",
    ("mappo_lagrangian", "joint"): "joint_mag",
    ("ppo_lagrangian", "lag"): "ppo_lagrangian_mag",
    # indicator-cost runs (second batch, tuned - the Phase-2 learners)
    ("mappo_lagrangian", "lag_ind"): "lagrangian",
    ("mappo_lagrangian", "pid_ind"): "pid_lagrangian",
    ("mappo_lagrangian", "joint_ind"): "joint",
    ("ppo_lagrangian", "lag_ind"): "ppo_lagrangian",
}
ORDER = [
    "none",
    "heuristic",
    "unconstrained",
    "lagrangian_mag",
    "pid_lagrangian_mag",
    "joint_mag",
    "ppo_lagrangian",
    "lagrangian",
    "pid_lagrangian",
    "joint",
]
COLORS = {
    "lagrangian_mag": "#f3a98a",
    "pid_lagrangian_mag": "#8fd9bd",
    "joint_mag": "#e9a0b9",
    "ppo_lagrangian_mag": "#c4aee8",
    "none": "#8a8985",
    "heuristic": "#eda100",
    "unconstrained": "#2a78d6",
    "ppo_lagrangian": "#8a5cd6",
    "lagrangian": "#eb6834",
    "pid_lagrangian": "#1baf7a",
    "joint": "#d63a6e",
}
KPIS = [
    "return",
    "peak_reduction_pct",
    "cost_total",
    "jain_index",
    "jain_frac",
    "violation_rate",
    "viol_voltage",
    "viol_loading",
    "viol_comfort",
    "loading_max",
    "vulnerable_burden_ratio",
    "energy_shed_mwh",
]


POPULATIONS = (0, 1, 2)  # training seeds = synthetic household populations
TRANSFER_POPULATION = 42  # population used by the Phase-1 evaluate.py (never trained on)
_ENV_CACHE: dict = {}


def make_env(seed: int = 42, **kw):
    """Environment on the household population generated with ``seed``.

    The population is the dataset seed: a policy trained with seed k has only ever seen
    population k, so the protocol's held-out *days* are evaluated on population k and the
    seed-42 population is reported separately as a transfer test (protocol section 7).
    """
    from govgrid_envs import GridMARLEnv

    key = (seed, tuple(sorted(kw.items())))
    if key not in _ENV_CACHE:
        _ENV_CACHE[key] = GridMARLEnv(n_households=14, seed=seed, **kw)
    return _ENV_CACHE[key]


def rollout(env, agent, seed: int, day_scale: float = 1.0, deterministic: bool = True):
    agents = env.possible_agents
    obs, _ = env.reset(seed=seed, options={"day_scale": day_scale})
    team, lmax = 0.0, 0.0
    while env.agents:
        o = np.stack([obs[a] for a in agents])
        a, _, _ = agent.act(o, env.state(), deterministic=deterministic)
        obs, _, _, _, info = env.step({ag: int(a[i]) for i, ag in enumerate(agents)})
        i0 = info[agents[0]]
        team += i0["team_reward"]
        lmax = max(lmax, i0["loading_max"])
    ep = info[agents[0]]["episode"]
    vb = ep["violations_by_constraint"]
    return {
        "return": team,
        "peak_reduction_pct": ep["peak_reduction_pct"],
        "cost_total": ep["cost_total"],
        "jain_index": ep["jain_index"],
        "jain_frac": ep["jain_frac"],
        "violation_rate": ep["violation_rate"],
        "viol_voltage": vb["voltage"] / env.horizon,
        "viol_loading": vb["loading"] / env.horizon,
        "viol_comfort": vb["comfort"] / env.horizon,
        "loading_max": lmax,
        "vulnerable_burden_ratio": ep["vulnerable_burden_ratio"],
        "energy_shed_mwh": ep["energy_base_mwh"] - ep["energy_res_mwh"],
    }


def ci95(x):
    x = np.asarray(x, float)
    if x.size < 2:
        return 0.0
    from scipy import stats

    return float(stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size))


def load_policies(run_root: Path):
    from govgridmarl.agents.heuristic import NoControl, RuleBasedDR
    from govgridmarl.agents.ppo_core import PPOFamilyAgent

    # fixed policies are evaluated on the three training populations (seed = population)
    pols: dict[str, dict[int, object]] = {
        "none": {k: NoControl() for k in POPULATIONS},
        "heuristic": {k: RuleBasedDR() for k in POPULATIONS},
    }
    runs: dict[str, dict[int, Path]] = {}
    for ck in sorted(run_root.glob("*/seed*/checkpoints/final.pt")):
        agent_name, run = ck.parents[2].name, ck.parents[1].name
        m = re.match(r"seed(\d+)_(.+)$", run)
        if not m:
            continue
        seed, tag = int(m.group(1)), m.group(2)
        if tag.startswith("lag_lr"):
            label = "lagrangian_lr" + tag[len("lag_lr") :]
        else:
            label = LABELS.get((agent_name, tag))
        if label is None:
            continue
        pols.setdefault(label, {})[seed] = PPOFamilyAgent.from_checkpoint(ck)
        runs.setdefault(label, {})[seed] = ck.parents[1]
    return pols, runs


def screen_contingencies(n_top: int = 3, days=(10_000, 10_001, 10_002)) -> pd.DataFrame:
    """Max loading under no control for every single-line outage (nominal days)."""

    env = make_env()
    rows = []
    for k in range(env._pf.n_lines):
        env.set_contingency(k)
        lm, conv = [], True
        for d in days:
            obs, _ = env.reset(seed=d, options={"day_scale": 1.0})
            while env.agents:
                _, _, _, _, info = env.step({a: 0 for a in env.agents})
                i0 = info[env.possible_agents[0]]
                lm.append(i0["loading_max"])
                conv &= bool(i0["pf_converged"])
        rows.append(
            {
                "line": k,
                "from_bus": int(env._pf.net.line.from_bus[k]) + 1,
                "to_bus": int(env._pf.net.line.to_bus[k]) + 1,
                "feasible": conv,
                "loading_max_no_control": float(max(lm)),
            }
        )
    env.set_contingency(None)
    df = pd.DataFrame(rows)
    # Binding but relievable outages only: no-control loading in (1, 2]. Outages above
    # 2.0 (several hundred % on branches whose synthetic rating derives from a near-zero
    # nominal flow) cannot be relieved by residential DR and are planning cases; they are
    # kept in the table with ``selected = False``.
    df["relievable"] = (
        df.feasible & (df.loading_max_no_control > 1.0) & (df.loading_max_no_control <= 2.0)
    )
    feas = df[df.relievable].sort_values("loading_max_no_control", ascending=False)
    df["selected"] = df.line.isin(feas.head(n_top).line)
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-root", default="results/runs_m5")
    ap.add_argument("--n-days", type=int, default=20)
    ap.add_argument("--n1-days", type=int, default=10)
    ap.add_argument("--out", default="results")
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
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "legend.frameon": False,
        }
    )

    pols, runs = load_policies(run_root)
    print(f"policies: { {k: len(v) for k, v in pols.items()} }")
    days = [10_000 + k for k in range(args.n_days)]
    n1_days = days[: args.n1_days]

    # ---- scenarios -------------------------------------------------------------
    screen = screen_contingencies()
    screen.to_csv(tab_dir / "n1_screening.csv", index=False)
    sel = screen[screen.selected].line.tolist()
    print("N-1 lines selected:", sel)
    scenarios = {
        "nominal": ({}, 1.0, days),
        "transfer_pop42": ({"population": TRANSFER_POPULATION}, 1.0, days),
        "peak_demand": ({"background_factor": 1.25}, 1.3, days),
    }
    for k in sel:
        r = screen[screen.line == k].iloc[0]
        scenarios[f"n1_line{k}_B{r.from_bus}-B{r.to_bus}"] = ({"contingency": k}, 1.0, n1_days)

    rows = []
    main_methods = [m for m in ORDER if m in pols]
    sweep_methods = sorted(m for m in pols if m.startswith("lagrangian_lr"))
    for scen, (ekw, ds, dlist) in scenarios.items():
        for method in main_methods + (sweep_methods if scen == "nominal" else []):
            if "population" in ekw and method in ("none", "heuristic"):
                continue
            for seed, agent in pols[method].items():
                pop = ekw.get("population", seed)
                env = make_env(pop, **{k: v for k, v in ekw.items() if k != "population"})
                res = [rollout(env, agent, d, ds) for d in dlist]
                row = {"scenario": scen, "method": method, "seed": seed, "population": pop}
                row.update({k: float(np.mean([r[k] for r in res])) for k in KPIS})
                row["loading_max_worst"] = float(max(r["loading_max"] for r in res))
                if scen == "nominal" and method not in ("none", "heuristic"):
                    # stochastic-policy numbers: the constraint the Lagrangian enforces holds
                    # in expectation under the sampled policy; argmax over-curtails (m5-t8)
                    import torch

                    torch.manual_seed(seed)
                    sres = [rollout(env, agent, d, ds, deterministic=False) for d in dlist]
                    row["peak_reduction_pct_stoch"] = float(
                        np.mean([r["peak_reduction_pct"] for r in sres])
                    )
                    row["violation_rate_stoch"] = float(
                        np.mean([r["violation_rate"] for r in sres])
                    )
                    row["jain_frac_stoch"] = float(np.mean([r["jain_frac"] for r in sres]))
                rows.append(row)
                if scen == "nominal":
                    print(
                        f"  {method:16s} seed {seed} | peak {row['peak_reduction_pct']:5.1f}% "
                        f"| J {row['jain_index']:.3f} | viol {row['violation_rate']:.3f} "
                        f"(V {row['viol_voltage']:.2f} L {row['viol_loading']:.2f} C {row['viol_comfort']:.2f})"
                    )
    per = pd.DataFrame(rows)
    per.to_csv(tab_dir / "phase2_per_seed.csv", index=False)

    # ---- ablation table (nominal) m5-t9 ------------------------------------------
    nom = per[per.scenario == "nominal"]
    abl = []
    for method in main_methods:
        d = nom[nom.method == method]
        r = {"method": method, "n_seeds": len(d)}
        for k in KPIS:
            r[f"{k}_mean"], r[f"{k}_ci95"] = float(d[k].mean()), ci95(d[k])
        for k in ("peak_reduction_pct_stoch", "violation_rate_stoch", "jain_frac_stoch"):
            if k in d and d[k].notna().any():
                r[f"{k}_mean"], r[f"{k}_ci95"] = float(d[k].mean()), ci95(d[k].dropna())
        abl.append(r)
    abl = pd.DataFrame(abl)
    abl.to_csv(tab_dir / "safety_ablation.csv", index=False)
    md = [
        "| method | seeds | peak reduction % | Jain (kWh) | Jain (frac) | violation rate | voltage | thermal | comfort | cost | vulnerable/other | stochastic: peak % / violation rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in abl.iterrows():

        def f(m, p=2):
            if r["n_seeds"] > 1:
                return f"{r[f'{m}_mean']:.{p}f} ± {r[f'{m}_ci95']:.{p}f}"
            return f"{r[f'{m}_mean']:.{p}f}"

        md.append(
            f"| {r.method} | {int(r.n_seeds)} | {f('peak_reduction_pct', 1)} | {f('jain_index', 3)} | "
            f"{f('jain_frac', 3)} | {f('violation_rate', 3)} | {f('viol_voltage', 3)} | {f('viol_loading', 3)} | "
            f"{f('viol_comfort', 3)} | {f('cost_total', 0)} | {f('vulnerable_burden_ratio', 2)} | "
            + (
                f"{r['peak_reduction_pct_stoch_mean']:.1f} / {r['violation_rate_stoch_mean']:.3f} |"
                if "peak_reduction_pct_stoch_mean" in r
                and r["peak_reduction_pct_stoch_mean"] == r["peak_reduction_pct_stoch_mean"]
                else "— |"
            )
        )
    (tab_dir / "safety_ablation.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    # ---- targets m5-t11 ----------------------------------------------------------
    tg = []
    for method in main_methods:
        d = nom[nom.method == method]
        chk = check_targets(
            {
                "peak_reduction_pct": d.peak_reduction_pct.mean(),
                "jain_index": d.jain_frac.mean(),  # gate basis: relative burden (2026-10-20)
                "violation_rate": d.violation_rate.mean(),
            }
        )
        seeds_pass = int(
            sum(
                check_targets(
                    {
                        "peak_reduction_pct": r.peak_reduction_pct,
                        "jain_index": r.jain_frac,
                        "violation_rate": r.violation_rate,
                    }
                ).passed
                for _, r in d.iterrows()
            )
        )
        tg.append(
            {
                "method": method,
                "peak_reduction_pct": d.peak_reduction_pct.mean(),
                "jain_frac": d.jain_frac.mean(),
                "jain_kwh": d.jain_index.mean(),
                "violation_rate": d.violation_rate.mean(),
                "peak_ok": chk.results["peak_reduction_pct"]["ok"],
                "jain_ok": chk.results["jain_index"]["ok"],
                "safety_ok": chk.results["violation_rate"]["ok"],
                "all_targets": chk.passed,
                "seeds_passing": f"{seeds_pass}/{len(d)}",
            }
        )
    targets = pd.DataFrame(tg)
    targets.to_csv(tab_dir / "phase2_targets.csv", index=False)
    print(targets.round(3).to_string(index=False))

    # ---- stress tests m5-t12 ----------------------------------------------------
    st = (
        per[per.method.isin(main_methods)]
        .groupby(["scenario", "method"])
        .agg(
            peak_reduction_pct=("peak_reduction_pct", "mean"),
            jain_index=("jain_index", "mean"),
            violation_rate=("violation_rate", "mean"),
            violation_rate_ci95=("violation_rate", ci95),
            viol_loading=("viol_loading", "mean"),
            viol_comfort=("viol_comfort", "mean"),
            loading_max=("loading_max", "mean"),
            loading_max_worst=("loading_max_worst", "max"),
        )
        .reset_index()
    )
    st.to_csv(tab_dir / "stress_tests.csv", index=False)
    print(st.round(3).to_string(index=False))

    # ---- multiplier-lr sweep m5-t5 ------------------------------------------------
    sw = []
    for method in ["lagrangian"] + sweep_methods:  # sweep runs use indicator costs
        for seed, rd in runs.get(method, {}).items():
            h = pd.read_csv(rd / "metrics.csv")
            lam_cols = [c for c in h.columns if c.startswith("lambda_")]
            tail = h.tail(30)
            nomrow = nom[(nom.method == method) & (nom.seed == seed)]
            cfg = json.loads((rd / "summary.json").read_text())
            row = {
                "method": method,
                "seed": seed,
                "multiplier_lr": float(re.sub(r".*lr", "", method)) if "lr" in method else 0.05,
                "final_violation_rate_train": float(tail.violation_rate.mean()),
                "eval_violation_rate": (
                    float(nomrow.violation_rate.mean()) if len(nomrow) else np.nan
                ),
                "eval_peak_reduction_pct": (
                    float(nomrow.peak_reduction_pct.mean()) if len(nomrow) else np.nan
                ),
                "converged": cfg.get("converged"),
            }
            for c in lam_cols:
                row[f"{c}_final"] = float(tail[c].mean())
                row[f"{c}_osc"] = float(tail[c].std())  # oscillation over the last 30 updates
                row[f"{c}_max"] = float(h[c].max())
            sw.append(row)
    if sw:
        swdf = pd.DataFrame(sw).sort_values(["multiplier_lr", "seed"])
        swdf.to_csv(tab_dir / "multiplier_lr_sweep.csv", index=False)
        print(swdf.round(3).to_string(index=False))

    # ---- significance vs unconstrained --------------------------------------------
    from scipy import stats

    sig = []
    base = nom[nom.method == "unconstrained"]
    for method in main_methods:
        if method in ("none", "heuristic", "unconstrained"):
            continue
        d = nom[nom.method == method]
        for k in ("peak_reduction_pct", "jain_index", "violation_rate", "return", "cost_total"):
            a, b = d[k].values, base[k].values
            if a.size > 1 and b.size > 1:
                p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
                delta = float(np.mean([np.sign(x - y) for x in a for y in b]))
            else:
                p, delta = np.nan, np.nan
            sig.append(
                {
                    "method": method,
                    "metric": k,
                    "mean": a.mean(),
                    "mean_unconstrained": b.mean(),
                    "welch_p": p,
                    "cliffs_delta": delta,
                }
            )
    pd.DataFrame(sig).to_csv(tab_dir / "safety_significance.csv", index=False)

    # ---- figures m5-t13 ---------------------------------------------------------------
    curves = {}
    for method in ("unconstrained", "lagrangian", "pid_lagrangian", "joint", "ppo_lagrangian"):
        dfs = []
        for seed, rd in runs.get(method, {}).items():
            h = pd.read_csv(rd / "metrics.csv")
            h["seed"] = seed
            dfs.append(h)
        if dfs:
            curves[method] = pd.concat(dfs)
    if curves:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        panels = [
            ("violation_rate", "training violation rate"),
            ("peak_reduction_pct", "peak reduction (%)"),
            ("jain", "Jain index"),
        ]
        for method, d in curves.items():
            g = d.groupby("timesteps")
            for ax, (m, lab) in zip(axes, panels):
                mu, sd = g[m].mean(), g[m].std().fillna(0)
                mu_s = mu.rolling(5, min_periods=1, center=True).mean()
                ax.plot(mu.index, mu_s, color=COLORS[method], lw=1.8, label=method)
                ax.fill_between(
                    mu.index,
                    (mu - sd).rolling(5, min_periods=1, center=True).mean(),
                    (mu + sd).rolling(5, min_periods=1, center=True).mean(),
                    color=COLORS[method],
                    alpha=0.12,
                    lw=0,
                )
                ax.set_ylabel(lab)
                ax.set_xlabel("environment steps")
        axes[1].axhline(15, color="#52514e", lw=0.8, ls=":")
        axes[2].axhline(0.85, color="#52514e", lw=0.8, ls=":")
        axes[0].legend(fontsize=7.5)
        fig.suptitle("Constrained training on IEEE-14 (mean ± std over seeds)", x=0.02, ha="left")
        fig.tight_layout()
        fig.savefig(fig_dir / "violation_curves.png")
        plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        for ax, method in zip(axes, ("lagrangian", "pid_lagrangian", "joint")):
            d = curves.get(method)
            if d is None:
                ax.set_visible(False)
                continue
            g = d.groupby("update")
            for c, col in zip(
                [c for c in d.columns if c.startswith("lambda_")],
                ["#2a78d6", "#eb6834", "#1baf7a", "#d63a6e"],
            ):
                mu, sd = g[c].mean(), g[c].std().fillna(0)
                ax.plot(mu.index, mu, color=col, lw=1.6, label=c.replace("lambda_", "λ "))
                ax.fill_between(mu.index, mu - sd, mu + sd, color=col, alpha=0.12, lw=0)
            ax2 = ax.twinx()
            v = g["violation_rate"].mean().rolling(5, min_periods=1, center=True).mean()
            ax2.plot(v.index, v, color="#52514e", lw=1.0, ls="--", label="violation rate")
            ax2.set_ylim(0, 1.05)
            ax2.grid(False)
            ax.set_title(method, loc="left")
            ax.set_xlabel("update")
            ax.set_ylabel("multiplier")
            ax.legend(fontsize=7, loc="upper left")
        fig.suptitle(
            "Multiplier dynamics (mean ± std over seeds; dashed: training violation rate)",
            x=0.02,
            ha="left",
        )
        fig.tight_layout()
        fig.savefig(fig_dir / "multiplier_dynamics.png")
        plt.close(fig)

    if sweep_methods:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        for method, col in zip(
            ["lagrangian"] + sweep_methods,
            ["#eb6834", "#2a78d6", "#1baf7a", "#8a5cd6", "#d63a6e"],
        ):
            rd = runs.get(method, {}).get(0)
            if rd is None:
                continue
            h = pd.read_csv(rd / "metrics.csv")
            lam_cols = [c for c in h.columns if c.startswith("lambda_")]
            lr = float(re.sub(r".*lr", "", method)) if "lr" in method else 0.05
            ax.plot(h["update"], h[lam_cols].sum(axis=1), color=col, lw=1.5, label=f"η = {lr:g}")
        ax.set_xlabel("update")
        ax.set_ylabel("Σ multipliers")
        ax.set_title("Dual-ascent rate sweep (seed 0)", loc="left")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "multiplier_lr_sweep.png")
        plt.close(fig)

    # stress bars
    scen_names = list(scenarios)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    meths = [m for m in main_methods if m != "none"]
    w = 0.8 / max(1, len(meths))
    for j, method in enumerate(meths):
        d = st[st.method == method].set_index("scenario").reindex(scen_names)
        x = np.arange(len(scen_names)) + (j - len(meths) / 2 + 0.5) * w
        axes[0].bar(
            x,
            d.violation_rate,
            w,
            yerr=d.violation_rate_ci95,
            color=COLORS[method],
            label=method,
            capsize=1.5,
            error_kw={"lw": 0.6},
        )
        axes[1].bar(x, d.peak_reduction_pct, w, color=COLORS[method], label=method)
    for ax in axes:
        ax.set_xticks(range(len(scen_names)))
        ax.set_xticklabels([s.replace("_", "\n", 1) for s in scen_names], fontsize=7.5)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("violation rate")
    axes[1].set_ylabel("peak reduction (%)")
    axes[1].axhline(15, color="#52514e", lw=0.8, ls=":")
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(
        "Stress and N-1 contingency tests (20 nominal / 10 contingency days)", x=0.02, ha="left"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "stress_violations.png")
    plt.close(fig)

    # benchmark bars
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.9))
    names = [m for m in main_methods]
    for ax, m, lab in zip(
        axes,
        ["peak_reduction_pct", "jain_index", "violation_rate", "cost_total"],
        ["peak reduction (%)", "Jain index", "violation rate", "daily cost"],
    ):
        vals = [abl.set_index("method").loc[n, f"{m}_mean"] for n in names]
        errs = [abl.set_index("method").loc[n, f"{m}_ci95"] for n in names]
        ax.bar(
            names,
            vals,
            yerr=errs,
            color=[COLORS[n] for n in names],
            width=0.6,
            capsize=2,
            error_kw={"lw": 0.8},
        )
        for i, v in enumerate(vals):
            ax.text(
                i,
                v,
                f"{v:.2f}" if m != "cost_total" else f"{v:,.0f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
        ax.set_title(lab, loc="left")
        ax.tick_params(axis="x", labelrotation=35, labelsize=7)
        ax.grid(axis="x", visible=False)
    axes[0].axhline(15, color="#52514e", lw=0.8, ls=":")
    axes[1].axhline(0.85, color="#52514e", lw=0.8, ls=":")
    fig.suptitle(
        "Phase-2 benchmark on 20 held-out days (mean ± 95 % CI over seeds)", x=0.02, ha="left"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "safety_benchmark.png")
    plt.close(fig)

    passing = targets[targets.all_targets].method.tolist()
    print(f"methods meeting all three targets: {passing}")
    print(
        f"RESULT: {'PASS' if passing else 'FAIL'} — Phase-2 joint targets "
        "(Jain >= 0.85, peak >= 15 %, zero violations on the 20 protocol days)"
    )
    return 0 if passing else 1


if __name__ == "__main__":
    raise SystemExit(main())
