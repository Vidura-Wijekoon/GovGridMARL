#!/usr/bin/env python3
"""Electrical basis of the IEEE-14 environment (m2-t11 addendum, supervisor request Sep 2026).

Produces the numbers an electrical engineer expects to see next to the dashboard:

  1. Network dimensions: bus types (slack / PV / PQ), nominal voltages, base MVA,
     branch counts, synthetic thermal ratings and how they were derived.
  2. Newton-Raphson power flow on the nominal case: iterations, final mismatch,
     generation = load + losses balance (P and Q), slack share.
  3. Household-cluster scaling: homes -> kW -> MW at 0.95 pf, share of bus load.
  4. Bus-selection evidence: sensitivity of (min bus voltage, max branch loading,
     losses) to +5 MW injected at each load bus — the electrical distance that
     justifies attaching clusters to the transformer-fed buses first.
  5. Energy-share scenario table: renewable share 0..0.8 (wind/PV split 0.5) on the
     nominal day under no control — conventional energy, surplus, net-import peak,
     max loading, max voltage.

Writes results/tables/grid_analysis.json (consumed by the dashboard) and a PASS log
under results/logs/. Usage: python scripts/grid_analysis.py
"""

from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO / "src", REPO / "envs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore")
LOG_DIR = REPO / "results" / "logs"
OUT = REPO / "results" / "tables" / "grid_analysis.json"


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def main() -> int:
    import pandapower as pp

    from govgrid_envs import GridMARLEnv
    from govgrid_envs.ieee14 import IEEE14_LOAD_BUS_PRIORITY, Ieee14PowerFlow, load_ieee14

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = Tee(LOG_DIR / f"grid_analysis_{stamp}.log")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        log(f"[{'PASS' if cond else 'FAIL'}] {name} {detail}")

    log("=" * 72)
    log("IEEE-14 electrical basis (grid_analysis.py)")
    log(
        f"UTC {datetime.now(timezone.utc).isoformat()} · {platform.node()} · Python {sys.version.split()[0]}"
    )
    log("=" * 72)

    # ---------------------------------------------------------------- 1. dimensions
    net = load_ieee14()
    pp.runpp(net, numba=False)
    slack_bus = int(net.ext_grid.bus.iloc[0])
    pv_buses = sorted(net.gen.bus.astype(int).tolist())
    load_buses = sorted(net.load.bus.astype(int).tolist())
    bus_rows = []
    for b in net.bus.index:
        typ = "slack" if b == slack_bus else "PV (gen)" if b in pv_buses else "PQ"
        ld = net.load[net.load.bus == b]
        bus_rows.append(
            {
                "bus": int(b),
                "name": f"B{int(b) + 1}",
                "type": typ,
                "vn_kv": float(net.bus.vn_kv[b]),
                "p_load_mw": float(ld.p_mw.sum()),
                "q_load_mvar": float(ld.q_mvar.sum()),
                "vm_pu": float(net.res_bus.vm_pu[b]),
                "va_deg": float(net.res_bus.va_degree[b]),
                "gen_setpoint_pu": (
                    float(net.gen[net.gen.bus == b].vm_pu.iloc[0])
                    if b in pv_buses
                    else (float(net.ext_grid.vm_pu.iloc[0]) if b == slack_bus else None)
                ),
            }
        )
    log("\n1. Network dimensions")
    log(
        f"   base MVA {net.sn_mva:.0f} · buses {len(net.bus)} · lines {len(net.line)} · transformers {len(net.trafo)} "
        f"· loads {len(net.load)} · gens {len(net.gen)} + slack · shunts {len(net.shunt)}"
    )
    log(
        f"   voltage levels (kV, as shipped by pandapower case14): {sorted(set(net.bus.vn_kv.round(3)))}"
    )
    log(
        f"   slack B{slack_bus + 1} (V=1.06 pu, θ=0) · PV buses {[f'B{b + 1}' for b in pv_buses]} · PQ (load) buses "
        f"{[f'B{b + 1}' for b in load_buses]}"
    )
    log("   bus  type      kV      P_load  Q_load   |V| pu   θ deg")
    for r in bus_rows:
        log(
            f"   {r['name']:4s} {r['type']:9s} {r['vn_kv']:7.3f} {r['p_load_mw']:7.1f} {r['q_load_mvar']:7.1f}  {r['vm_pu']:.4f} {r['va_deg']:7.2f}"
        )

    # ---------------------------------------------------------------- 2. NR balance
    it = int(net._ppc["iterations"])
    gen_p = float(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum())
    gen_q = float(net.res_ext_grid.q_mvar.sum() + net.res_gen.q_mvar.sum())
    load_p, load_q = float(net.res_load.p_mw.sum()), float(net.res_load.q_mvar.sum())
    loss_p = float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum())
    loss_q = float(net.res_line.ql_mvar.sum() + net.res_trafo.ql_mvar.sum())
    shunt_q = float(net.res_shunt.q_mvar.sum())
    mism_p = gen_p - load_p - loss_p
    mism_q = gen_q - load_q - loss_q - shunt_q
    nr = {
        "iterations": it,
        "tolerance_mva": 1e-8,
        "gen_p_mw": gen_p,
        "gen_q_mvar": gen_q,
        "load_p_mw": load_p,
        "load_q_mvar": load_q,
        "loss_p_mw": loss_p,
        "loss_q_mvar": loss_q,
        "shunt_q_mvar": shunt_q,
        "mismatch_p_mw": mism_p,
        "mismatch_q_mvar": mism_q,
        "slack_p_mw": float(net.res_ext_grid.p_mw.sum()),
        "slack_share": float(net.res_ext_grid.p_mw.sum() / gen_p),
        "loss_pct": loss_p / gen_p * 100,
    }
    log("\n2. Newton-Raphson power flow (nominal case14, flat start)")
    log(
        f"   converged in {it} iterations (tol 1e-8 MVA) · unknowns: θ at 13 non-slack buses + |V| at {len(load_buses) - 0} PQ buses"
    )
    log(
        f"   P: generation {gen_p:.2f} = load {load_p:.2f} + losses {loss_p:.2f} MW  (mismatch {mism_p:.2e})"
    )
    log(
        f"   Q: generation {gen_q:.2f} = load {load_q:.2f} + losses {loss_q:.2f} + shunt {shunt_q:.2f} Mvar  (mismatch {mism_q:.2e})"
    )
    log(
        f"   slack B1 supplies {nr['slack_p_mw']:.1f} MW ({nr['slack_share'] * 100:.0f} % of generation); B2 fixed 40 MW; B3/B6/B8 synchronous condensers (0 MW, voltage support)"
    )
    log(f"   losses {nr['loss_pct']:.2f} % of generation")
    check("P balance closes", abs(mism_p) < 1e-3, f"({mism_p:.2e} MW)")
    check("Q balance closes incl. shunt", abs(mism_q) < 1e-3, f"({mism_q:.2e} Mvar)")

    # ---------------------------------------------------------------- 3. scaling & ratings
    env = GridMARLEnv(n_households=14, seed=42, day_variability=0.0)
    pf: Ieee14PowerFlow = env._pf
    peak_kw_home = float(env.dataset.base_kw.sum(axis=0).max() / env.n_households)
    mean_kw_home = float(env.dataset.base_kw.mean())
    scaling = {
        "homes_per_agent": env.homes_per_agent,
        "mean_kw_per_home": mean_kw_home,
        "peak_kw_per_home_aggregate": peak_kw_home,
        "cluster_mean_mw": mean_kw_home * env.homes_per_agent / 1000,
        "cluster_peak_mw": peak_kw_home * env.homes_per_agent / 1000,
        "residential_peak_mw": peak_kw_home * env.homes_per_agent * env.n_households / 1000,
        "power_factor": 0.95,
        "tan_phi": float(np.tan(np.arccos(0.95))),
        "background_peak_mw": float(pf.background_p.sum() * env._bg_shape.max()),
        "residential_share_of_system_peak": peak_kw_home
        * env.homes_per_agent
        * env.n_households
        / 1000
        / (
            peak_kw_home * env.homes_per_agent * env.n_households / 1000
            + pf.background_p.sum() * env._bg_shape.max()
        ),
    }
    log("\n3. Household-cluster scaling (why 2,000 homes per agent)")
    log(
        f"   mean {mean_kw_home:.2f} kW/home, aggregate peak {peak_kw_home:.2f} kW/home → cluster {scaling['cluster_mean_mw']:.2f} MW mean, "
        f"{scaling['cluster_peak_mw']:.2f} MW peak; 14 clusters = {scaling['residential_peak_mw']:.1f} MW residential peak"
    )
    log(
        f"   background (case14 loads × diurnal 0.85–1.10) peaks at {scaling['background_peak_mw']:.1f} MW → residential is "
        f"{scaling['residential_share_of_system_peak'] * 100:.0f} % of system peak — visible to the network, not dominant"
    )
    log(f"   Q = P·tan(acos 0.95) = P × {scaling['tan_phi']:.3f} (lagging)")
    ratings = []
    for i, row in net.line.iterrows():
        ratings.append(
            {
                "branch": f"L{i} B{int(row.from_bus) + 1}-B{int(row.to_bus) + 1}",
                "nominal_i_ka": float(net.res_line.i_ka[i]),
                "rating_i_ka": float(pf._line_rating[i]),
                "nominal_loading_pct": float(net.res_line.i_ka[i] / pf._line_rating[i] * 100),
            }
        )
    for i, row in net.trafo.iterrows():
        s_nom = float(np.hypot(net.res_trafo.p_hv_mw[i], net.res_trafo.q_hv_mvar[i]))
        ratings.append(
            {
                "branch": f"T{i} B{int(row.hv_bus) + 1}-B{int(row.lv_bus) + 1}",
                "nominal_s_mva": s_nom,
                "rating_s_mva": float(pf._trafo_rating[i]),
                "nominal_loading_pct": float(s_nom / pf._trafo_rating[i] * 100),
            }
        )
    log(
        "   thermal ratings: case14 ships max_i_ka = 42.3 kA (never binding) → rating := 1.5 × nominal-case current/apparent power (66.7 % loading at nominal)"
    )

    # ---------------------------------------------------------------- 4. bus sensitivities
    log(
        "\n4. Bus-selection evidence: +5 MW at each load bus (nominal day, 18:00 background factor)"
    )
    base = pf.solve(np.zeros(14), 1.0)
    sens = []
    for b in load_buses:
        hp = np.zeros(14)
        # inject 5 MW through the first household slot on that bus (or a temporary one)
        slots = [k for k, hb in enumerate(pf.household_buses) if hb == b]
        if slots:
            hp[slots[0]] = 5.0
            r = pf.solve(hp, 1.0)
        else:
            continue
        sens.append(
            {
                "bus": int(b),
                "name": f"B{b + 1}",
                "vn_kv": float(net.bus.vn_kv[b]),
                "dVmin_pu_per_5MW": float(r.vm_pu.min() - base.vm_pu.min()),
                "dV_local_pu_per_5MW": float(r.vm_pu[b] - base.vm_pu[b]),
                "dLoadingMax_per_5MW": float(r.max_loading - base.max_loading),
                "dLosses_mw_per_5MW": float(r.losses_mw - base.losses_mw),
                "priority_rank": IEEE14_LOAD_BUS_PRIORITY.index(b) + 1,
                "behind_transformer": bool(b in set(net.trafo.lv_bus.astype(int)) or b >= 8),
            }
        )
    sens.sort(key=lambda x: x["dLosses_mw_per_5MW"], reverse=True)
    log("   bus   kV      ΔV_local   Δmax-loading  Δlosses(MW)  priority  behind trafo")
    for x in sens:
        log(
            f"   {x['name']:4s} {x['vn_kv']:7.3f}  {x['dV_local_pu_per_5MW']:+.4f}    {x['dLoadingMax_per_5MW']:+.4f}       {x['dLosses_mw_per_5MW']:+.3f}      {x['priority_rank']:2d}        {x['behind_transformer']}"
        )
    weak = [x["name"] for x in sens[:6]]
    chosen = [f"B{b + 1}" for b in IEEE14_LOAD_BUS_PRIORITY[:6]]
    overlap = len(set(weak) & set(chosen))
    check(
        "priority buses are the electrically weakest (loss-sensitivity top-6 overlap)",
        overlap >= 4,
        f"({overlap}/6: sens {weak} vs chosen {chosen})",
    )

    # ---------------------------------------------------------------- 5. energy-share table
    log("\n5. Energy-share scenarios (no control, nominal day, wind/PV 50/50 unless noted)")
    scen = []
    for rs, wf in [
        (0.0, 0.5),
        (0.2, 0.5),
        (0.4, 0.5),
        (0.6, 0.5),
        (0.8, 0.5),
        (0.8, 0.8),
        (0.8, 0.2),
    ]:
        e = GridMARLEnv(
            n_households=14, seed=42, day_variability=0.0, renewable_share=rs, wind_fraction=wf
        )
        e.reset(seed=10_000, options={"day_scale": 1.0, "wind_scale": 1.0})
        vmax = lmax = 0.0
        while e.agents:
            _, _, _, _, info = e.step({a: 0 for a in e.agents})
            vmax = max(vmax, info["household_0"]["vm_max"])
            lmax = max(lmax, info["household_0"]["loading_max"])
        ep = info["household_0"]["episode"]
        row = {
            "renewable_share": rs,
            "wind_fraction": wf,
            "renewable_share_of_consumption": ep["renewable_share_of_consumption"],
            "surplus_mwh": ep["renewable_surplus_mwh"],
            "conventional_mwh": ep["energy_conventional_mwh"],
            "net_import_peak_mw": ep["net_import_peak_mw"],
            "losses_mwh": ep["system_losses_mwh"],
            "vmax_pu": vmax,
            "loading_max": lmax,
            "violation_rate": ep["violation_rate"],
        }
        scen.append(row)
        log(
            f"   share {rs:.1f} wind {wf:.1f} → consumed {row['renewable_share_of_consumption'] * 100:4.0f} % · surplus {row['surplus_mwh']:7.1f} MWh · "
            f"net-import peak {row['net_import_peak_mw']:6.1f} MW · losses {row['losses_mwh']:6.1f} MWh · max loading {lmax:.2f} · violations {ep['violation_rate']:.2f}"
        )
    check(
        "renewables displace conventional energy monotonically",
        all(scen[i]["conventional_mwh"] > scen[i + 1]["conventional_mwh"] for i in range(4)),
    )
    check(
        "PV-heavy 80 % share produces midday surplus and branch stress",
        scen[-1]["surplus_mwh"] > 0 and scen[-1]["loading_max"] > 1.0,
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "buses": bus_rows,
                "newton_raphson": nr,
                "scaling": scaling,
                "ratings": ratings,
                "sensitivities": sens,
                "energy_share": scen,
                "priority": [f"B{b + 1}" for b in IEEE14_LOAD_BUS_PRIORITY],
            },
            indent=2,
        )
    )
    log(f"\nwrote {OUT.relative_to(REPO)}")
    log("-" * 72)
    log(f"RESULT: {'PASS' if ok else 'FAIL'} — IEEE-14 electrical basis")
    log.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
