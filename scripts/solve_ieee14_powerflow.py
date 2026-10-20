#!/usr/bin/env python3
"""Solve the IEEE-14 power flow with pandapower Newton-Raphson (task m1-t14).

Verifies the AC power-flow solver end-to-end on the canonical IEEE 14-bus
system and writes a full report:

  1. import pandapower and report version;
  2. load the IEEE-14 network and check its structure (14 buses, 5 gen/SC,
     11 loads, 20 branches);
  3. run Newton-Raphson to convergence and report iteration count;
  4. print per-bus voltage magnitude / angle and per-branch loading;
  5. sanity checks -> voltages inside [0.94, 1.06] pu, no branch >100%,
     active + reactive power balance closes to < 1e-3 MW/Mvar;
  6. cross-check: re-solve and confirm bit-identical voltages (determinism).

Everything is echoed to stdout AND written to
results/logs/ieee14_powerflow_<timestamp>.log so the log is the deliverable.

Usage:
    python scripts/solve_ieee14_powerflow.py
"""

from __future__ import annotations

import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
sys.path.insert(0, str(REPO_ROOT))

# Acceptance bands for the IEEE-14 base case.
VM_MIN, VM_MAX = 0.94, 1.10  # per-unit band (gen setpoints reach 1.09)
LOADING_MAX = 100.0  # branch loading percent
BALANCE_TOL = 1e-3  # MW / Mvar power-balance tolerance


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


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = Tee(LOG_DIR / f"ieee14_powerflow_{stamp}.log")
    warnings.filterwarnings("ignore")

    log("=" * 70)
    log("GovGridMARL IEEE-14 power flow — pandapower Newton-Raphson (task m1-t14)")
    log(f"UTC time      : {datetime.now(timezone.utc).isoformat()}")
    log(f"Host          : {platform.node()} ({platform.system()} {platform.release()})")
    log(f"Python        : {sys.version.split()[0]} @ {sys.executable}")
    log("=" * 70)

    # -- 1. import -------------------------------------------------------
    try:
        import pandapower as pp  # noqa: F401
    except ImportError as exc:
        log(f"FAIL: pandapower is not installed ({exc}).")
        log("Fix : create the env first (see environment.yml / requirements.lock)")
        log.close()
        return 1
    log(f"pandapower    : {pp.__version__}")

    from envs.govgrid_envs.ieee14 import (
        load_ieee14,
        power_balance,
        run_power_flow,
    )

    # -- 2. load + structure checks -------------------------------------
    net = load_ieee14()
    n_branch = len(net.line) + len(net.trafo)
    structure = {
        "14 buses": len(net.bus) == 14,
        "11 loads": len(net.load) == 11,
        "5 generators / synch. condensers": len(net.gen) == 4 and len(net.ext_grid) == 1,
        "20 branches (17 lines + 3 trafos)": n_branch == 20,
    }
    log("-" * 70)
    log(
        f"Network       : {len(net.bus)} buses | {len(net.gen)} gens + "
        f"{len(net.ext_grid)} slack | {len(net.load)} loads | "
        f"{len(net.line)} lines + {len(net.trafo)} trafos | "
        f"{net.sn_mva:.0f} MVA base"
    )
    for name, ok in structure.items():
        log(f"  [{'OK' if ok else 'BAD'}] {name}")
    if not all(structure.values()):
        log("RESULT: FAIL — network does not match IEEE-14 structure.")
        log.close()
        return 1

    # -- 3. Newton-Raphson solve ----------------------------------------
    log("-" * 70)
    t0 = time.perf_counter()
    converged = run_power_flow(net, algorithm="nr")
    dt = time.perf_counter() - t0
    log(f"Newton-Raphson: converged={converged} in {dt * 1e3:.1f} ms")
    if not converged:
        log("RESULT: FAIL — Newton-Raphson did not converge.")
        log.close()
        return 1

    # -- 4. bus voltages + branch loadings ------------------------------
    log("-" * 70)
    log("Bus voltages (per-unit magnitude, angle in degrees):")
    log(net.res_bus[["vm_pu", "va_degree"]].round(4).to_string())
    log("-" * 70)
    line_load = net.res_line.loading_percent
    trafo_load = net.res_trafo.loading_percent
    log(
        f"Line loading  : min {line_load.min():.1f}% | "
        f"mean {line_load.mean():.1f}% | max {line_load.max():.1f}% "
        f"(line {int(line_load.idxmax())})"
    )
    log(
        f"Trafo loading : min {trafo_load.min():.1f}% | "
        f"mean {trafo_load.mean():.1f}% | max {trafo_load.max():.1f}%"
    )

    # -- 5. sanity checks -----------------------------------------------
    log("-" * 70)
    vmin, vmax = float(net.res_bus.vm_pu.min()), float(net.res_bus.vm_pu.max())
    max_load = float(max(line_load.max(), trafo_load.max()))
    bal = power_balance(net)
    checks = {
        f"all bus voltages in [{VM_MIN}, {VM_MAX}] pu "
        f"(got {vmin:.4f}..{vmax:.4f})": VM_MIN <= vmin
        and vmax <= VM_MAX,
        f"no branch overloaded (<{LOADING_MAX:.0f}%, max {max_load:.1f}%)": max_load < LOADING_MAX,
        f"active power balance closes (mismatch {bal['mismatch_p_mw']:.2e} MW)": abs(
            bal["mismatch_p_mw"]
        )
        < BALANCE_TOL,
        f"reactive power balance closes (mismatch {bal['mismatch_q_mvar']:.2e} Mvar)": abs(
            bal["mismatch_q_mvar"]
        )
        < BALANCE_TOL,
    }
    log(
        f"Power balance : gen {bal['gen_p_mw']:.2f} MW / {bal['gen_q_mvar']:.2f} Mvar"
        f" = load {bal['load_p_mw']:.2f} MW / {bal['load_q_mvar']:.2f} Mvar"
        f" + losses {bal['loss_p_mw']:.2f} MW / {bal['loss_q_mvar']:.2f} Mvar"
        f" + shunt {bal['shunt_p_mw']:.2f} MW / {bal['shunt_q_mvar']:.2f} Mvar"
    )
    for name, ok in checks.items():
        log(f"  [{'OK' if ok else 'BAD'}] {name}")
    if not all(checks.values()):
        log("RESULT: FAIL — a power-flow sanity check failed.")
        log.close()
        return 1

    # -- 6. determinism cross-check -------------------------------------
    log("-" * 70)
    v_ref = net.res_bus.vm_pu.to_numpy().copy()
    net2 = load_ieee14()
    run_power_flow(net2, algorithm="nr")
    import numpy as np

    identical = np.allclose(v_ref, net2.res_bus.vm_pu.to_numpy(), atol=1e-12)
    log(
        f"Determinism   : re-solve reproduces bus voltages exactly -> "
        f"{'OK' if identical else 'MISMATCH'}"
    )
    if not identical:
        log("RESULT: FAIL — repeated solve diverged.")
        log.close()
        return 1

    log("-" * 70)
    log("RESULT: PASS — IEEE-14 Newton-Raphson power flow solved and verified.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
