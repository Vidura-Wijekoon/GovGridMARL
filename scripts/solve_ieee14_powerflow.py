"""Standalone script for task m1-t14: solve IEEE-14 power flow with pandapower.

Run once the environment is provisioned (Month 1):

    python scripts/solve_ieee14_powerflow.py

Expected: Newton-Raphson converges; prints bus voltages and line loadings.
"""
from __future__ import annotations


def main() -> None:
    try:
        import pandapower as pp
        import pandapower.networks as pn
    except ImportError:
        raise SystemExit(
            "pandapower not installed. Create the env first (see environment.yml)."
        )

    net = pn.case14()
    pp.runpp(net, algorithm="nr")  # Newton-Raphson
    print(f"Converged: {net.converged}")
    print(net.res_bus[["vm_pu", "va_degree"]].round(4).to_string())


if __name__ == "__main__":
    main()
