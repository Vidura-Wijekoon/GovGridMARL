"""IEEE-14 bus system via pandapower (tasks m1-t14, m2-t11).

Provides the canonical 14-bus network and a helper to run Newton-Raphson
power flow. Phase-1, task m1-t14: confirm the solver converges.
"""

from __future__ import annotations


def load_ieee14():
    """Return the pandapower IEEE-14 network.

    Implemented in Month 1 (m1-t14)::

        import pandapower.networks as pn
        return pn.case14()
    """
    raise NotImplementedError("load_ieee14 — scheduled for Month 1 (m1-t14)")


def run_power_flow(net):
    """Run Newton-Raphson power flow and return convergence status (m1-t14)."""
    raise NotImplementedError("run_power_flow — scheduled for Month 1 (m1-t14)")
