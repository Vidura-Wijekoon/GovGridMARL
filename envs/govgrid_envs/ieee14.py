"""IEEE-14 bus system via pandapower (tasks m1-t14, m2-t11).

Provides the canonical 14-bus network, a Newton-Raphson power-flow helper
(m1-t14), and the household-to-bus mapping plus a fast repeated power-flow
solver used by the multi-agent environment (m2-t11).

Design notes (m2-t11)
---------------------
* ``case14`` has 11 load buses (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13 in
  0-based pandapower indexing). Household agents are *residential clusters*
  (``homes_per_agent`` homes each, default 2,000) so that their MW-scale
  demand is visible to the network. They are attached round-robin to the
  low-voltage load buses first (buses 8-13 sit behind the transformers and
  are the electrically weakest), then the remaining load buses.
* The original case14 loads are kept as an uncontrollable *background*
  (commercial / industrial) demand, modulated by a mild diurnal factor.
* ``case14`` ships without thermal limits (``max_i_ka`` = 42 kA,
  transformers 9,900 MVA), so no line could ever overload. We impose
  synthetic ratings at ``rating_factor`` x the nominal base-case current
  (default 1.5, i.e. 50 % headroom), a standard trick in RL-on-IEEE-cases
  studies. The factor is a config knob and is documented in the Methods.
* Power flow: LightSim2Grid's ``GridModel`` (C++ Newton-Raphson, ~0.1 ms
  per solve on IEEE-14) with a transparent fallback to ``pandapower.runpp``
  (~20 ms). Both give identical results to 1e-6 pu (checked in tests).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np

# LV / weak buses first (behind transformers), then the HV load buses.
IEEE14_LOAD_BUS_PRIORITY: tuple[int, ...] = (8, 9, 10, 11, 12, 13, 4, 3, 1, 2, 5)


def load_ieee14():
    """Return the pandapower IEEE-14 network (task m1-t14).

    Uses pandapower's built-in ``case14`` — the standard IEEE 14-bus test
    system (14 buses, 4 generators + slack, 11 loads, 15 lines + 5
    transformers = 20 branches), 100 MVA base.
    """
    import pandapower.networks as pn

    return pn.case14()


def run_power_flow(net, algorithm: str = "nr", **kwargs) -> bool:
    """Run power flow on ``net`` and return convergence status (task m1-t14)."""
    import pandapower as pp

    kwargs.setdefault("numba", False)
    pp.runpp(net, algorithm=algorithm, **kwargs)
    return bool(net.converged)


def power_balance(net) -> dict[str, Any]:
    """Return a generation / load / loss balance summary after a solved flow.

    All quantities in MW / Mvar. ``ext_grid`` (slack) injections are included
    in generation. Shunt elements (e.g. the case14 capacitor) are treated as
    consumption with sign, so a capacitor injecting reactive power shows a
    negative ``shunt_q_mvar``. Mismatch = generation - load - losses - shunt;
    it should be ~0 for a converged solve.
    """
    gen_p = float(net.res_ext_grid.p_mw.sum()) + float(net.res_gen.p_mw.sum())
    gen_q = float(net.res_ext_grid.q_mvar.sum()) + float(net.res_gen.q_mvar.sum())
    load_p = float(net.res_load.p_mw.sum())
    load_q = float(net.res_load.q_mvar.sum())
    loss_p = float(net.res_line.pl_mw.sum()) + float(net.res_trafo.pl_mw.sum())
    loss_q = float(net.res_line.ql_mvar.sum()) + float(net.res_trafo.ql_mvar.sum())
    shunt_p = float(net.res_shunt.p_mw.sum()) if len(net.shunt) else 0.0
    shunt_q = float(net.res_shunt.q_mvar.sum()) if len(net.shunt) else 0.0
    return {
        "gen_p_mw": gen_p,
        "gen_q_mvar": gen_q,
        "load_p_mw": load_p,
        "load_q_mvar": load_q,
        "loss_p_mw": loss_p,
        "loss_q_mvar": loss_q,
        "shunt_p_mw": shunt_p,
        "shunt_q_mvar": shunt_q,
        "mismatch_p_mw": gen_p - load_p - loss_p - shunt_p,
        "mismatch_q_mvar": gen_q - load_q - loss_q - shunt_q,
    }


# --------------------------------------------------------------------------- #
# m2-t11: household mapping                                                    #
# --------------------------------------------------------------------------- #
def household_bus_map(n_households: int) -> list[int]:
    """Round-robin assignment of household clusters to IEEE-14 load buses."""
    return [
        IEEE14_LOAD_BUS_PRIORITY[i % len(IEEE14_LOAD_BUS_PRIORITY)] for i in range(n_households)
    ]


def attach_households(net, n_households: int, power_factor: float = 0.95) -> list[int]:
    """Add one controllable load per household cluster; return pandapower load ids.

    Loads start at 0 MW and are driven by the environment each step. The
    reactive power follows the active power at a fixed lagging power factor.
    """
    import pandapower as pp

    buses = household_bus_map(n_households)
    ids = []
    for i, b in enumerate(buses):
        ids.append(
            int(
                pp.create_load(
                    net, bus=b, p_mw=0.0, q_mvar=0.0, name=f"household_{i}", controllable=True
                )
            )
        )
    net.load.loc[ids, "power_factor"] = power_factor
    return ids


WIND_BUS = 1  # B2 (135 kV generator bus): transmission-connected wind farm


def attach_renewables(
    net, load_buses: list[int], wind_bus: int = WIND_BUS
) -> tuple[list[int], int]:
    """Add static generators for distributed PV (one per load bus) and one wind farm.

    Returns ``(pv_sgen_ids, wind_sgen_id)``. All start at 0 MW; the environment
    drives them each step from the renewable-share scenario (see
    :class:`~govgrid_envs.grid_marl_env.GridMARLEnv`). Renewables are modelled
    as PQ injections at unity power factor (inverter-interfaced), so they
    displace slack-bus generation and change bus voltages / branch flows.
    """
    import pandapower as pp

    pv_ids = [
        int(pp.create_sgen(net, bus=b, p_mw=0.0, q_mvar=0.0, name=f"pv_bus{b}")) for b in load_buses
    ]
    wind_id = int(pp.create_sgen(net, bus=wind_bus, p_mw=0.0, q_mvar=0.0, name="wind_farm"))
    return pv_ids, wind_id


def set_thermal_limits(net, rating_factor: float = 1.5) -> np.ndarray:
    """Impose synthetic line / transformer ratings at ``rating_factor`` x base flow.

    ``case14`` has no meaningful ratings; this makes the branch-loading
    constraint meaningful. Must be called on the *nominal* network (before
    households are energised). Line ratings are written to ``max_i_ka``
    (only used by pandapower for ``loading_percent``). Transformer ratings
    are **returned** (MVA) rather than written to ``sn_mva`` because
    pandapower defines ``vk_percent`` relative to ``sn_mva`` — changing it
    would change the transformer impedance.
    """
    import pandapower as pp

    pp.runpp(net, numba=False)
    i_line = np.maximum(net.res_line.i_ka.values, 1e-3)
    net.line["max_i_ka"] = i_line * rating_factor
    s_trafo = np.maximum(
        np.hypot(net.res_trafo.p_hv_mw.values, net.res_trafo.q_hv_mvar.values), 1.0
    )
    return s_trafo * rating_factor


def background_shape(horizon: int = 24) -> np.ndarray:
    """Mild commercial/industrial diurnal factor for the case14 background loads."""
    t = (np.arange(horizon) + 0.5) * (24.0 / horizon)
    s = 0.85 + 0.25 * np.exp(-0.5 * ((t - 13.0) / 4.0) ** 2)
    return s


# --------------------------------------------------------------------------- #
# Fast repeated power flow                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class PFResult:
    converged: bool
    vm_pu: np.ndarray  # (n_bus,)
    line_loading: np.ndarray  # (n_line,) fraction of rating (1.0 = 100 %)
    trafo_loading: np.ndarray  # (n_trafo,)
    slack_p_mw: float  # total conventional generation incl. slack (MW)
    losses_mw: float
    ren_p_mw: float = 0.0  # total renewable injection (PV + wind) this step (MW)
    ext_grid_p_mw: float = 0.0  # slack bus injection alone (negative = export)
    gen_p_mw: tuple = ()  # per generator (case14 order), MW
    n_iter: int = -1  # Newton-Raphson iterations (pandapower backend only)

    @property
    def max_loading(self) -> float:
        return float(max(self.line_loading.max(), self.trafo_loading.max()))


class Ieee14PowerFlow:
    """Repeated AC power flow on IEEE-14 with per-step load updates.

    Parameters
    ----------
    n_households:
        Number of household cluster loads to attach (see :func:`attach_households`).
    rating_factor:
        Synthetic thermal rating multiplier (see :func:`set_thermal_limits`).
    backend:
        ``"auto"`` (LightSim2Grid if importable), ``"lightsim"`` or ``"pandapower"``.
    """

    def __init__(self, n_households: int, rating_factor: float = 1.5, backend: str = "auto"):
        import pandapower as pp

        self.net = load_ieee14()
        trafo_rating = set_thermal_limits(self.net, rating_factor)
        self.background_ids = list(self.net.load.index)
        self.background_p = self.net.load.p_mw.values.copy()
        self.background_q = self.net.load.q_mvar.values.copy()
        self.household_ids = attach_households(self.net, n_households)
        self.household_buses = self.net.load.loc[self.household_ids, "bus"].astype(int).tolist()
        self.load_buses = self.net.load.loc[self.background_ids, "bus"].astype(int).tolist()
        self.pv_ids, self.wind_id = attach_renewables(self.net, self.load_buses)
        self.wind_bus = WIND_BUS
        self.n_bus = len(self.net.bus)
        self._line_rating = self.net.line.max_i_ka.values.copy()
        self._trafo_rating = trafo_rating
        self.backend = backend
        self._gm = None
        if backend in ("auto", "lightsim"):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    from lightsim2grid.gridmodel import init_from_pandapower

                    pp.runpp(self.net, numba=False)
                    self._gm = init_from_pandapower(self.net)
                self.backend = "lightsim"
            except Exception as exc:  # pragma: no cover - depends on install
                if backend == "lightsim":
                    raise
                warnings.warn(
                    f"LightSim2Grid unavailable ({exc}); using pandapower backend", stacklevel=2
                )
                self.backend = "pandapower"
        else:
            self.backend = "pandapower"
        self._V0 = np.ones(self.n_bus, dtype=complex)
        self.contingency: int | None = None

    # -- N-1 contingencies (m5-t12) -------------------------------------------
    def set_contingency(self, line_idx: int | None) -> None:
        """Take one line out of service (``None`` restores the intact network).

        Ratings stay those of the intact nominal case, so the remaining branches
        carry the diverted flow against their original limits - the standard
        N-1 security check. Islanding contingencies are reported as diverged.
        """
        if self.contingency is not None:
            self._set_line(self.contingency, True)
        self.contingency = None if line_idx is None else int(line_idx)
        if self.contingency is not None:
            if not 0 <= self.contingency < len(self.net.line):
                raise ValueError(f"line index out of range: {self.contingency}")
            self._set_line(self.contingency, False)

    def _set_line(self, idx: int, in_service: bool) -> None:
        self.net.line.loc[idx, "in_service"] = bool(in_service)
        if self.backend == "lightsim" and self._gm is not None:
            if in_service:
                self._gm.reactivate_powerline(int(idx))
            else:
                self._gm.deactivate_powerline(int(idx))

    @property
    def n_lines(self) -> int:
        return int(len(self.net.line))

    # -- API -----------------------------------------------------------------
    def solve(
        self,
        household_p_mw: np.ndarray,
        background_factor: float = 1.0,
        power_factor: float = 0.95,
        pv_mw: np.ndarray | None = None,
        wind_mw: float = 0.0,
    ) -> PFResult:
        """Set household loads (MW), background scaling and renewable injections, then solve.

        ``pv_mw`` is one value per load bus (order = ``self.load_buses``); ``wind_mw``
        is the wind-farm injection at ``self.wind_bus``. Both default to zero
        (the Phase-1 baseline without renewables).
        """
        tan_phi = float(np.tan(np.arccos(power_factor)))
        hp = np.asarray(household_p_mw, dtype=float)
        hq = hp * tan_phi
        bp = self.background_p * background_factor
        bq = self.background_q * background_factor
        pv = np.zeros(len(self.pv_ids)) if pv_mw is None else np.asarray(pv_mw, dtype=float)
        if self.backend == "lightsim":
            return self._solve_lightsim(hp, hq, bp, bq, pv, float(wind_mw))
        return self._solve_pandapower(hp, hq, bp, bq, pv, float(wind_mw))

    def _solve_lightsim(self, hp, hq, bp, bq, pv, wind) -> PFResult:
        gm = self._gm
        for k, lid in enumerate(self.background_ids):
            gm.change_p_load(lid, float(bp[k]))
            gm.change_q_load(lid, float(bq[k]))
        for k, lid in enumerate(self.household_ids):
            gm.change_p_load(lid, float(hp[k]))
            gm.change_q_load(lid, float(hq[k]))
        for k, sid in enumerate(self.pv_ids):
            gm.change_p_sgen(sid, float(pv[k]))
        gm.change_p_sgen(self.wind_id, wind)
        V = gm.ac_pf(self._V0.copy(), 30, 1e-7)
        if V.shape[0] == 0:  # diverged
            return self._diverged()
        vm = np.abs(V)
        _, _, _, a_line = gm.get_line_res1()
        p_hv, q_hv, _, _ = gm.get_trafo_res1()
        line_loading = np.asarray(a_line) / self._line_rating
        trafo_loading = np.hypot(p_hv, q_hv) / self._trafo_rating
        gen_p, _, _ = gm.get_gen_res()
        total_load = float(hp.sum() + bp.sum())
        gen_total = float(np.sum(gen_p))
        ren_total = float(pv.sum() + wind)
        return PFResult(
            True,
            vm,
            line_loading,
            trafo_loading,
            slack_p_mw=gen_total,
            losses_mw=gen_total + ren_total - total_load,
            ren_p_mw=ren_total,
            ext_grid_p_mw=float(gen_p[-1]),  # LightSim appends the slack after the gens
            gen_p_mw=tuple(float(x) for x in gen_p),
        )

    def _solve_pandapower(self, hp, hq, bp, bq, pv, wind) -> PFResult:
        import pandapower as pp

        net = self.net
        net.load.loc[self.background_ids, "p_mw"] = bp
        net.load.loc[self.background_ids, "q_mvar"] = bq
        net.load.loc[self.household_ids, "p_mw"] = hp
        net.load.loc[self.household_ids, "q_mvar"] = hq
        net.sgen.loc[self.pv_ids, "p_mw"] = pv
        net.sgen.loc[self.wind_id, "p_mw"] = wind
        try:
            pp.runpp(net, numba=False, init="results")
        except Exception:
            return self._diverged()
        if not net.converged:
            return self._diverged()
        gen_total = float(net.res_ext_grid.p_mw.sum() + net.res_gen.p_mw.sum())
        return PFResult(
            True,
            net.res_bus.vm_pu.values.copy(),
            net.res_line.loading_percent.values / 100.0,
            np.hypot(net.res_trafo.p_hv_mw.values, net.res_trafo.q_hv_mvar.values)
            / self._trafo_rating,
            slack_p_mw=gen_total,
            losses_mw=float(net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()),
            ren_p_mw=float(pv.sum() + wind),
            ext_grid_p_mw=float(net.res_ext_grid.p_mw.sum()),
            gen_p_mw=tuple(float(x) for x in net.res_gen.p_mw.values)
            + (float(net.res_ext_grid.p_mw.sum()),),
            n_iter=int(net._ppc.get("iterations", -1)) if getattr(net, "_ppc", None) else -1,
        )

    def _diverged(self) -> PFResult:
        return PFResult(
            False,
            np.zeros(self.n_bus),
            np.full(len(self._line_rating), 2.0),
            np.full(len(self._trafo_rating), 2.0),
            0.0,
            0.0,
        )


__all__ = [
    "IEEE14_LOAD_BUS_PRIORITY",
    "load_ieee14",
    "run_power_flow",
    "power_balance",
    "household_bus_map",
    "attach_households",
    "attach_renewables",
    "WIND_BUS",
    "set_thermal_limits",
    "background_shape",
    "PFResult",
    "Ieee14PowerFlow",
]
