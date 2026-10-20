"""Synthetic residential load-profile generator (task m2-t10).

Why synthetic
-------------
REDD and UK-DALE (task m2-t9, see ``docs/data/datasets.md``) require manual,
licence-gated downloads and ship as multi-GB appliance-level recordings. For
a reproducible, licence-free environment we generate *household aggregate*
profiles whose daily energy, diurnal shape and dispersion are calibrated to
published statistics of those datasets and of the UK regulator's Typical
Domestic Consumption Values (TDCV). The real loaders in
:mod:`govgridmarl.data.loaders` produce the same tabular format, so the
environment is agnostic to the data source.

Calibration anchors (documented in ``docs/data/datasets.md``)
------------------------------------------------------------
* Daily energy: Ofgem TDCV "medium" electricity profile 2,700 kWh/yr
  (about 7.4 kWh/day); low/high profiles 1,800 / 4,100 kWh/yr.
* Diurnal shape: two-peak residential curve (morning 07-09 h, evening
  17-21 h) with a night trough around 0.25 of the daily mean, consistent
  with the UK-DALE house-level aggregate and the Elexon Profile Class 1
  shape.
* Household dispersion: log-normal scaling with sigma approximately 0.35,
  the order of the between-house spread reported for UK-DALE / REDD houses.
* Socio-economic groups (proposal section 6.2): low / mid / high income
  differ in energy level, flexible share (shiftable appliances) and comfort
  tolerance, following the energy-justice literature reviewed in Area 3.

Output format
-------------
:func:`generate_profiles` returns a :class:`LoadDataset` holding

* ``base_kw``      ndarray (n_households, horizon): inflexible + flexible load
* ``flexible_kw``  ndarray (n_households, horizon): the shiftable component
* ``groups``       list[str] of group labels per household
* ``meta``         per-household parameters (group, scale, flex_share, ...)
* ``price``        ndarray (horizon,): tariff signal in currency/kWh
* ``pv_kw``        ndarray (horizon,): community PV availability (per home)

All arrays are hourly by default (``dt_hours = 24 / horizon``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Group definitions (proposal section 6.2; energy-justice literature, Area 3)  #
# --------------------------------------------------------------------------- #
GROUPS: tuple[str, ...] = ("low", "mid", "high")

GROUP_PARAMS: dict[str, dict[str, float]] = {
    # daily_kwh: TDCV low/medium/high ~ 1800/2700/4100 kWh per year
    # flex_share: share of daily energy that is shiftable/curtailable
    # comfort_tol: max daily curtailment fraction of flexible load before
    #              disutility becomes unacceptable (comfort bound)
    # evening_weight: how peaked the evening is relative to morning
    "low": {"daily_kwh": 4.9, "flex_share": 0.15, "comfort_tol": 0.30, "evening_weight": 1.15},
    "mid": {"daily_kwh": 7.4, "flex_share": 0.25, "comfort_tol": 0.40, "evening_weight": 1.00},
    "high": {"daily_kwh": 11.2, "flex_share": 0.35, "comfort_tol": 0.50, "evening_weight": 0.90},
}

# Default split of households into groups (roughly UK income terciles).
DEFAULT_GROUP_SHARES: dict[str, float] = {"low": 0.34, "mid": 0.33, "high": 0.33}

# Vulnerability tagging (task m4-t2). A cluster is *vulnerable* when it is
# fuel-poverty exposed (all low-income clusters) or carries a medical /
# elderly dependency (a ``VULNERABLE_MID_SHARE`` share of the mid group).
# High-income clusters are never tagged. The flag is used only for the
# equity analysis and the fairness term; it never changes the physics.
VULNERABLE_GROUPS: tuple[str, ...] = ("low",)
VULNERABLE_MID_SHARE: float = 0.20


@dataclass
class LoadDataset:
    """Container for a set of household load profiles (see module docstring)."""

    base_kw: np.ndarray
    flexible_kw: np.ndarray
    groups: list[str]
    price: np.ndarray
    pv_kw: np.ndarray
    dt_hours: float
    meta: list[dict[str, Any]] = field(default_factory=list)
    source: str = "synthetic"

    @property
    def n_households(self) -> int:
        return int(self.base_kw.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.base_kw.shape[1])

    @property
    def inflexible_kw(self) -> np.ndarray:
        return self.base_kw - self.flexible_kw

    @property
    def vulnerable(self) -> list[bool]:
        """Vulnerability flags (m4-t2); low-income by default when meta carries none."""
        return [bool(m.get("vulnerable", m.get("group") in VULNERABLE_GROUPS)) for m in self.meta]

    def daily_energy_kwh(self) -> np.ndarray:
        return self.base_kw.sum(axis=1) * self.dt_hours

    def summary(self) -> dict[str, Any]:
        e = self.daily_energy_kwh()
        agg = self.base_kw.sum(axis=0)
        return {
            "source": self.source,
            "n_households": self.n_households,
            "horizon": self.horizon,
            "dt_hours": self.dt_hours,
            "daily_kwh_mean": float(e.mean()),
            "daily_kwh_min": float(e.min()),
            "daily_kwh_max": float(e.max()),
            "aggregate_peak_kw": float(agg.max()),
            "aggregate_peak_hour": int(agg.argmax() * self.dt_hours),
            "load_factor": float(agg.mean() / agg.max()),
            "flex_share_mean": float(self.flexible_kw.sum() / self.base_kw.sum()),
            "groups": {g: int(sum(1 for x in self.groups if x == g)) for g in GROUPS},
            "n_vulnerable": int(sum(self.vulnerable)),
        }


# --------------------------------------------------------------------------- #
# Shape primitives                                                             #
# --------------------------------------------------------------------------- #
def _gauss(t: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def diurnal_shape(horizon: int = 24, evening_weight: float = 1.0) -> np.ndarray:
    """Normalised two-peak residential shape (mean = 1 over the day).

    Base 0.25 (night trough), morning peak centred 07:45, evening peak centred
    18:30 with a longer tail; evening/morning ratio set by ``evening_weight``.
    """
    t = (np.arange(horizon) + 0.5) * (24.0 / horizon)
    s = 0.25 + 0.55 * _gauss(t, 7.75, 1.1) + 1.25 * evening_weight * _gauss(t, 18.5, 1.9)
    s += 0.15 * _gauss(t, 13.0, 2.5)  # small midday shoulder
    return s / s.mean()


def default_price(horizon: int = 24) -> np.ndarray:
    """Three-tier time-of-use tariff (currency/kWh), Economy-7-like shape.

    Off-peak 00-07 h, standard otherwise, peak 16-20 h. Values are indicative
    of UK ToU tariffs (e.g. ~0.12 / 0.26 / 0.40 per kWh).
    """
    t = (np.arange(horizon) + 0.5) * (24.0 / horizon)
    p = np.full(horizon, 0.26)
    p[t < 7] = 0.12
    p[(t >= 16) & (t < 20)] = 0.40
    return p


def default_pv(horizon: int = 24, peak_kw_per_home: float = 0.8) -> np.ndarray:
    """Clear-sky-like PV availability per home (kW), bell centred at 12:30."""
    t = (np.arange(horizon) + 0.5) * (24.0 / horizon)
    return peak_kw_per_home * np.clip(_gauss(t, 12.5, 2.6) - 0.05, 0.0, None)


def tag_vulnerability(
    groups: list[str], rng: np.random.Generator, mid_share: float = VULNERABLE_MID_SHARE
) -> list[bool]:
    """Vulnerability flags per household (m4-t2); deterministic given ``rng``."""
    out = []
    for g in groups:
        if g in VULNERABLE_GROUPS:
            out.append(True)
        elif g == "mid":
            out.append(bool(rng.random() < mid_share))
        else:
            out.append(False)
    return out


def assign_groups(n_households: int, rng: np.random.Generator, shares=None) -> list[str]:
    """Deterministically assign households to income groups by target shares."""
    shares = dict(shares or DEFAULT_GROUP_SHARES)
    counts = {g: int(round(shares[g] * n_households)) for g in GROUPS}
    # fix rounding so counts sum to n
    diff = n_households - sum(counts.values())
    order = sorted(GROUPS, key=lambda g: -shares[g])
    i = 0
    while diff != 0:
        counts[order[i % len(order)]] += 1 if diff > 0 else -1
        diff += -1 if diff > 0 else 1
        i += 1
    labels = [g for g in GROUPS for _ in range(counts[g])]
    rng.shuffle(labels)
    return labels


# --------------------------------------------------------------------------- #
# Main generator                                                               #
# --------------------------------------------------------------------------- #
def generate_profiles(
    n_households: int,
    horizon: int = 24,
    seed: int = 42,
    group_shares: dict[str, float] | None = None,
    noise_sigma: float = 0.08,
    household_sigma: float = 0.35,
    day_scale: float = 1.0,
) -> LoadDataset:
    """Generate calibrated synthetic household load profiles (task m2-t10).

    Parameters
    ----------
    n_households:
        Number of household agents.
    horizon:
        Steps per day (24 = hourly, 48 = half-hourly).
    seed:
        RNG seed; the same seed always yields the same dataset.
    group_shares:
        Fraction of households per income group (defaults to terciles).
    noise_sigma:
        Multiplicative step-wise noise (log-normal sigma) to avoid perfectly
        smooth curves.
    household_sigma:
        Between-household log-normal dispersion of daily energy.
    day_scale:
        Global multiplier (e.g. 1.15 for a cold day).

    Returns
    -------
    LoadDataset
    """
    if n_households <= 0:
        raise ValueError("n_households must be positive")
    if horizon <= 0 or (24 % horizon != 0 and horizon % 24 != 0):
        # accept 24, 48, 96 (or divisors like 12) — anything else is suspicious
        raise ValueError("horizon should divide or be a multiple of 24")
    rng = np.random.default_rng(seed)
    dt = 24.0 / horizon
    groups = assign_groups(n_households, rng, group_shares)
    vulnerable = tag_vulnerability(groups, np.random.default_rng(seed + 7))

    base = np.zeros((n_households, horizon), dtype=np.float64)
    flex = np.zeros_like(base)
    meta: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        p = GROUP_PARAMS[g]
        scale = float(np.exp(rng.normal(0.0, household_sigma) - 0.5 * household_sigma**2))
        daily_kwh = p["daily_kwh"] * scale * day_scale
        # jitter peak timing per household (+-45 min) by rolling the shape
        shape = diurnal_shape(horizon, p["evening_weight"])
        shift = int(round(rng.normal(0.0, 0.75) / dt))
        shape = np.roll(shape, shift)
        noise = np.exp(rng.normal(0.0, noise_sigma, size=horizon))
        kw = daily_kwh / 24.0 * shape * noise  # mean kW * shape
        # flexible component: concentrated in evening/morning (appliances,
        # EV, heat pump), represented as share of load with a smoother shape
        flex_shape = 0.5 + 0.5 * _gauss((np.arange(horizon) + 0.5) * dt, 18.0, 3.0)
        flex_shape = flex_shape / flex_shape.mean()
        flex_kw = np.minimum(kw * 0.9, kw * p["flex_share"] * flex_shape)
        base[i] = kw
        flex[i] = flex_kw
        meta.append(
            {
                "household": i,
                "group": g,
                "scale": scale,
                "daily_kwh": float(kw.sum() * dt),
                "flex_share": p["flex_share"],
                "comfort_tol": p["comfort_tol"],
                "vulnerable": vulnerable[i],
            }
        )

    return LoadDataset(
        base_kw=base,
        flexible_kw=flex,
        groups=groups,
        price=default_price(horizon),
        pv_kw=default_pv(horizon),
        dt_hours=dt,
        meta=meta,
        source=f"synthetic(seed={seed})",
    )


__all__ = [
    "GROUPS",
    "GROUP_PARAMS",
    "DEFAULT_GROUP_SHARES",
    "VULNERABLE_GROUPS",
    "VULNERABLE_MID_SHARE",
    "tag_vulnerability",
    "LoadDataset",
    "generate_profiles",
    "diurnal_shape",
    "default_price",
    "default_pv",
    "assign_groups",
]
