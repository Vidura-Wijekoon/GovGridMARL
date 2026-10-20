"""Loaders for residential load datasets: REDD and UK-DALE (task m2-t9).

Both loaders produce the same :class:`~govgridmarl.data.synthetic.LoadDataset`
structure as the synthetic generator so the environment can be switched
between sources with a config flag (``env.data.source``).

Access (see ``docs/data/datasets.md`` for the full procedure)
------------------------------------------------------------
* UK-DALE: UKERC Energy Data Centre, CC-BY 4.0. Download the *6-second
  house aggregate* channels (``house_<n>/channel_1.dat``) or the HDF5
  release; point ``load_ukdale`` at the extracted ``ukdale`` directory.
* REDD: request access from the MIT REDD maintainers (research licence,
  no redistribution). Use the low-frequency release; point ``load_redd`` at
  the ``low_freq`` directory.

Neither dataset is redistributed in this repository; ``data/raw`` is
git-ignored. When a path does not exist the loaders raise
:class:`FileNotFoundError` with the access instructions, so an experiment
never silently falls back to synthetic data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from govgridmarl.data.synthetic import (
    GROUP_PARAMS,
    GROUPS,
    LoadDataset,
    assign_groups,
    default_price,
    default_pv,
    tag_vulnerability,
)

_ACCESS_HINT = "See docs/data/datasets.md for download / licence steps (task m2-t9)."


def _read_dat(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a UK-DALE / REDD ``channel_<k>.dat`` (unix_ts, watts) file."""
    arr = np.loadtxt(path, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr[:, 0], arr[:, 1]


def _daily_profile(ts: np.ndarray, watts: np.ndarray, horizon: int) -> np.ndarray:
    """Average a (timestamp, W) series into a mean day of ``horizon`` bins (kW)."""
    bins = ((ts % 86400) / 86400.0 * horizon).astype(int) % horizon
    sums = np.bincount(bins, weights=watts, minlength=horizon)
    counts = np.bincount(bins, minlength=horizon).astype(np.float64)
    counts[counts == 0] = np.nan
    prof = sums / counts / 1000.0
    # fill empty bins by interpolation
    if np.isnan(prof).any():
        idx = np.arange(horizon)
        ok = ~np.isnan(prof)
        prof = np.interp(idx, idx[ok], prof[ok])
    return prof


def _build_dataset(
    profiles: list[np.ndarray], horizon: int, seed: int, source: str, flex_share: float = 0.25
) -> LoadDataset:
    rng = np.random.default_rng(seed)
    base = np.stack(profiles)
    n = base.shape[0]
    # Group labels are not observable in the public data; assign by daily
    # energy tercile (low energy -> "low"), matching the synthetic convention.
    energy = base.sum(axis=1)
    order = np.argsort(energy)
    groups = [""] * n
    for rank, i in enumerate(order):
        groups[i] = GROUPS[min(2, rank * 3 // max(1, n))]
    if n < 3:  # degenerate: fall back to random assignment
        groups = assign_groups(n, rng)
    flex = base * flex_share
    vulnerable = tag_vulnerability(groups, np.random.default_rng(seed + 7))
    meta = [
        {
            "household": i,
            "group": groups[i],
            "daily_kwh": float(energy[i] * 24.0 / horizon),
            "comfort_tol": GROUP_PARAMS[groups[i]]["comfort_tol"],
            "vulnerable": vulnerable[i],
        }
        for i in range(n)
    ]
    return LoadDataset(
        base_kw=base,
        flexible_kw=flex,
        groups=groups,
        price=default_price(horizon),
        pv_kw=default_pv(horizon),
        dt_hours=24.0 / horizon,
        meta=meta,
        source=source,
    )


def load_ukdale(path: str | Path, horizon: int = 24, seed: int = 42) -> LoadDataset:
    """Load UK-DALE house aggregates (channel_1) into mean daily profiles."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"UK-DALE not found at {root}. {_ACCESS_HINT}")
    houses = sorted(root.glob("house_*/channel_1.dat"))
    if not houses:
        raise FileNotFoundError(f"No house_*/channel_1.dat under {root}. {_ACCESS_HINT}")
    profiles = [_daily_profile(*_read_dat(h), horizon) for h in houses]
    return _build_dataset(profiles, horizon, seed, source=f"ukdale({len(profiles)} houses)")


def load_redd(path: str | Path, horizon: int = 24, seed: int = 42) -> LoadDataset:
    """Load REDD low-frequency mains (channel_1 + channel_2) into daily profiles."""
    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"REDD not found at {root}. {_ACCESS_HINT}")
    profiles = []
    for house in sorted(root.glob("house_*")):
        mains = [house / "channel_1.dat", house / "channel_2.dat"]
        series = [_read_dat(m) for m in mains if m.exists()]
        if not series:
            continue
        ts = series[0][0]
        watts = sum(np.interp(ts, s[0], s[1]) for s in series)
        profiles.append(_daily_profile(ts, watts, horizon))
    if not profiles:
        raise FileNotFoundError(f"No REDD houses with mains channels under {root}. {_ACCESS_HINT}")
    return _build_dataset(profiles, horizon, seed, source=f"redd({len(profiles)} houses)")


def load_dataset(source: str, n_households: int, horizon: int, seed: int, path=None) -> LoadDataset:
    """Dispatch on ``source`` in {synthetic, ukdale, redd}; tile/trim to n_households."""
    from govgridmarl.data.synthetic import generate_profiles

    if source == "synthetic":
        return generate_profiles(n_households, horizon, seed)
    ds = load_ukdale(path, horizon, seed) if source == "ukdale" else load_redd(path, horizon, seed)
    if ds.n_households == n_households:
        return ds
    # Real sets have few houses (UK-DALE 5, REDD 6): resample with jitter.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, ds.n_households, size=n_households)
    jit = np.exp(rng.normal(0, 0.15, size=(n_households, 1)))
    return LoadDataset(
        base_kw=ds.base_kw[idx] * jit,
        flexible_kw=ds.flexible_kw[idx] * jit,
        groups=[ds.groups[i] for i in idx],
        price=ds.price,
        pv_kw=ds.pv_kw,
        dt_hours=ds.dt_hours,
        meta=[dict(ds.meta[i], household=k) for k, i in enumerate(idx)],
        source=f"{ds.source} resampled to {n_households}",
    )


__all__ = ["load_ukdale", "load_redd", "load_dataset"]
