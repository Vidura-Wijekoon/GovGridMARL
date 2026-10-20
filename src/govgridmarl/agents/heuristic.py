"""Rule-based demand-response heuristic (benchmark baseline, task m3-t8).

The industry-style reference every learner must beat: a *price-and-peak
triggered* curtailment rule, the kind used in critical-peak-pricing and
direct-load-control programmes.

    level_t = high   if price_t >= peak_price or agg_load_t >= peak_load_frac * P_ref
            = medium if price_t >= mid_price
            = 0      otherwise

with a per-household daily budget: once a household's cumulative
curtailment reaches ``budget`` of its flexible energy it stops curtailing
(a crude comfort guard). Everyone gets the same rule, which is why it
scores well on Jain but leaves peak reduction on the table.
"""

from __future__ import annotations

import numpy as np

from govgridmarl.agents.base import BaseAgent


class RuleBasedDR(BaseAgent):
    """Threshold rule on the price / aggregate-load observation channels."""

    mode = "heuristic"
    team_reward = False

    def __init__(
        self,
        n_levels: int = 5,
        high_level: int | None = None,
        mid_level: int = 1,
        peak_price: float = 0.40,
        mid_price: float = 0.26,
        peak_load_frac: float = 0.85,
        budget: float = 0.35,
        **_: object,
    ) -> None:
        self.n_levels = int(n_levels)
        self.high = self.n_levels - 1 if high_level is None else int(high_level)
        self.mid = int(mid_level)
        self.peak_price, self.mid_price = float(peak_price), float(mid_price)
        self.peak_load_frac, self.budget = float(peak_load_frac), float(budget)

    # obs layout: see GridMARLEnv docstring (price_now = idx 5, agg = idx 10, burden = idx 11)
    def act(self, observation, state=None, deterministic: bool = True, pref=None):
        obs = np.asarray(observation)
        squeeze = obs.ndim == 2
        if squeeze:
            obs = obs[None]
        price = obs[..., 5] * 0.4
        agg = obs[..., 10]
        burden = obs[..., 11]
        act = np.zeros(obs.shape[:-1], dtype=np.int64)
        act[(price >= self.mid_price - 1e-9)] = self.mid
        act[(price >= self.peak_price - 1e-9) | (agg >= self.peak_load_frac)] = self.high
        act[burden >= self.budget] = 0
        zeros = np.zeros_like(act, dtype=np.float32)
        out = (act, zeros, zeros)
        return tuple(o[0] for o in out) if squeeze else out

    def update(self, batch):  # pragma: no cover - not a learner
        return {}


class NoControl(BaseAgent):
    """Always level 0 — the no-DR reference used for peak-reduction %."""

    mode = "none"
    team_reward = False

    def act(self, observation, state=None, deterministic: bool = True, pref=None):
        obs = np.asarray(observation)
        shape = obs.shape[:-1]
        z = np.zeros(shape, dtype=np.int64)
        return z, np.zeros(shape, np.float32), np.zeros(shape, np.float32)

    def update(self, batch):  # pragma: no cover
        return {}


__all__ = ["RuleBasedDR", "NoControl"]
