"""MOPPO - preference-conditioned multi-objective PPO (Phase 2, task m4-t8).

One network learns a *family* of Pareto policies instead of one policy per
weight vector (the weighted-sum sweep of m4-t5). The design follows the
"envelope / preference-conditioned" line of multi-objective RL (Yang et al.,
2019; Abels et al., 2019) applied to clipped PPO with a centralised critic:

* objectives ``J = (efficiency, fairness)`` - the environment's unweighted
  components (``info["objectives"]``, with the fixed-weight comfort term
  folded into efficiency so that lambda = 0 is exactly the Phase-1 MAPPO);
* preference ``lambda >= 0`` is the fairness weight; one value is drawn per
  episode and per environment from ``Uniform(0, lambda_max)`` (plus the grid
  points of the evaluation lambda set with a small probability, so the
  policy sees the exact weights it will be evaluated at);
* the actor and the critic receive ``lambda / lambda_max`` as an extra input;
  the critic is vector-valued (one head per objective) so that advantages
  are estimated per objective and scalarised afterwards,
  ``A = A_eff + lambda * A_fair`` - the standard MO-PPO update.

At evaluation the preference is fixed; sweeping it over ``LAMBDA_GRID``
traces the front with a single trained model (m4-t9 compares the coverage
of that front with the six separately trained weighted-sum policies).
"""

from __future__ import annotations

import numpy as np
import torch

from govgridmarl.agents.ppo_core import PPOFamilyAgent
from govgridmarl.agents.scalarization import LAMBDA_GRID, OBJECTIVES


class MOPPOAgent(PPOFamilyAgent):
    """Preference-conditioned multi-objective MAPPO (CTDE; m4-t8)."""

    mode = "moppo"
    team_reward = False
    pref_dim = 1
    n_values = len(OBJECTIVES)
    n_objectives = len(OBJECTIVES)
    objective_names = OBJECTIVES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        ex = self.cfg.extra
        self.lambda_max = float(ex.get("lambda_max", max(LAMBDA_GRID)))
        self.eval_lambda = float(ex.get("eval_lambda", 2.0))
        self.grid_prob = float(ex.get("grid_prob", 0.3))
        self.lambda_grid = tuple(float(x) for x in ex.get("lambda_grid", LAMBDA_GRID))
        self._rng = np.random.default_rng(int(ex.get("pref_seed", 0)))

    # ------------------------------------------------------------ prefs
    def sample_lambda(self, n: int) -> np.ndarray:
        """Fairness weights for ``n`` new episodes."""
        u = self._rng.uniform(0.0, self.lambda_max, size=n)
        g = self._rng.choice(self.lambda_grid, size=n)
        use_grid = self._rng.random(n) < self.grid_prob
        return np.where(use_grid, g, u).astype(np.float32)

    def pref_vector(self, lam) -> np.ndarray:
        """Network input for preferences: ``lambda / lambda_max`` as (n, 1)."""
        lam = np.asarray(lam, dtype=np.float32).reshape(-1)
        return (lam / max(self.lambda_max, 1e-9)).reshape(-1, 1).astype(np.float32)

    def default_pref(self, B: int) -> torch.Tensor:
        return torch.full((B, 1), self.eval_lambda / max(self.lambda_max, 1e-9), device=self.device)

    # ------------------------------------------------------------ objectives
    @staticmethod
    def objective_vector(obj: dict, w: dict) -> np.ndarray:
        """(efficiency incl. fixed-weight comfort & safety, fairness) from ``info["objectives"]``."""
        eff = (
            float(obj["efficiency"])
            + float(w.get("comfort", 0.1)) * float(obj["comfort"])
            + float(w.get("safety", 0.0)) * float(obj["safety"])
        )
        return np.array([eff, float(obj["fairness"])], dtype=np.float32)

    @staticmethod
    def scalarise(vec, lam):
        """``J_eff + lambda * J_fair`` for arrays whose last axis is the objective axis."""
        v = np.asarray(vec, dtype=np.float32)
        lam = np.asarray(lam, dtype=np.float32)
        return v[..., 0] + lam * v[..., 1]

    def extra_state(self) -> dict:
        return {"lambda_max": self.lambda_max, "eval_lambda": self.eval_lambda}

    def load_extra_state(self, extra: dict) -> None:
        self.lambda_max = float(extra.get("lambda_max", self.lambda_max))
        self.eval_lambda = float(extra.get("eval_lambda", self.eval_lambda))


__all__ = ["MOPPOAgent"]
