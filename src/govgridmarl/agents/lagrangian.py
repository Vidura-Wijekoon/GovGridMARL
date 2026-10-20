"""Constrained RL: PPO-Lagrangian and PID-Lagrangian (Phase 2, tasks m5-t3 .. m5-t6).

Constrained objective (CMOMG, ``docs/design/cmomg_formulation.md`` section 4)::

    max_pi  J_R(pi)   s.t.   J_{C,k}(pi) = E[ sum_t c_k(s_t) ] <= d_k,   k = 1..K

solved as the saddle point of the Lagrangian
``L(pi, lambda) = J_R(pi) - sum_k lambda_k (J_{C,k}(pi) - d_k)`` by alternating

* a **policy step** - clipped PPO on the penalised advantage
  ``A = (A_R - sum_k lambda_k A_{C,k}) / (1 + sum_k lambda_k)``, where ``A_{C,k}``
  is a GAE estimate from a separate *cost critic* per constraint
  (Ray, Achiam & Amodei, 2019, "Benchmarking Safe Exploration"); the
  ``1 + sum lambda`` normalisation keeps the effective step size bounded;
* a **multiplier step** on the observed episode costs ``J_C`` of the batch:

  - ``lagrangian`` (m5-t3): dual ascent ``lambda <- [lambda + eta (J_C - d)]_+``;
  - ``pid_lagrangian`` (m5-t6, Stooke, Achiam & Abbeel, 2020): the
    multiplier is the output of a PID controller on the constraint error,
    ``lambda = [K_p e + K_i I + K_d (e - e_prev)]_+`` with a clamped integral
    ``I <- [I + e]_+``. The proportional and derivative terms damp the
    oscillation / overshoot that pure dual ascent (integral-only control)
    shows when the learning rate is large enough to react within a run
    (m5-t5 sweep).

Two agents share the implementation and differ only in the input regime
(``ppo_core``): :class:`PPOLagrangianAgent` (centralised controller, m5-t3)
and :class:`MAPPOLagrangianAgent` (CTDE, cost critics on the global state,
m5-t4 - the thesis learner). Constraints are chosen by name from the
environment's ``info["cost_vec"]`` (``voltage``, ``loading``, ``comfort``,
``fairness``); the joint fairness + safety agent of m5-t10 is the same class
with all four names.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from govgridmarl.agents.ppo_core import PPOFamilyAgent
from govgridmarl.safety.constraints import SAFETY_CONSTRAINTS, ConstraintSet


class LagrangianController:
    """Updates Lagrange multipliers from observed episode costs (m5-t3 / m5-t6).

    Parameters
    ----------
    names, limits:
        Constraint names and budgets ``d_k`` (default 0: zero violations).
    method:
        ``"lagrangian"`` (dual ascent with rate ``lr``) or ``"pid_lagrangian"``
        (gains ``kp``, ``ki``, ``kd``; ``ki`` plays the role of ``lr``).
    max_multiplier:
        Upper clamp on every multiplier (keeps the penalised advantage finite).
    ema:
        Optional exponential smoothing of the cost estimate before the update
        (0 = none); reduces noise from the small number of episodes per batch.
    """

    def __init__(
        self,
        names: Sequence[str] = SAFETY_CONSTRAINTS,
        limits=None,
        method: str = "lagrangian",
        lr: float = 0.05,
        kp: float = 0.5,
        ki: float = 0.05,
        kd: float = 0.1,
        max_multiplier: float = 20.0,
        init: float = 0.0,
        ema: float = 0.0,
    ) -> None:
        if method not in ("lagrangian", "pid_lagrangian"):
            raise ValueError("method must be 'lagrangian' or 'pid_lagrangian'")
        self.set = ConstraintSet(names, limits)
        self.names = self.set.names
        self.limits = self.set.limits.astype(np.float64)
        self.method = method
        self.lr, self.kp, self.ki, self.kd = float(lr), float(kp), float(ki), float(kd)
        self.max_multiplier = float(max_multiplier)
        self.ema = float(ema)
        K = len(self.names)
        self.multipliers = np.full(K, float(init))
        self.integral = np.zeros(K)
        self.prev_err = np.zeros(K)
        self.cost_estimate = np.zeros(K)
        self.n_updates = 0
        self.history: list[dict[str, float]] = []

    def update(self, episode_costs) -> dict[str, float]:
        """One multiplier step from the mean episode costs ``J_C`` (K,) of a batch."""
        jc = np.asarray(episode_costs, dtype=np.float64).reshape(-1)
        if jc.shape[0] != len(self.names):
            raise ValueError(f"expected {len(self.names)} costs, got {jc.shape[0]}")
        if self.ema > 0 and self.n_updates > 0:
            self.cost_estimate = self.ema * self.cost_estimate + (1 - self.ema) * jc
        else:
            self.cost_estimate = jc.copy()
        err = self.cost_estimate - self.limits
        if self.method == "lagrangian":
            self.multipliers = np.clip(self.multipliers + self.lr * err, 0.0, self.max_multiplier)
        else:
            self.integral = np.clip(
                self.integral + err, 0.0, self.max_multiplier / max(self.ki, 1e-9)
            )
            deriv = err - self.prev_err
            self.multipliers = np.clip(
                self.kp * err + self.ki * self.integral + self.kd * deriv, 0.0, self.max_multiplier
            )
        self.prev_err = err
        self.n_updates += 1
        rec = {f"lambda_{n}": float(v) for n, v in zip(self.names, self.multipliers)}
        rec.update({f"cost_{n}": float(v) for n, v in zip(self.names, jc)})
        self.history.append(rec)
        return rec

    def state_dict(self) -> dict[str, Any]:
        return {
            "names": list(self.names),
            "limits": self.limits.tolist(),
            "method": self.method,
            "multipliers": self.multipliers.tolist(),
            "integral": self.integral.tolist(),
            "prev_err": self.prev_err.tolist(),
            "cost_estimate": self.cost_estimate.tolist(),
            "n_updates": self.n_updates,
        }

    def load_state_dict(self, d: dict[str, Any]) -> None:
        self.multipliers = np.asarray(d.get("multipliers", self.multipliers), dtype=float)
        self.integral = np.asarray(d.get("integral", self.integral), dtype=float)
        self.prev_err = np.asarray(d.get("prev_err", self.prev_err), dtype=float)
        self.cost_estimate = np.asarray(d.get("cost_estimate", self.cost_estimate), dtype=float)
        self.n_updates = int(d.get("n_updates", self.n_updates))


class LagrangianPPOFamilyAgent(PPOFamilyAgent):
    """PPO-family learner with cost critics and a Lagrangian multiplier controller.

    Config keys (``cfg.agent`` extras): ``constraints`` (names), ``limits``
    (dict), ``method`` (``lagrangian`` | ``pid_lagrangian``), ``multiplier_lr``,
    ``kp``/``ki``/``kd``, ``max_multiplier``, ``cost_value_coef``, ``multiplier_ema``,
    ``cost_signal`` (``indicator`` | ``magnitude``, see m5-t5: magnitude costs on
    IEEE-14 are O(0.03) per episode and leave dual ascent stalled; the indicator
    form counts violating hours and is the tuned default).
    """

    mode = "mappo_lagrangian"
    team_reward = False

    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("cfg") if "cfg" in kwargs else (args[4] if len(args) > 4 else None)
        from govgridmarl.agents.ppo_core import PPOConfig

        pc = cfg if isinstance(cfg, PPOConfig) else PPOConfig.from_mapping(cfg)
        ex = pc.extra
        names = tuple(ex.get("constraints", SAFETY_CONSTRAINTS))
        self.constraint_names = names
        self.n_costs = len(names)
        super().__init__(*args, **kwargs)
        ex = self.cfg.extra
        self.controller = LagrangianController(
            names,
            ex.get("limits"),
            method=str(ex.get("method", "lagrangian")),
            lr=float(ex.get("multiplier_lr", 0.05)),
            kp=float(ex.get("kp", 0.5)),
            ki=float(ex.get("ki", ex.get("multiplier_lr", 0.05))),
            kd=float(ex.get("kd", 0.1)),
            max_multiplier=float(ex.get("max_multiplier", 20.0)),
            init=float(ex.get("multiplier_init", 0.0)),
            ema=float(ex.get("multiplier_ema", 0.0)),
        )
        self.cost_value_coef = float(ex.get("cost_value_coef", 0.5))
        # which cost signal the trainer collects (m5-t5): "indicator" (violating steps,
        # 0..T per episode - the tuned default) or "magnitude" (band / overload / excess)
        self.cost_signal = str(ex.get("cost_signal", "indicator"))
        if self.cost_signal not in ("indicator", "magnitude"):
            raise ValueError("cost_signal must be 'indicator' or 'magnitude'")

    @property
    def multipliers(self) -> np.ndarray:
        return self.controller.multipliers

    # ------------------------------------------------------------ hooks
    def _shape_advantage(self, batch, adv: torch.Tensor) -> torch.Tensor:
        lam = torch.as_tensor(self.multipliers, dtype=torch.float32, device=self.device)
        cadv = torch.as_tensor(batch["cost_advantages"], dtype=torch.float32, device=self.device)
        penalty = (cadv * lam).sum(-1)  # (M, N)
        return (adv - penalty) / (1.0 + float(lam.sum()))

    def _extra_loss(self, batch, idx, obs, state, pref, stats) -> torch.Tensor:
        cret = torch.as_tensor(batch["cost_returns"], dtype=torch.float32, device=self.device)
        _, _, cv = self._forward(obs[idx], state[idx], pref if pref is None else pref[idx], True)
        loss = 0.5 * F.mse_loss(cv, cret[idx])
        stats["cost_value_loss"] = stats.get("cost_value_loss", 0.0) + float(loss.item())
        return self.cost_value_coef * loss

    def _post_update(self, batch) -> dict:
        if "ep_cost_mean" not in batch:
            return {}
        rec = self.controller.update(batch["ep_cost_mean"])
        return {k: v for k, v in rec.items() if k.startswith("lambda_")}

    def extra_state(self) -> dict:
        return {"controller": self.controller.state_dict()}

    def load_extra_state(self, extra: dict) -> None:
        if "controller" in extra:
            self.controller.load_state_dict(extra["controller"])


class MAPPOLagrangianAgent(LagrangianPPOFamilyAgent):
    """Constrained MAPPO: local actors, reward and cost critics on the global state (m5-t4)."""

    mode = "mappo_lagrangian"


class PPOLagrangianAgent(LagrangianPPOFamilyAgent):
    """Constrained centralised PPO controller (m5-t3 reference)."""

    mode = "ppo_lagrangian"
    team_reward = True


__all__ = [
    "LagrangianController",
    "LagrangianPPOFamilyAgent",
    "MAPPOLagrangianAgent",
    "PPOLagrangianAgent",
]
