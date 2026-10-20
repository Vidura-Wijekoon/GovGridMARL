"""Experiment tracking (task m1-t4).

One thin, backend-agnostic ``Tracker`` used by every training / evaluation
entrypoint. The backend is chosen in Hydra config (``tracking.backend``):

* ``mlflow`` (default) — local SQLite store at ``<repo>/mlflow.db``; view with
  ``mlflow ui`` (or ``make dashboard``). No cloud account needed.
* ``wandb``  — optional cloud dashboard; requires ``wandb login``.
* ``none``   — no-op, for debugging.

Design rules:
- Never crash a training run because tracking failed (warn + degrade to none).
- The MLflow store lives at the REPO ROOT regardless of Hydra's per-run
  working directory, so all runs land in one dashboard.

Typical use::

    tracker = Tracker.from_config(cfg)          # cfg = composed Hydra config
    tracker.log_params(flatten(cfg))
    tracker.log_metrics({"reward": r, "loss": l}, step=t)
    tracker.log_artifact("results/plots/curve.png")
    tracker.finish()
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from govgridmarl.utils.logging import get_logger

REPO_ROOT = Path(__file__).resolve().parents[3]
# MLflow >=3.13 deprecates the plain ./mlruns file store -> use SQLite.
DEFAULT_MLFLOW_URI = f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}"
DEFAULT_ARTIFACT_URI = (REPO_ROOT / "mlartifacts").as_uri()

_log = get_logger("govgridmarl.tracking")


def flatten(cfg: Any, prefix: str = "", max_len: int = 500) -> dict[str, Any]:
    """Flatten a (possibly nested) mapping / OmegaConf into dot-keyed params."""
    try:  # accept OmegaConf without importing it as a hard dependency
        from omegaconf import OmegaConf

        if OmegaConf.is_config(cfg):
            cfg = OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass

    out: dict[str, Any] = {}
    if isinstance(cfg, Mapping):
        for k, v in cfg.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, Mapping):
                out.update(flatten(v, key))
            else:
                out[key] = str(v)[:max_len]
    return out


class Tracker:
    """Unified facade over MLflow / W&B / no-op."""

    def __init__(
        self,
        backend: str = "mlflow",
        experiment: str = "govgridmarl",
        run_name: str | None = None,
        uri: str | None = None,
        project: str = "govgridmarl",
        entity: str | None = None,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        self.backend = backend
        self._mlflow = None
        self._wandb_run = None

        if backend == "none":
            return
        try:
            if backend == "mlflow":
                import mlflow

                mlflow.set_tracking_uri(
                    uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_MLFLOW_URI
                )
                if mlflow.get_experiment_by_name(experiment) is None:
                    mlflow.create_experiment(experiment, artifact_location=DEFAULT_ARTIFACT_URI)
                mlflow.set_experiment(experiment)
                mlflow.start_run(run_name=run_name, tags=dict(tags or {}))
                self._mlflow = mlflow
                _log.info(
                    "MLflow tracking -> %s (experiment=%s)",
                    mlflow.get_tracking_uri(),
                    experiment,
                )
            elif backend == "wandb":
                import wandb

                self._wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    tags=list((tags or {}).values()) or None,
                )
                _log.info("W&B tracking -> %s", self._wandb_run.url)
            else:
                raise ValueError(f"Unknown tracking backend: {backend!r}")
        except Exception as exc:  # noqa: BLE001 — tracking must never kill a run
            _log.warning("Tracking disabled (%s init failed: %s)", backend, exc)
            self.backend = "none"

    # ------------------------------------------------------------------ API
    @classmethod
    def from_config(cls, cfg: Any) -> Tracker:
        """Build a Tracker from the composed Hydra config (cfg.tracking.*)."""
        t = getattr(cfg, "tracking", None)
        if t is None:
            return cls(backend="none")
        get = lambda k, d=None: getattr(t, k, d)  # noqa: E731
        return cls(
            backend=get("backend", "mlflow"),
            experiment=get("experiment", "govgridmarl"),
            run_name=get("run_name"),
            uri=get("uri"),
            project=get("project", "govgridmarl"),
            entity=get("entity"),
        )

    def log_params(self, params: Mapping[str, Any]) -> None:
        if self._mlflow:
            # MLflow caps at 100 params per call; chunk to stay safe.
            items = list(params.items())
            for i in range(0, len(items), 90):
                self._mlflow.log_params(dict(items[i : i + 90]))
        elif self._wandb_run:
            self._wandb_run.config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        clean = {k: float(v) for k, v in metrics.items()}
        try:
            if self._mlflow:
                self._mlflow.log_metrics(clean, step=step)
            elif self._wandb_run:
                self._wandb_run.log(clean, step=step)
        except Exception as exc:  # noqa: BLE001 — tracking must never kill a run
            _log.warning("log_metrics failed (%s); continuing", exc)

    def log_artifact(self, path: str | Path) -> None:
        path = str(path)
        if self._mlflow:
            self._mlflow.log_artifact(path)
        elif self._wandb_run:
            import wandb

            wandb.save(path)

    def finish(self, status: str = "FINISHED") -> None:
        if self._mlflow:
            self._mlflow.end_run(status=status)
        elif self._wandb_run:
            self._wandb_run.finish()
