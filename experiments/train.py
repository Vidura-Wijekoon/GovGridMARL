"""Hydra training entrypoint (tasks m3-t4 / m3-t5).

Usage
-----
    python experiments/train.py                                # MAPPO on IEEE-14
    python experiments/train.py agent=ippo seed=1              # swap learner
    python experiments/train.py +experiment=baseline_cloud     # short CPU profile
    python experiments/train.py -m agent=ppo,ippo,mappo seed=0,1,2,3,4   # sweep (m3-t4/m3-t9)
    python experiments/train.py tracking.backend=none          # no MLflow

Outputs land in ``results/runs/<agent>/seed<k>/`` (metrics.csv, eval.csv,
summary.json, checkpoints/) regardless of Hydra's per-run working directory,
so the evaluation script (``experiments/evaluate.py``) can find them.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT / "src", REPO_ROOT / "envs"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
warnings.filterwarnings("ignore", category=UserWarning)

import hydra  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402


def build_env_fn(cfg: DictConfig):
    from govgrid_envs import GridMARLEnv
    from govgridmarl.data import load_dataset

    e = OmegaConf.to_container(cfg.env, resolve=True)
    e.pop("_target_", None)
    e.pop("name", None)
    data = e.pop("data", {}) or {}
    rw = OmegaConf.to_container(cfg.reward, resolve=True).get("weights", {})
    n, h, seed = int(e["n_households"]), int(e.get("horizon", 24)), int(cfg.seed)
    dataset = None
    if data.get("source", "synthetic") != "synthetic":
        dataset = load_dataset(data["source"], n, h, seed, data.get("path"))

    def _fn():
        return GridMARLEnv(seed=seed, reward_weights=rw, dataset=dataset, **e)

    return _fn


def run(cfg: DictConfig) -> dict:
    from govgridmarl.agents import make_agent
    from govgridmarl.training import Trainer
    from govgridmarl.utils import Tracker, flatten, set_seed
    from govgridmarl.utils.logging import get_logger

    log = get_logger()
    set_seed(int(cfg.seed))
    env_fn = build_env_fn(cfg)
    probe = env_fn()
    probe.reset(seed=0)
    obs_dim = probe.observation_space(probe.possible_agents[0]).shape[0]
    n_agents, n_actions, state_dim = len(probe.possible_agents), probe.n_levels, probe.state_dim
    probe.close()

    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True)
    name = agent_cfg.pop("name")
    agent_cfg.pop("_target_", None)
    device = "cpu" if cfg.device in ("auto", "cpu") else str(cfg.device)
    if cfg.device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = make_agent(name, obs_dim, state_dim, n_agents, n_actions, cfg=agent_cfg, device=device)

    run_dir = REPO_ROOT / str(cfg.get("run_root", "results/runs")) / name / f"seed{cfg.seed}"
    if cfg.get("run_tag"):
        run_dir = run_dir.parent / f"{run_dir.name}_{cfg.run_tag}"
    if run_dir.exists():
        import shutil

        shutil.rmtree(run_dir)
    tracker = Tracker.from_config(cfg)
    tracker.log_params({**flatten(cfg), "run_dir": str(run_dir), "device": device})
    tr = OmegaConf.to_container(cfg.train, resolve=True)
    tr.pop("seeds", None)
    tr.pop("batch_size", None)
    trainer = Trainer(env_fn, agent, run_dir, seed=int(cfg.seed), tracker=tracker, **tr)
    log.info(
        "agent=%s device=%s obs_dim=%d state_dim=%d n_agents=%d run_dir=%s",
        name,
        device,
        obs_dim,
        state_dim,
        n_agents,
        run_dir,
    )
    try:
        summary = trainer.fit()
        OmegaConf.save(cfg, run_dir / "config.yaml")
        tracker.finish()
        return summary
    except Exception:
        tracker.finish(status="FAILED")
        raise


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    main()
