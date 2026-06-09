"""Hydra training entrypoint.

Usage
-----
    python experiments/train.py                       # default: MAPPO on IEEE-14
    python experiments/train.py agent=ippo            # swap learner
    python experiments/train.py +experiment=fairness_safety
    python experiments/train.py -m agent=ippo,mappo seed=0,1,2  # sweep

The actual training loop is implemented in Month 3 (m3-t5); this wires up
config composition, seeding, and tracking so the interface is stable.
"""

from __future__ import annotations

try:
    import hydra
    from omegaconf import DictConfig, OmegaConf

    _HYDRA = True
except ImportError:  # skeleton stage: hydra not yet installed
    _HYDRA = False


def _run(cfg) -> None:
    from govgridmarl.utils import set_seed
    from govgridmarl.utils.logging import get_logger

    log = get_logger()
    set_seed(getattr(cfg, "seed", 42))
    log.info("Loaded config:\n%s", cfg)
    log.info("TODO(m3-t5): instantiate env + agent + Trainer and call .fit()")


if _HYDRA:

    @hydra.main(version_base=None, config_path="../configs", config_name="config")
    def main(cfg: DictConfig) -> None:  # noqa: F821
        print(OmegaConf.to_yaml(cfg))
        _run(cfg)

else:

    def main() -> None:
        print("Hydra not installed yet. Install deps (see environment.yml), then rerun.")


if __name__ == "__main__":
    main()
