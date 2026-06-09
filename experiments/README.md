# `experiments/`

Hydra-driven entrypoints and sweep definitions.

```bash
python experiments/train.py                          # MAPPO on IEEE-14 (defaults)
python experiments/train.py +experiment=baseline_ieee14
python experiments/train.py +experiment=fairness_safety
python experiments/evaluate.py                       # report eval-protocol metrics
```

Results are written to `results/<date>/<time>/` and logged to the configured
tracker (W&B / MLflow). See `configs/` for the full composition tree.
