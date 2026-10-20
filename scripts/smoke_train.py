#!/usr/bin/env python3
"""GPU smoke training job (task m1-t3).

Trains a small actor-critic style MLP on a synthetic grid-control task
(random "bus states" -> setpoint regression) for a few hundred steps.
This is NOT a real experiment — it only proves the training loop, CUDA
kernels, backprop, and checkpointing all work on the provisioned compute.

The full stdout is mirrored to results/logs/smoke_train_<timestamp>.log,
which serves as the m1-t3 deliverable ("working GPU job log").

Usage:
    python scripts/smoke_train.py                 # auto: cuda if available
    python scripts/smoke_train.py --device cpu    # force CPU
    python scripts/smoke_train.py --steps 500
    python scripts/smoke_train.py --track         # also log to MLflow (m1-t4)
"""

from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = REPO_ROOT / "results" / "logs"
CKPT_DIR = REPO_ROOT / "results" / "checkpoints"

N_AGENTS = 4  # mimic 4 grid agents (e.g. feeder controllers)
OBS_DIM = 28  # ~ IEEE-14: 14 buses x (V, theta)
ACT_DIM = 5  # discrete-ish setpoint head per agent


class Tee:
    """Write to stdout and a log file simultaneously."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self._fh.write(msg + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def build_model(torch):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(OBS_DIM, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, ACT_DIM),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--track",
        action="store_true",
        help="Log this run to the experiment tracker (MLflow, task m1-t4).",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = Tee(LOG_DIR / f"smoke_train_{stamp}.log")

    log("=" * 70)
    log("GovGridMARL smoke training job (task m1-t3)")
    log(f"UTC time   : {datetime.now(timezone.utc).isoformat()}")
    log(f"Host       : {platform.node()} ({platform.system()} {platform.release()})")
    log(f"Python     : {sys.version.split()[0]}")
    log(f"Args       : steps={args.steps} batch={args.batch_size} seed={args.seed}")
    log("=" * 70)

    try:
        import torch
    except ImportError as exc:
        log(f"FAIL: PyTorch is not installed ({exc}).")
        log.close()
        return 1

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log("FAIL: --device cuda requested but CUDA is not available.")
        log.close()
        return 1

    log(f"torch      : {torch.__version__} | CUDA built: {torch.version.cuda}")
    if device == "cuda":
        log(f"GPU        : {torch.cuda.get_device_name(0)}")
    log(f"Device     : {device}")
    log("-" * 70)

    tracker = None
    if args.track:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from govgridmarl.utils.tracking import Tracker

        tracker = Tracker(
            backend="mlflow", experiment="govgridmarl-smoke", run_name=f"smoke_{stamp}"
        )
        tracker.log_params(
            {
                "device": device,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seed": args.seed,
                "n_agents": N_AGENTS,
                "torch": torch.__version__,
            }
        )

    torch.manual_seed(args.seed)

    # One tiny policy per "agent", trained jointly — enough to exercise
    # multi-network optimisation the way the real MARL loop will.
    models = [build_model(torch).to(device) for _ in range(N_AGENTS)]
    params = [p for m in models for p in m.parameters()]
    opt = torch.optim.Adam(params, lr=3e-4)
    loss_fn = torch.nn.MSELoss()

    # Fixed random "ground truth" mapping to make the loss meaningfully decrease.
    targets_w = torch.randn(OBS_DIM, ACT_DIM, device=device) / OBS_DIM**0.5

    first_loss = last_loss = None
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        obs = torch.randn(args.batch_size, OBS_DIM, device=device)
        target = obs @ targets_w
        loss = sum(loss_fn(m(obs), target) for m in models) / N_AGENTS
        opt.zero_grad()
        loss.backward()
        opt.step()

        if first_loss is None:
            first_loss = loss.item()
        last_loss = loss.item()
        if step % max(1, args.steps // 10) == 0 or step == 1:
            log(f"step {step:>5d}/{args.steps} | loss {loss.item():.5f}")
        if tracker is not None:
            tracker.log_metrics({"loss": loss.item()}, step=step)

    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    sps = args.steps / dt
    samples_s = sps * args.batch_size * N_AGENTS

    log("-" * 70)
    log(f"Wall time  : {dt:.2f} s | {sps:.1f} steps/s | {samples_s:,.0f} agent-samples/s")
    if device == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1024**2
        log(f"GPU memory : {mem:.1f} MB peak allocated")

    # checkpoint round-trip
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"smoke_{stamp}.pt"
    torch.save({f"agent_{i}": m.state_dict() for i, m in enumerate(models)}, ckpt_path)
    reloaded = torch.load(ckpt_path, map_location=device, weights_only=True)
    log(f"Checkpoint : saved + reloaded OK ({ckpt_path.name}, {len(reloaded)} agents)")

    improved = first_loss is not None and last_loss < 0.5 * first_loss
    log(
        f"Loss       : {first_loss:.5f} -> {last_loss:.5f} "
        f"({'decreased >=50%' if improved else 'insufficient decrease'})"
    )
    log("-" * 70)
    if tracker is not None:
        tracker.log_metrics({"steps_per_s": sps, "final_loss": last_loss})
        tracker.finish("FINISHED" if improved else "FAILED")
    if improved:
        log(f"RESULT: PASS — smoke training job completed on {device}.")
        log.close()
        return 0
    log("RESULT: FAIL — loss did not decrease as expected.")
    log.close()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
