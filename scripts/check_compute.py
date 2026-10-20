#!/usr/bin/env python3
"""Compute provisioning check (task m1-t3).

Verifies the PyTorch install, CUDA availability, and GPU health, then runs a
short matmul benchmark. Everything is echoed to stdout AND appended to
results/logs/compute_check_<timestamp>.log so the log itself is the deliverable.

Usage:
    python scripts/check_compute.py             # fails (exit 1) if no CUDA GPU
    python scripts/check_compute.py --allow-cpu # report only, never fail
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


def benchmark_matmul(device: str, size: int = 4096, iters: int = 20) -> float:
    """Return achieved TFLOP/s for a size^3 matmul on the given device."""
    import torch

    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    # warmup
    for _ in range(3):
        a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    flops = 2 * size**3 * iters
    return flops / dt / 1e12


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Do not fail when no CUDA device is present.",
    )
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log = Tee(LOG_DIR / f"compute_check_{stamp}.log")

    log("=" * 70)
    log("GovGridMARL compute check (task m1-t3)")
    log(f"UTC time      : {datetime.now(timezone.utc).isoformat()}")
    log(f"Host          : {platform.node()} ({platform.system()} {platform.release()})")
    log(f"Python        : {sys.version.split()[0]} @ {sys.executable}")
    log("=" * 70)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        log(f"FAIL: PyTorch is not installed ({exc}).")
        log("Fix : run scripts/setup_env.sh or pip install -r requirements.lock")
        log.close()
        return 1

    log(f"torch         : {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"CUDA (built)  : {torch.version.cuda}")

    if torch.cuda.is_available():
        log(f"cuDNN         : {torch.backends.cudnn.version()}")
        log(f"GPU count     : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            log(
                f"GPU {i}         : {props.name} | {mem_gb:.1f} GB | "
                f"CC {props.major}.{props.minor} | {props.multi_processor_count} SMs"
            )
        device = "cuda"
    else:
        log("GPU           : none detected")
        device = "cpu"

    log("-" * 70)
    log(f"Running matmul benchmark on {device} ...")
    tflops = benchmark_matmul(device)
    log(f"matmul 4096^3 : {tflops:.2f} TFLOP/s on {device}")

    if device == "cuda":
        # round-trip sanity: results computed on GPU match CPU
        x = torch.randn(256, 256)
        ok = torch.allclose((x.cuda() @ x.cuda()).cpu(), x @ x, atol=1e-3)
        log(f"GPU<->CPU numerical round-trip: {'OK' if ok else 'MISMATCH'}")

    log("-" * 70)
    if device == "cpu" and not args.allow_cpu:
        log("RESULT: FAIL — no CUDA GPU (rerun with --allow-cpu to accept CPU).")
        log.close()
        return 1
    log(f"RESULT: PASS — compute check succeeded on {device}.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
