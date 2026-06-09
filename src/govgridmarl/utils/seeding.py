"""Reproducible seeding across Python, NumPy, and (optionally) PyTorch.

Reproducibility is a first-class requirement of the thesis (multi-seed
results in Phase 1, repro package in Phase 4). Keep this dependency-light:
PyTorch is imported lazily so the helper works in minimal environments.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, *, deterministic: bool = True) -> int:
    """Seed Python, NumPy and PyTorch RNGs.

    Parameters
    ----------
    seed:
        The seed value to apply across all RNGs.
    deterministic:
        If True, request deterministic cuDNN behaviour where available.

    Returns
    -------
    int
        The seed that was applied (useful for logging).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy optional at skeleton stage
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # torch optional at skeleton stage
        pass

    return seed
