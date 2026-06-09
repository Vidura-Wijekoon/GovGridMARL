"""Lightweight structured logging helper.

Wraps the standard library logging with a consistent format used across
training, evaluation, and the governance audit trail.
"""

from __future__ import annotations

import logging

_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def get_logger(name: str = "govgridmarl", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FMT))
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
