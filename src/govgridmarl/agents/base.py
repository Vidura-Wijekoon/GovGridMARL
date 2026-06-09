"""Abstract agent interface shared by all learners."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Minimal policy interface: act, update, save, load."""

    @abstractmethod
    def act(self, observation: Any) -> Any:
        """Return an action for a single observation."""
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Any) -> dict:
        """Run one optimisation step; return a dict of metrics."""
        raise NotImplementedError

    def save(self, path: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError

    def load(self, path: str) -> None:  # pragma: no cover - stub
        raise NotImplementedError
