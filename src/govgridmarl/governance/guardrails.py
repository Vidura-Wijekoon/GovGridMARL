"""Deterministic runtime guardrail: allow/deny safety gate (Phase 3, m9-t4)."""

from __future__ import annotations


class Guardrail:  # pragma: no cover - stub
    """Vetoes unsafe actions before they reach the grid. TODO(m9-t4)."""

    def check(self, action, state) -> bool:
        raise NotImplementedError("Guardrail.check — scheduled for Month 9")
