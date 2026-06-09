"""Tamper-evident audit logging of agent decisions (Phase 3, m9-t2).

Each record hashes the previous record (hash chain) to make the log
tamper-evident. Logged fields are mapped to EU AI Act Annex IV (m9-t7).
Phase-3 milestone: 100 % audit coverage over an evaluation run.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditRecord:
    """A single decision record in the hash-chained audit log."""

    timestamp: float
    agent_id: str
    decision: Any
    constraint_checks: dict
    prev_hash: str = ""
    explanation: dict = field(default_factory=dict)

    def digest(self) -> str:
        payload = json.dumps(
            {
                "timestamp": self.timestamp,
                "agent_id": self.agent_id,
                "decision": self.decision,
                "constraint_checks": self.constraint_checks,
                "prev_hash": self.prev_hash,
                "explanation": self.explanation,
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class AuditLog:
    """Append-only, hash-chained audit log (in-memory reference impl)."""

    def __init__(self) -> None:
        self._records: list[tuple[AuditRecord, str]] = []

    def append(
        self, agent_id: str, decision: Any, constraint_checks: dict, explanation: dict | None = None
    ) -> str:
        prev_hash = self._records[-1][1] if self._records else ""
        rec = AuditRecord(
            timestamp=time.time(),
            agent_id=agent_id,
            decision=decision,
            constraint_checks=constraint_checks,
            prev_hash=prev_hash,
            explanation=explanation or {},
        )
        h = rec.digest()
        self._records.append((rec, h))
        return h

    def verify(self) -> bool:
        """Return True if the hash chain is internally consistent."""
        prev = ""
        for rec, h in self._records:
            if rec.prev_hash != prev or rec.digest() != h:
                return False
            prev = h
        return True

    def __len__(self) -> int:
        return len(self._records)
