"""Pre-registered acceptance thresholds and the Phase-2 gate check (task m4-t10).

Fixed from the proposal (section 6.7) and ``configs/eval/default.yaml``:
peak reduction >= 15 % against no control, Jain >= 0.85 on the kWh burden,
zero safety violations (violation rate 0 on the 20 held-out days). The
thresholds are *minima*; a configuration passes the gate only when every
target holds simultaneously on the same policy (m5-t11).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PHASE2_TARGETS: dict[str, float] = {
    "peak_reduction_pct": 15.0,  # >=
    "jain_index": 0.85,  # >=
    "violation_rate": 0.0,  # <=
}
HIGHER_IS_BETTER = {"peak_reduction_pct": True, "jain_index": True, "violation_rate": False}


@dataclass
class TargetCheck:
    passed: bool
    results: dict[str, dict[str, Any]] = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        for k, r in self.results.items():
            op = ">=" if HIGHER_IS_BETTER.get(k, True) else "<="
            flag = "ok" if r["ok"] else "MISS"
            parts.append(f"{k} {r['value']:.3f} {op} {r['target']:.3f}: {flag}")
        return ("PASS" if self.passed else "FAIL") + " | " + "; ".join(parts)


def check_targets(
    metrics: dict[str, float],
    targets: dict[str, float] | None = None,
    tol: float = 1e-9,
) -> TargetCheck:
    """Compare a metrics dict against the Phase-2 thresholds."""
    targets = dict(PHASE2_TARGETS, **(targets or {}))
    res: dict[str, dict[str, Any]] = {}
    ok_all = True
    for k, tgt in targets.items():
        v = float(metrics.get(k, float("nan")))
        if HIGHER_IS_BETTER.get(k, True):
            ok = v + tol >= tgt
        else:
            ok = v - tol <= tgt
        ok = bool(ok) and v == v  # NaN fails
        res[k] = {"value": v, "target": float(tgt), "ok": ok}
        ok_all &= ok
    return TargetCheck(ok_all, res)


__all__ = ["PHASE2_TARGETS", "HIGHER_IS_BETTER", "TargetCheck", "check_targets"]
