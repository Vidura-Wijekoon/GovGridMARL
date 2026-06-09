"""SHAP-based attribution of curtailment decisions (Phase 3, m8-t1).

Phase-3 milestone: SHAP fidelity >= 90 %. Produces per-household, per-step
attributions and natural-language explanation templates (m8-t6).
"""

from __future__ import annotations


class ShapExplainer:  # pragma: no cover - stub
    """Wraps shap.Explainer around a trained policy. TODO(m8-t1)."""

    def explain(self, policy, observations):
        raise NotImplementedError("ShapExplainer.explain — scheduled for Month 8")
