# GovGridMARL — Scope Note (task m1-t5)

**Author:** Vidura · **Date:** July 2026 · **Horizon:** 10 months (Jul 2026 → Apr 2027) · **Compute:** single laptop GPU
**Sources:** proposal v4; 99-paper literature corpus (coverage: MARL/DR 37, XAI 17, fairness 16, privacy/FL 14, governance 13, safety/safe-RL 6)

## Research question and hypothesis (restated from proposal §2–3)

**RQ (overall):** How can a multi-agent reinforcement learning framework with a *runtime governance layer* be designed for residential smart-grid demand response such that it simultaneously satisfies operational, fairness, privacy, and accountability requirements?

**Hypothesis:** Embedding governance as a first-class engineering constraint (not a post-hoc metric) yields competitive operational performance with quantifiably stronger fairness, privacy, and accountability guarantees than unconstrained MARL baselines.

**Sub-questions (mapped to RO1–RO4):**

- **RQ1** (RO1): How should a heterogeneous residential DR environment (IEEE-14, socio-economically differentiated households) be designed so fairness is measurable and optimizable?
- **RQ2** (RO2): What efficiency–fairness and privacy–utility trade-offs arise when MAPPO is extended with Lagrangian Jain-index constraints and DP-federated learning?
- **RQ3** (RO3): Can per-decision SHAP explanations plus a runtime YAML governance engine (policy gates, HITL override, immutable audit log) achieve 100% coverage at acceptable runtime overhead?
- **RQ4** (RO4): How does GovGridMARL compare to rule-based DR and unconstrained MAPPO on all KPIs, and can the artefact be documented to EU AI Act Annex IV?

## Success metrics (unchanged targets from proposal §6.7)

| Operational | Target | vs | Governance | Target |
|---|---|---|---|---|
| Peak demand reduction | ≥ 15% | rule-based DR | Jain fairness index | ≥ 0.85 |
| Renewable utilization | ≥ +10% | unconstrained MARL | Privacy budget ε | ≤ 1.0 |
| Frequency deviation | < 0.2 Hz | unconstrained MARL | SHAP coverage / gate pass / audit completeness | 100% |

*Pre-registered fallback:* where a target proves unreachable on laptop-scale compute, report the full trade-off curve against a pre-declared minimum threshold (per proposal risk plan) rather than a single point.

## Scope boundaries (10 months, single laptop GPU)

**In scope:** IEEE-14 bus network (Grid2Op + pandapower + lightsim2grid); 10–20 household agents in 3 income groups; MAPPO + Lagrangian fairness/safety constraints (CTDE); baselines = rule-based DR, unconstrained MAPPO, IPPO; *simulated* federated rounds in-process with Opacus DP (ε ∈ {0.5, 1, 5, ∞}); SHAP global + per-decision local explanations with template-based natural-language summaries; YAML governance engine with **SQLite** audit store and minimal Streamlit dashboard; 3–5 seeds per configuration; MLflow tracking (m1-t4); open repo with conda/pip lockfiles (m1-t2).

**Out of scope (declared future work):** IEEE-118 training (qualitative scalability discussion only); real household data or hardware-in-the-loop; genuinely distributed edge deployment (federation is simulated); PostgreSQL (SQLite suffices at this scale); LLM-generated explanation text (templates instead; local-LLM is a stretch goal); adversarial robustness / cyberattack defence; P2P energy trading, topology control, and voltage regulation as objectives; formal secure aggregation protocols (CKKS etc.).

**Feasibility basis:** IEEE-14 with lightsim2grid runs at thousands of env-steps/s on CPU; a 1M-step MAPPO run fits in hours on one consumer GPU (verified pipeline: m1-t2/t3/t4), so the full evaluation grid (≤4 configs × ≤5 seeds) fits in days, not weeks.

## Milestone alignment

P1 baseline MARL converges on IEEE-14 (M1–3) → P2 Jain ≥ 0.85, ≥ 15% peak cut, zero safety violations (M4–5) → P3 DP + XAI + audit coverage targets (M6–7) → P4 end-to-end pipeline + Annex IV checklist (M8–9) → P5 submission with public repo and CI (M10).
