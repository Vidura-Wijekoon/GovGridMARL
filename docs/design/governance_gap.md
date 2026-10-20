# Governance-gap synthesis (task m2-t5)

The full argument, with the coverage table and the framework-comparison matrix, is §9 of the literature-review paper (`docs/lit_notes/review_paper_source.md`; IEEE build `GovGridMARL_Literature_Review_Paper_IEEE_v2.docx`). This note is the *thesis-facing* version: it states the gap in one paragraph, then maps each missing dimension to the GovGridMARL component that closes it and to the roadmap month that delivers it. It is the bridge between the review (contribution C1) and the artefact (C2–C4).

## 1. The gap in one paragraph

Across 105 works (2004–2026) the five capability dimensions are individually mature — MARL 42, privacy 26 (+4 partial), governance 26, fairness 25, XAI 24 — but their intersection is empty in practice. MARL co-occurs with fairness in five works (none a DR controller), with XAI in three, with governance in three; only one work [#31] nominally spans all five and it is single-agent with no runtime governance layer. The operationally strongest MARL-DR systems [#2, #13, #16, #21] report 14–23 % peak reductions and score N on every governance dimension. The reason is structural: these controllers optimise an *efficiency-only* objective (peak, cost), so fairness, privacy, explainability and accountability are at best measured after training and never shape the policy. Under the EU AI Act (Annex III energy infrastructure → high-risk), NIS2 and ISO/IEC 42001 those properties become obligations, so an efficiency-only MARL controller is, as of 2026, not deployable. That is the governance gap, and the GovGridMARL programme treats it as an engineering constraint placed inside the learning objective and the runtime rather than as a reporting exercise.

## 2. Gap → component → roadmap

| Missing in the literature | Evidence (review) | GovGridMARL component | Where in repo | Roadmap |
|---|---|---|---|---|
| Efficiency-only objective; no fairness in the optimiser | fair-RL not energy-specific [#51–#54]; DR fairness not RL [#46, #49] | Jain-index constraint on curtailment burden across income groups, Lagrangian in MAPPO; MOPPO Pareto front | `fairness/jain.py`, `agents/lagrangian.py`, `agents/moppo.py` | M4–M5 |
| Safety handled in isolation, never with fairness | [#105], [#12] partial | Voltage / thermal / comfort costs measured every step (done) and priced by the same Lagrangian | `safety/constraints.py`, `GridMARLEnv.step` | M2 (measure) → M5 (enforce) |
| Privacy as a by-product of decentralisation, no stated ε | [#16, #23] P; [#67] Y but single-agent | Simulated FedAvg over household actors with Opacus DP, ε ∈ {0.5, 1, 5, ∞}; privacy–utility curve | `privacy/fedavg.py`, `privacy/dp.py` | M7 |
| XAI only for single-agent RL; no per-decision explanations | [#89, #90] | SHAP over the 17 named observation features per household decision; template NL summaries | `explainability/shap_explainer.py`, `docs/design/agents_and_spaces.md` §2 | M8 |
| Governance frameworks conceptual, no runtime enforcement | [#26, #27, #30] "conceptual" | YAML policy engine (gates + fallback), HITL override, hash-chained audit log, Annex IV mapping | `governance/guardrails.py`, `governance/audit_log.py` | M9–M10 |
| No integrated evaluation | Table 2 all rows | One protocol: peak ≥ 15 % vs rule-based DR, Jain ≥ 0.85, ε ≤ 1, 100 % SHAP / gate / audit coverage | `docs/evaluation_protocol.md`, `experiments/evaluate.py` | M3 (protocol) → M11 |

## 3. Why the Phase-1 baseline is efficiency-only on purpose

The baseline learners (PPO, IPPO, MAPPO) are trained with `reward=efficiency` (fairness and safety weights = 0). This reproduces exactly the class of controller the review criticises, on our own environment, so that every later governance mechanism is measured as a *delta* against it: how much peak reduction is given up for Jain ≥ 0.85, how much for ε ≤ 1, how much runtime for 100 % audit coverage. Without that reference point the thesis' central claim — "competitive operational performance with quantifiably stronger guarantees" — could not be tested.

## 4. Claims the thesis will and will not make

* Will: the *integration* is novel and measurable; each mechanism is standard and cited.
* Will: results on IEEE-14 with 14 cluster agents and synthetic-calibrated loads; IEEE-118 only as a scalability discussion.
* Will not: claim real-world household data, real distributed deployment, or formal secure aggregation (all declared out of scope in `docs/scope_note.md`).
