# Month-1 Log — Project setup & literature kickoff (task m1-t15)

**Period:** June–July 2026 (calendar slipped; closed 6 Sep 2026) · **Phase:** P1 Foundation & Baseline · **Author:** Vidura

## Deliverables

| # | Task | Status | Evidence |
|---|------|--------|----------|
| m1-t1 | Initialise version-controlled repo | Done | commit `546d20a`, CHANGELOG 0.1.0 |
| m1-t2 | Reproducible Python environment | Done | `requirements.lock` (179 pins), `environment.yml`, CHANGELOG 0.1.1 |
| m1-t3 | Provision & test compute | Done | `results/logs/compute_check_20260707_061317.log`, `smoke_train_20260707_062552.log` |
| m1-t4 | Experiment tracking | Done | `govgridmarl.utils.tracking.Tracker` (MLflow SQLite at repo root), `make dashboard` |
| m1-t5 | Refine research questions & scope | Done | `docs/scope_note.md` |
| m1-t6 | Reference library | Done | `Literature_Review/GovGridMARL_references.bib` (105 entries) |
| m1-t7–t9 | Deep-reads (MARL-DR, safe RL, fairness) | Done | 99 explainer HTMLs in `Literature_Review_Videos/`, review DB |
| m1-t10 | Lit-review: MARL-for-DR cluster | Done | `docs/lit_notes/review_paper_source.md` §Area 1 |
| m1-t11 | Framework comparison matrix | Done | `docs/framework_comparison.md`, `GovGridMARL_Framework_Comparison.xlsx` |
| m1-t12 | Thesis LaTeX template | **Skipped (decision)** | Writing in Markdown → pandoc → Word/IEEE templates; see below |
| m1-t13 | Grid2Op smoke test | Done | `grid2op_smoke_20260729_124637.log` (PASS), CHANGELOG 0.1.2 |
| m1-t14 | IEEE-14 power flow | Done | `ieee14_powerflow_20260730_004228.log` (PASS), CHANGELOG 0.1.3 |
| m1-t15 | Month-1 log & supervisor meeting #1 | This document | — |

## Decisions

1. **m1-t12 skipped.** The literature-review paper was required in Word (single-column) and IEEE two-column formats; both are produced from one Markdown source (`docs/lit_notes/review_paper_source.md`) by pandoc and a python-docx builder. The thesis will follow the same pipeline, so a LaTeX template adds no value. Revisit only if the faculty mandates LaTeX.
2. **Simulation backend.** Grid2Op works (m1-t13) but its action model (topology / redispatch) does not match household curtailment. The environment is therefore built directly on pandapower `case14` with LightSim2Grid's C++ solver for speed (~0.1 ms/solve); Grid2Op stays available for the IEEE-118 scalability discussion (Month 6).
3. **Voltage band.** case14 generator set-points legitimately hold bus 8 at 1.09 pu, so the upper voltage limit for the safety constraint is 1.10 pu, not 1.05 (see m1-t14 log).
4. **Reproducibility.** Grid2Op chronics rotate on every `reset()` independent of the seed; `env.set_id()` + `env.seed()` are required before `reset()` for bit-exact replays. The pandapower environment is fully seeded through `numpy.random.default_rng`.

## Known issues

* The repo's `.git` on the laptop is partially corrupt (`._git_corrupt_delete_me`); commits after `ca55584` were not recorded. Action: re-init the history from the current tree before making the repo public (m3-t13 / P5).
* The literature DB has 105 papers but only 99 explainer videos; six entries lack a video.

## Supervisor meeting #1 — agenda & outcomes

**Agenda:** (1) scope note sign-off; (2) literature-review paper (IEEE v2) feedback — "sounds like AI" rewrite applied; (3) backend decision (pandapower vs Grid2Op); (4) Month 2–3 plan.

**Outcomes to record after the meeting:** approval of scope boundaries (IEEE-14 only, simulated federation, SQLite audit store), agreement on the Phase-1 gate criteria (baseline MAPPO converges on IEEE-14; ≥ 50 papers reviewed — met with 105), and any requested changes to the CMOMG formulation (`docs/design/cmomg_formulation.md`).

## Status at close

Month 1 complete (14/15 tasks done, 1 skipped by decision). Month 2 and Month 3 deliverables were produced in a catch-up sprint on 6 Sep 2026; see `docs/reports/phase1_milestone_report.md`.
