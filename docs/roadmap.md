# Research Roadmap (Jun 2026 -> May 2027)

The thesis runs over 12 months in 5 phases, 180 tasks total (15/month),
grounded in the proposal Gantt. Source of truth: `GovGridMARL_Research_Path.html`.

## Phases & milestones

| Phase | Window | Focus | Milestone gate |
|---|---|---|---|
| **P1 - Foundation & Baseline** | Jun-Aug 2026 | Setup, literature, baseline MARL | Review >=50 papers; baseline MARL converges on IEEE-14 |
| **P2 - Fairness & Safety** | Sep-Nov 2026 | Multi-objective + constrained RL | Jain >=0.85; >=15% peak cut; zero safety violations |
| **P3 - Privacy, XAI & Audit** | Dec 2026-Feb 2027 | FL+DP, SHAP, governance layer | DP epsilon <=1.0; SHAP fidelity >=90%; audit log 100% coverage |
| **P4 - Integration & Compliance** | Mar-Apr 2027 | End-to-end pipeline + EU AI Act | E2E pipeline passes; Annex IV checklist met |
| **P5 - Submission** | May 2027 | Finalise, submit, publish | Thesis submitted; repo public with CI passing |

## Months

| # | Month | Theme |
|---|---|---|
| 1 | Jun 2026 | Project setup & literature kickoff |
| 2 | Jul 2026 | Literature synthesis & environment modelling |
| 3 | Aug 2026 | Baseline MARL convergence on IEEE-14 |
| 4 | Sep 2026 | Multi-objective & fairness formulation |
| 5 | Oct 2026 | Safe RL & constraint satisfaction |
| 6 | Nov 2026 | Consolidation & scale test (IEEE-118) |
| 7 | Dec 2026 | Federated learning & privacy |
| 8 | Jan 2027 | Explainability (XAI / SHAP) |
| 9 | Feb 2027 | Audit logging & governance layer |
| 10 | Mar 2027 | Integration & end-to-end pipeline |
| 11 | Apr 2027 | Evaluation, DSR validation & write-up |
| 12 | May 2027 | Finalise, submit & publish |

## Month 1 - current focus (Jun 2026)

1. **Initialise version-controlled repo**  <- this deliverable
2. Build reproducible Python environment (`environment.yml` / lockfile)
3. Provision & test compute (GPU smoke job)
4. Set up experiment tracking (W&B / MLflow + Hydra)
5. Refine research questions & scope (1-page note)
6. Assemble reference library (105-entry BibTeX -> Zotero)
7. Deep-read core MARL-for-DR papers (x15)
8. Deep-read safe / constrained RL
9. Deep-read fairness & energy-justice
10. Draft lit-review: MARL-for-DR cluster
11. Build framework comparison matrix
12. Set up thesis LaTeX template
13. Smoke-test Grid2Op (L2RPN sandbox)
14. Solve IEEE-14 power flow (pandapower Newton-Raphson)
15. Month-1 log & supervisor meeting #1
