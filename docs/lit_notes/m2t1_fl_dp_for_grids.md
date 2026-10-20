# Deep-read notes — Federated Learning + Differential Privacy for smart grids (task m2-t1)

**Author:** Vidura · **Date:** July 2026 · **Cluster:** privacy / FL (14 papers per scope note)
**Feeds:** RQ2/RO2 (privacy–utility trade-off), the privacy layer of GovGridMARL, and the lit-review draft (m2-t4).
**Citation key:** papers referenced by title and by their explainer ID `paperNN` (see `Literature_Review_Videos/`). Row IDs `[#N]` cross-reference the Paper Database sheet where confirmed in `framework_comparison.md`; verify remaining row IDs against `GovGridMARL_Literature_Review_Database.xlsx`.

> **Status flag for review:** methods, mechanisms and contributions below are drawn from the corpus and my reading of these works. Where a *specific quantitative result* is cited it is marked "(verify)" — confirm the exact figure against the PDF before it enters the thesis body. The synthesis and gap argument are mine to defend; treat them as a first draft of your position.

---

## 1. Why this cluster matters for GovGridMARL

The proposal commits to *calibrated* differential privacy with a **stated budget (ε ≤ 1.0)** implemented via **Opacus** over **simulated federated rounds** (scope note, "In scope"). This cluster establishes (a) what FL architectures the smart-grid community actually uses, (b) which privacy mechanisms are applied and at what utility cost, and (c) the gap the thesis exploits: privacy is almost always a *side-effect of decentralisation* rather than a budgeted, reported guarantee combined with fairness and governance (framework_comparison.md, gap summary).

## 2. Sub-theme A — FL architectures for the grid

- **Federated Learning for Smart Grid** and **Crossing Roads of Federated Learning and Smart Grids** — foundational/survey framing: horizontal FL across smart meters or DERs, FedAvg-style aggregation, non-IID load profiles as the central statistical difficulty. Establish the canonical client=household / server=utility topology GovGridMARL simulates in-process.
- **Federated Learning Architectures for Privacy-Preserving Smart Grid Data Processing** (`paper20`) — taxonomy of architectures (centralised-aggregator vs. hierarchical vs. P2P); useful for justifying the *simulated single-aggregator* choice as the minimal faithful model at laptop scale.
- **Cluster-based FRL for residential microgrids** (`paper67`) — clusters households before federating to mitigate non-IID; directly relevant to the 3-income-group structure (RQ1) — income group is a natural, defensible clustering variable.
- **Personalized FL for load forecasting** (`paper3`) and **Personalized Federated Hypernetworks for privacy preservation in multi-task RL** (`paper2`) — personalisation (per-client heads / hypernetworks) recovers accuracy lost to non-IID; the multi-task-RL hypernetwork paper is the closest prior art to *FL + RL* and worth a dedicated paragraph in the review.

## 3. Sub-theme B — Privacy mechanisms (DP, HE, secure aggregation)

- **Scalable and Private Federated Learning Using Distributed Differential Privacy and Secure Aggregation** (`paper34`) — the reference design for **DP + secure aggregation** together; the "distributed DP" idea (noise added under secure aggregation so no single client over-perturbs) is the rigorous version of what GovGridMARL approximates with central Opacus noise. Cite as the gold standard the thesis *simplifies* (declared out-of-scope: formal secure aggregation).
- **PRIVACY-PRESERVING DECENTRALIZED FEDERATED LEARNING** / **PrivateDFL (HD + DP)** (`paper33`) — decentralised (no central server) FL with DP; supports the argument that DP is the portable privacy primitive across topologies, whereas HE/secure-agg are topology-specific.
- **A federated LSTM network for load forecasting … with homomorphic encryption** (`paper5`) and **Smart Grid IoT … Federated Learning + Homomorphic Encryption** — HE route: strong confidentiality, heavy compute. Good contrast case for *why* the thesis picks DP (calibratable budget, cheap) over HE (no budget knob, expensive) at laptop scale.
- **P2P FL with CKKS** (`paper32`) — CKKS approximate HE for P2P; reinforces "secure aggregation protocols (CKKS etc.)" as explicitly out-of-scope but acknowledged.

## 4. Sub-theme C — FL + RL / control (closest prior art)

- **Federated DRL for a shared energy storage system** (`paper4`) and **Privacy-Preserving Energy Management of a Shared Energy Storage System — a Federated Deep RL approach** — the two papers that most resemble GovGridMARL's *federated RL for DR/storage*. Key point for the gap: they federate the RL policy but report privacy only qualitatively (decentralisation ⇒ "private"); **no ε is stated**. This is the single strongest anchor for the RO2 gap claim.
- **Personalized Federated Hypernetworks in multi-task RL** (`paper2`) — see §2; method-transfer candidate if per-agent personalisation is later needed.

## 5. Sub-theme D — Attacks, vulnerabilities, surveys (threat model)

- **FL for Smart Grid survey — vulnerabilities** (`paper66`), **Federated Learning for Smart Grid survey** (`paper81`), **Smart-meter privacy survey** (`paper83`), **Privacy Preservation in Smart Meters: Current Status, Challenges and Future Directions** — establish the threat model: gradient-inversion / membership-inference against shared updates, and NILM-style disaggregation as the concrete privacy harm (load traces leak occupancy/appliance use).
- **FL for energy disaggregation / NILM** (`paper31`) — makes the harm concrete: smart-meter traces reveal behaviour, which is *why* per-household DP with a small ε is the right protection for the DR setting.
- **Blockchain-enabled FL** (`paper6`, `paper68`; **Towards a cybersecure and privacy-enhanced smart grid…** [#24 partial-gov]) — blockchain for auditability/integrity of federated updates; relevant to the *audit-log* governance layer later, but privacy is still not budgeted.
- **Federated Deep Learning for Scalable and Explainable …** — bridges to the XAI cluster (m2-t2): federated + explainable is rare and partial.

## 6. Privacy–utility trade-off — what the evidence supports

Across the cluster the consistent finding is a **monotone accuracy cost that grows as ε shrinks**, with non-IID data amplifying the cost; personalisation and clustering partially recover it (§2). No surveyed grid paper reports a *fairness*-conditioned privacy curve. This directly justifies GovGridMARL's design: sweep **ε ∈ {0.5, 1, 5, ∞}** and report the full privacy–utility (and privacy–fairness) curve rather than a single point (scope note pre-registered fallback). Specific reported deltas in these papers are model/dataset-specific — **(verify)** each figure before quoting.

## 7. Directly usable decisions for the build

- Privacy primitive: **Opacus DP-SGD**, per-household client, central noise as a faithful-but-simplified stand-in for distributed DP (`paper34` is the rigorous target we cite and simplify).
- Budget: report **(ε, δ)** with δ = 1/N_households scale; headline target **ε ≤ 1.0**, swept to ∞ as the no-privacy baseline.
- Federation: **simulated in-process rounds** (FedAvg over agent policy/value nets), non-IID induced naturally by income-group load differences → clustering by income group (`paper67`) is a defensible, thesis-relevant design.
- Threat model to state explicitly in the thesis: gradient-inversion + NILM disaggregation of shared updates (`paper31`, `paper66`, `paper83`).

## 8. Gap this cluster leaves (feeds m2-t5 governance-gap synthesis)

1. Privacy is reported **qualitatively** (decentralisation) far more often than as a **calibrated ε budget** — especially in the FL+RL control papers (`paper4`, shared-storage FDRL).
2. **No** surveyed grid-FL paper combines a stated privacy budget with a **fairness** guarantee in one optimiser; none adds a **runtime governance/audit** layer.
3. HE / secure-aggregation give strong confidentiality but **no tunable budget** and heavy cost → DP is the right laptop-scale choice, with HE/CKKS acknowledged as out-of-scope future work.

→ GovGridMARL's privacy contribution is not a new DP mechanism but the **first placement of a budgeted DP guarantee inside a fairness-constrained, governance-audited MARL-DR controller**.

## 9. Open questions to resolve before implementation (P3, Months 6–7)

- δ target and accountant (RDP vs. PRV) under Opacus for the chosen N and round count.
- Whether to add DP noise to *policy* updates, *value* updates, or both — affects the RL stability/privacy trade-off (no clear precedent in the cluster; a small ablation is warranted).
- Clustering granularity: income group (3) vs. per-bus vs. per-household personalisation.

---

*Cross-refs:* scope_note.md (RQ2/RO2, ε targets, out-of-scope), framework_comparison.md (gap summary, [#24]). Next: m2-t2 XAI/SHAP deep-read.
