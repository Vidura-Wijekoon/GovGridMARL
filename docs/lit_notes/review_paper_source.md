# Governance-Aware Multi-Agent Reinforcement Learning for Residential Smart-Grid Demand Response: A Systematic Review of Operational, Fairness, Privacy, Explainability, and Regulatory Dimensions

**Vidura** ¹ · [Supervisor name] ¹
¹ MSc Artificial Intelligence & Data Science

*Corresponding author:* businessaividura@viduraaitech.space

---

## Abstract

Multi-agent reinforcement learning (MARL) is now the leading machine-learning approach to residential smart-grid demand response (DR), and it reports double-digit reductions in peak demand and operating cost across increasingly realistic simulations. As grid-facing artificial intelligence is reclassified as high-risk under the EU AI Act and related regimes, including NIS2, ISO/IEC 42001, and the guidance of national regulators such as Ofgem, a gap has emerged between what these controllers optimise and what they are now legally and socially required to guarantee. The obligations in question are distributional fairness, data privacy, decision transparency, and auditable accountability. This paper reviews 105 works published between 2004 and 2026 across five research areas: MARL for DR and grid control; fairness and energy justice; privacy and federated learning (FL); explainable AI (XAI) in energy; and AI governance, ethics, and regulation. A set of cross-cutting methodological foundations is also included. Each work was coded against five capability dimensions, namely MARL, fairness, privacy, XAI, and governance. Although every dimension is individually mature, the dimensions are rarely combined. Of the 105 works, only one addresses all five, and that work is single-agent and provides no runtime governance layer. No reviewed work combines a genuine multi-agent architecture with calibrated differential privacy, a fairness-constrained optimiser, per-decision explanations, and an enforceable audit trail for residential DR. We define this shortfall as a five-dimension governance gap, present a framework-comparison matrix that locates it, and argue that governance should be treated as a design-time engineering constraint rather than a property measured after the event. The paper closes with a research agenda for constrained, privacy-preserving, explainable, and regulation-aligned MARL for demand response.

**Keywords:** multi-agent reinforcement learning; demand response; smart grid; algorithmic fairness; energy justice; federated learning; differential privacy; explainable AI; AI governance; EU AI Act

---

## 1. Introduction

Electricity systems must now absorb variable renewable generation, electrified heat and transport, and demand that peaks sharply within narrow windows. Demand response, which shifts flexible residential load away from those peaks, is one of the cheapest options available to system operators, and reinforcement learning (RL) has been shown to learn DR policies that outperform hand-tuned rules. Because a distribution network hosts many semi-autonomous consumers, the natural formulation is multi-agent RL, in which each household or aggregator acts as an agent whose local decisions couple through a shared power-flow model.

By the evidence assembled here, the technical development of MARL for DR has been successful. Centralised-training / decentralised-execution (CTDE) architectures, and multi-agent proximal policy optimisation (MAPPO) in particular, scale to hundreds of heterogeneous loads and report peak reductions above 20% and cost reductions approaching 19% in large simulations [#13, #16, #21]. This progress has, however, outpaced the field's attention to a second set of requirements that are no longer optional. Grid-facing AI is now classified as high-risk under the EU AI Act [#32, #88], carries binding cybersecurity duties under NIS2 [#42], and is governed by management-system standards such as ISO/IEC 42001 [#41] and by sector regulators [#34]. These regimes require an automated controller to be fair to the households it curtails, to protect the fine-grained consumption data it ingests, to explain the decisions it makes, and to remain accountable through records that survive audit.

The central observation of this review is that these two research programmes, high-performance MARL for DR and the governance requirements now imposed on it, have developed largely in parallel. Operationally strong controllers seldom measure fairness, state a privacy budget, explain individual decisions, or embed a runtime governance layer. The governance, fairness, privacy, and XAI literatures are, by contrast, extensive, but they are demonstrated for the most part outside a multi-agent DR controller, often as conceptual frameworks with "no code-level implementation" [#26, #28, #30].

This paper makes three contributions. First, it provides a systematic map of the field along five dimensions, coding 105 works for MARL, fairness, privacy, XAI, and governance capability. Second, it quantifies the integration gap. The individual dimensions are mature, with 25 to 42 papers each, yet their intersection is empty in practical terms, because only one paper nominally spans all five and that paper is single-agent. Third, it synthesises a framework-comparison matrix and sets out a research agenda in which governance is engineered into the learning objective and the runtime from the outset.

The review is organised around four questions. **RQ-A:** How mature and how well integrated are MARL methods for residential DR and grid control? **RQ-B:** How are fairness, privacy, and explainability currently formalised in energy AI, and are these formulations compatible with a learning controller? **RQ-C:** What do the emerging governance and regulatory regimes require of a grid-facing controller, and how far are current systems from meeting those requirements in practice? **RQ-D:** What would an integrated, governance-aware MARL-DR framework need to provide in order to close the gap?

---

## 2. Review methodology

### 2.1 Search strategy

The corpus was assembled between May and July 2026. The primary method was structured searching of Google Scholar, arXiv, IEEE Xplore, ScienceDirect, the ACM Digital Library, and MDPI. This was supplemented by backward and forward citation tracing and by direct retrieval of primary regulatory sources, namely the EU AI Act, NIS2, ISO/IEC 42001, and the Ofgem guidance. Each search string combined three facets: a method facet ("multi-agent reinforcement learning", "MARL", "deep reinforcement learning", "federated learning", "explainable AI", "SHAP"); an application facet ("demand response", "smart grid", "microgrid", "energy management"); and a governance facet ("fairness", "energy justice", "differential privacy", "AI Act", "accountability", "governance"). Representative strings and their yields are recorded in the Search Documentation sheet of the review database.

### 2.2 Inclusion, exclusion, and scoring

A work was included if it addressed at least one of the five capability dimensions in an energy context, or in a reinforcement-learning context that transfers directly to energy, and if it was published, or issued in the case of standards and regulatory guidance, no later than mid-2026. Core methodological references from outside the energy domain, such as the MAPPO and PettingZoo papers and the Design Science Research literature, were retained as cross-cutting material where they support the proposed artefact. Each record received a relevance score from 1 to 5 and was coded with binary or partial flags indicating whether it addresses MARL, fairness, privacy, XAI, and governance. This coding scheme provides the analytical basis for the synthesis in Section 9.

### 2.3 Corpus composition

The final corpus comprises **105 works** in six groups: MARL for DR and grid control (25); AI governance, ethics, and regulation (20); privacy and FL (19); XAI in energy (18); fairness and energy justice (16); and cross-cutting methodological work (7). Counted by dimension, 42 works address MARL, 26 address privacy (with a further four addressing it partially), 26 address governance, 25 address fairness, and 24 address XAI. The corpus combines peer-reviewed journal and conference papers with preprints, standards, and regulatory guidance. This mix reflects a fast-moving field in which primary regulation and industry tooling have themselves become objects of study.

---

## 3. Area 1 — MARL for demand response and grid control

The 25 works in this area establish that MARL is technically mature for grid control and residential DR. The recurring architecture is CTDE with on-policy actor–critic learning. Within it, MAPPO stands out as the strongest cooperative baseline, a result carried over from the general MARL literature [#102] and confirmed in grid settings [#1, #7]. Hierarchical and action-space-factorised variants become necessary once the joint action space grows, as in topology control [#1, #7, #19].

On residential DR specifically, the evidence is quantitatively encouraging. Fast-timescale MAPPO with localised rewards delivers frequency support across hundreds of heterogeneous loads [#2]; incentive-based decentralised MARL (MARL-iDR) reduces the peak-to-average ratio by 14.48% while preserving participant privacy through decentralisation [#16]; coordinated urban MARL reports 23.4% peak-demand and 18.7% cost reductions across 1,247 buildings [#13]; and MAPPO-based storage scheduling achieves a 14.68% cost reduction with degradation-aware battery management [#21]. Multi-objective formulations are also beginning to appear. MOPPO improves grid-failure prevention by 30% over single-objective RL [#19], and a PPO-with-Lagrangian DR framework handles multi-stakeholder objectives [#12]. These formulations mark a shift toward the constrained optimisation that fairness and safety will require.

The supporting infrastructure is also well developed: the pandapower library provides the Newton-Raphson power-flow backend and the IEEE-bus test cases [#25]; RL2Grid standardises the benchmarking of RL on Grid2Op [#20]; and graph-based RL improves an agent's perception of network topology [#91]. The range of applications is wide, covering EV-charging networks [#23], peer-to-peer trading under physical voltage constraints [#9], transactive microgrids [#5], industrial and multi-energy parks [#10, #11, #15, #22], and blockchain-integrated residential management [#24].

The surveys within this area are candid about what is missing. Pigott et al.'s survey of MARL for energy networks identifies partial observability, non-stationarity, and scalability as open computational problems, and flags safety, fairness, and governance as unresolved [#8]; an ICT-smart-grid review reaches a similar conclusion and stresses an unmet need for safety guarantees [#14]. Across the area the coded gap is consistent: the recorded limitations read "no fairness or governance components" [#1, #13, #21] and "no governance or privacy layer" [#5, #9]. MARL can meet operational targets, but it does not yet provide the guarantees that its deployment context demands.

---

## 4. Area 2 — Fairness and energy justice

Where Area 1 provides the controller, Area 2 provides the objective that the controller currently ignores. The 16 works here fall into three strands.

The first is energy-justice theory, which frames distributive, procedural, and recognitional justice as the basis for deciding who benefits from DR and who bears its cost [#35, #47, #56, #57, #61]. A recurring empirical concern is that the savings from automated DR flow to households that already own smart appliances and have reliable connectivity, while poorer households are left behind. This is the equity blind spot of efficiency-only optimisation. Studies show that machine-learning systems can introduce or amplify such bias in energy settings [#55], and frameworks now exist that quantify energy justice using operational metrics [#61].

The second strand consists of energy-specific fairness mechanisms, most of which are not learning-based. The Fairness-First DR approach raises the Jain fairness index from 0.65 to 0.92 with only a minor loss of peak reduction [#46]. This is the most frequently cited quantitative fairness result in the corpus and a natural target for the present programme. Equity-aware load shedding integrates an explicit equity indicator into AC optimal power flow [#48]; emergency-DR fairness accounts for the frequency and amount of curtailment per customer [#49]; fairness-promoting PV-curtailment schemes apply the Jain index to voltage regulation [#50]; and a classic demand-side-management analysis quantifies the underlying tension between efficiency and fairness [#59]. These works show how to measure fairness in DR, but they implement it in single-agent optimisers rather than in a multi-agent setting.

The third strand is fair reinforcement learning considered in general terms: a survey taxonomy of fairness definitions and constrained-optimisation methods [#51], welfare-function MORL for Pareto-fair policies [#52], fairness-constrained PPO [#53], and group fairness across multiple tasks [#54]. None of these is energy-specific, and the coded gap for each is recorded as "not energy-specific." The work that understands reinforcement learning does not address DR, and the work that addresses DR fairness does not use reinforcement learning. Only five works address both MARL and fairness [#31, #51–#54], and none of them is a residential-DR controller. This separation is the first instance of the integration gap.

---

## 5. Area 3 — Privacy and federated learning

Residential DR runs on fine-grained consumption data, and the 19 works in this area document both the risk it creates and the available defences. The risk is concrete: high-frequency smart-meter traces can predict household occupancy at scale [#71] and can support the non-intrusive disaggregation of individual appliances [#75], so sharing raw traces, or even model updates, leaks behaviour. Surveys catalogue the resulting attack surface and the standard countermeasures, which include differential privacy (DP), homomorphic encryption (HE), and secure multi-party computation [#62, #66, #72].

Federated learning is the dominant architectural response: it keeps data on the device and shares only model updates. For load forecasting, FL has been shown to match or exceed centralised accuracy, and it substantially reduces privacy leakage [#63, #73, #78]. Personalised and cluster-based variants handle the non-IID load distributions that arise across heterogeneous households [#67, #78], and the cluster-based approach is directly relevant to socio-economically differentiated agents. The privacy mechanisms vary in strength, from adaptive DP with dynamic clipping [#77], to distributed DP combined with secure aggregation for stronger end-to-end guarantees [#79], to HE-based [#74] and blockchain-anchored [#70, #76] designs that add tamper-evident logging of updates.

The works most relevant to this review are the few that federate a reinforcement-learning controller. Personalised federated DRL satisfies local privacy constraints for residential energy management without significant loss of performance [#67]; personalised federated hypernetworks report large gains in microgrid profit and training speed-ups of more than 100× through transfer in a multi-task RL setting [#68]; and federated RL manages residential microgrid clusters privately [#69]. These are the closest prior art to a private MARL-DR controller. Each of them, however, reports privacy qualitatively, as a property of decentralisation, rather than as a calibrated (ε, δ) budget, and none couples the privacy layer to fairness or governance. Nine works address both MARL and privacy, but the combination stops at the label "privacy-preserving". A stated, auditable privacy budget inside a multi-agent DR controller remains unexplored.

---

## 6. Area 4 — Explainable AI in energy

Transparency is both a trust requirement and, under the AI Act, a legal one. The 18 works here show that XAI is well developed for energy prediction and diagnosis but underdeveloped for sequential control. SHAP is the dominant method. It provides feature attribution for load and generation forecasting [#87], for anomaly and cyber-attack detection [#94, #95], for building energy management [#83, #85], and for energy-equity analysis, where it reveals how infrastructure and demography drive outcomes [#58]. Comparative reviews find that SHAP is the most trustworthy attribution method but is also the most computationally expensive, whereas LIME is faster but less faithful. This trade-off matters for real-time DR [#84].

Explainability for reinforcement-learning controllers is much rarer. Feature-importance methods have been applied to single-agent DRL for active corrective control [#89], and PPO combined with LIME yields interpretable decisions for microgrid resilience management [#90]. Both are single-agent, however, and the coded gap for each is recorded as "single-agent; not MARL." A newer line of work proposes large language models that translate model outputs into stakeholder-facing natural-language explanations [#92, #93]; this is attractive for household-facing DR justifications but remains experimental. Building- and forecasting-oriented reviews confirm the breadth of XAI in energy [#81, #86, #96], but they also show that the per-decision explanation of a multi-agent DR policy in real time is largely unaddressed. Only three works combine MARL and XAI, and none does so for residential DR.

---

## 7. Area 5 — AI governance, ethics, and regulation

The 20 works in this area supply the requirements that the other four must satisfy, and they show the gap between concept and implementation most clearly. Domain-specific ethical frameworks for smart-grid AI name transparency, accountability, fairness, and privacy as the governing principles [#26, #27], and they argue that generic AI-ethics guidance is insufficient for power systems [#27]. Multi-phase assessment frameworks treat cybersecurity and ethics as co-equal concerns [#28, #29], and accountability frameworks link AI risks to operational impacts and to the responsible parties [#30]. In almost every case the coded gap is "conceptual; no code-level implementation" [#26, #30]. These frameworks describe what governance should ensure, but they do not specify how to enforce it at runtime.

The regulatory sources make these requirements binding. The EU AI Act classifies grid-facing AI as high-risk and imposes duties of conformity assessment, technical documentation (Annex IV), transparency, and human oversight [#32, #88]; XAI is presented as a principal means of meeting the transparency obligation [#88, #45]. NIS2 adds binding cybersecurity duties and 24-hour incident reporting for energy entities [#42]; ISO/IEC 42001 provides a Plan-Do-Check-Act standard for AI management systems [#41]; Ofgem's guidance sets out four core outcomes for the sector, namely safety, security, fairness, and sustainability [#34]; and model-card tooling provides standardised documentation for compliance [#43]. Work on human-in-the-loop (HITL) systems argues that meaningful human control and override authority are architectural properties that must be designed into the system from the start rather than added afterwards [#39, #40].

Two findings point toward the shape of a solution. First, industry runtime-governance tooling now exists. Policy engines that use YAML, Rego, or Cedar rules deliver deterministic allow-or-deny decisions with tamper-evident audit logs and explicit mapping to the AI Act, at sub-millisecond latency, although this tooling is not energy-specific [#38]. Second, the closest academic precedent to an integrated system couples DRL, FL, and an "ethics verification module" [#31]. That system is single-agent, however, and stops short of a runtime policy-gate-and-audit layer over a multi-agent controller. The enforcement mechanisms and the integration target both already exist, but they have not yet been brought together for MARL-DR.

---

## 8. Cross-cutting methodological foundations

Seven works support the construction and evaluation of any artefact in this space. Design Science Research provides the build-and-evaluate methodology and its five-stage process [#99], and recent work clarifies the distinction between formative and summative artefact validation [#100]. On the engineering side, PettingZoo supplies the standard multi-agent environment API, based on the Agent-Environment-Cycle abstraction [#101]; MAPPO supplies the on-policy cooperative learning algorithm [#102]; and empirical studies of Lagrangian methods in safe RL document their sensitivity to hyperparameters and their convergence behaviour [#103], which is a caution for any constrained formulation. Safe-RL reviews and applications complete this set: taxonomies of constraint-handling methods [#17, #18] and a low-carbon multi-energy scheduler that pairs a security network with Lagrangian relaxation to keep actions feasible [#105]. Together these works define the methodological and implementation basis on which an integrated framework can be built and validated.

---

## 9. Synthesis: the five-dimension governance gap

### 9.1 Individually mature, jointly absent

Coding all 105 works against the five capability dimensions produces a clear pattern, which Table 1 summarises. Each dimension is individually well populated: MARL appears in 42 works, privacy in 26 (with a further four partial), governance in 26, fairness in 25, and XAI in 24. Their pairwise and higher-order intersections, however, fall away sharply. MARL co-occurs with fairness in only five works, none of which is an energy-DR controller; it co-occurs with XAI in three and with governance in three. The fairness-and-privacy pair appears five times, and the full governance quartet of fairness, privacy, XAI, and governance appears in only three works [#26, #27, #31], all of which are conceptual or single-model. One work is coded as addressing all five dimensions [#31], but it is a single-agent DRL and FL system with an ethics-verification module, not a multi-agent architecture with a runtime governance layer. In operational terms, the intersection of all five dimensions within a genuine MARL-DR controller is **empty**.

**Table 1. Dimension coverage across the 105-work corpus.**

| Dimension | Papers addressing | Co-occurs with MARL |
|---|---:|---:|
| MARL | 42 | — |
| Privacy / FL | 26 (+4 partial) | 9 |
| Governance | 26 | 3 |
| Fairness | 25 | 5 |
| XAI | 24 | 3 |
| All five (operational) | 1 (single-agent) | — |

### 9.2 Framework-comparison matrix

Restricting attention to the most representative MARL-DR and fairness frameworks makes the gap concrete. Table 2 sets leading systems against the five dimensions, where Y denotes an explicit component, P a partial or implicit one, and N an absent one. The operationally strong MARL-DR frameworks [#2, #13, #16, #21] score N on fairness, on privacy as a stated budget, on XAI, and on governance. Fairness-aware controllers score Y on fairness but are not learning-based [#46, #49], and fair-RL methods are not energy-specific [#52, #53]. Privacy, where it appears, is mainly a by-product of decentralisation rather than a calibrated budget [#16, #23]. Safety constraints appear in isolation [#105] and are never combined with fairness in a single optimiser. Runtime governance, in the form of policy gates, HITL override, and an immutable audit trail, is absent from every surveyed controller.

**Table 2. Framework comparison (representative subset).**

| Framework | Algorithm | Fair | Priv | XAI | Safe | Gov |
|---|---|:--:|:--:|:--:|:--:|:--:|
| Fast-timescale DR [#2] | MAPPO, localized rewards | N | N | N | N | N |
| MARL-iDR [#16] | Decentralised incentive MARL | N | P | N | N | N |
| Coordinated urban MARL [#13] | MARL, 1,247 buildings | N | N | N | N | N |
| GridLearn [#6] | MARL + power flow | N | P | N | P | N |
| MO-PPO-Lagrangian [#12] | PPO + Lagrangian | N | N | N | P | N |
| CTDE-DDPG EV [#23] | CTDE-DDPG | N | P | N | N | N |
| Fairness-First DR [#46] | Heuristic (non-RL) | Y | N | N | N | N |
| Fair-PPO [#53] | PPO + fairness | Y | N | N | N | N |
| Welfare MORL [#52] | MORL welfare | Y | N | N | N | N |
| PFDRL [#67] | Personalised federated DRL | N | Y | N | N | N |
| Transparent microgrid [#90] | PPO + LIME | N | N | Y | N | N |
| Safe RL low-carbon [#105] | Safe RL + Lagrangian | N | N | N | Y | N |
| Ethical DRL+FL [#31] | Single-agent DRL+FL+ethics | Y | Y | Y | P | Y |
| **Integrated target (this programme)** | **MAPPO + Lagrangian, CTDE + FL/DP + SHAP + runtime governance** | **Y** | **Y** | **Y** | **Y** | **Y** |

### 9.3 Interpretation

Three structural findings follow. First, the division in the field lies between capability and constraint: the more operationally impressive a controller is, the less it tends to measure fairness, to budget privacy, to explain its decisions, or to log for audit. Second, the governance literature specifies the requirements thoroughly but implements them almost nowhere. The frameworks stop at principles, and the one runtime-enforcement toolkit is not energy-specific [#38]. Third, the enabling components now exist independently: CTDE and MAPPO controllers, the Jain index and equity indicators, FL with DP, SHAP and LIME, and YAML or Rego policy engines with audit logs. The open problem is therefore integration rather than invention. This is the space that an integrated, governance-aware MARL-DR framework would occupy, and it supports treating governance as an engineering constraint that is embedded in the learning objective and the runtime rather than reported as a metric after the fact.

---

## 10. Open challenges and future directions

**Constraint integration without instability.** Combining fairness and safety as simultaneous Lagrangian constraints on a MAPPO learner inherits the hyperparameter sensitivity and convergence difficulties documented for Lagrangian safe RL [#103]. Satisfying a Jain-index fairness floor [#46] and hard operational limits [#105] together, within a single multi-agent optimiser, is an open methodological problem.

**Extending calibrated privacy from forecasting to control.** FL for grid forecasting is mature [#63, #73], but private policy learning with a stated (ε, δ) budget is unexplored for MARL-DR, and it is not yet clear whether to perturb the policy updates, the value updates, or both. Distributed-DP-with-secure-aggregation designs [#79] indicate the rigorous target that a laptop-scale simulation must approximate transparently.

**Real-time, per-decision explanation of a multi-agent policy.** XAI in energy is centred on forecasting and diagnosis [#84, #87]; explaining, faithfully and within the required latency, why a particular household was curtailed at a particular moment remains open. SHAP's trade-off between fidelity and latency [#84] and template-based or LLM-based natural-language rendering [#93] are the plausible ingredients.

**Runtime governance as architecture.** Turning conceptual ethical frameworks [#26, #28] into an enforceable layer of policy gates, HITL override, and tamper-evident audit with EU AI Act Annex IV mapping requires importing runtime-security patterns [#38] into the energy domain and connecting them to the controller's decision loop [#39].

**Evaluation, benchmarking, and reproducibility.** Standard benchmarks exist for RL grid control [#20] but not for the governance dimensions. The community lacks shared metrics and datasets for fairness, privacy budget, explanation fidelity, and audit completeness, and DSR-style artefact validation [#100] is rarely applied. Reproducible, seed-controlled evaluation across these axes is itself a contribution the field needs.

---

## 11. Conclusion

Across 105 works, MARL for residential demand response is operationally strong and, on its own terms, mature, and fairness, privacy, explainability, and governance each have substantial and growing literatures. The decisive finding is that these strands remain almost entirely separate. Only one surveyed work nominally spans all five capability dimensions, and it is single-agent and has no runtime governance layer. No work combines a genuine multi-agent DR controller with calibrated differential privacy, a fairness-constrained objective, per-decision explanations, and an enforceable audit trail. As grid-facing AI becomes legally high-risk, this integration gap is no longer only an academic concern. The constituent techniques already exist, and the contribution now required is to compose them so that governance is engineered into the learning objective and the runtime of a MARL demand-response controller. That is the direction this review recommends and the programme it motivates.

---

## References

*Numbered by review-database ID [#N]. Where author names were not disambiguated in the source database they are given as recorded; verify author lists, venues, and DOIs against the primary sources before submission.*
[#1] Wang, X., Wang, B., Du, P. (2023). *Multi-Agent Reinforcement Learning for Power Grid Topology Optimization.* arXiv preprint. https://arxiv.org/abs/2310.02605

[#2] Dobbe, M.R. et al. (2023). *Multi-Agent Reinforcement Learning for Fast-Timescale Demand Response of Residential Loads.* arXiv preprint. https://arxiv.org/abs/2301.02593

[#3] Wan, Z. et al. (2022). *Multi-Agent Deep Reinforcement Learning Approach for Temporally Extended Demand Response.* IEEE Trans. Smart Grid. https://doi.org/10.1109/TSG.2022

[#4] Vazquez-Canteli, J.R., Nagy, Z. (2020). *Demand Response Through Price-Setting Multi-Agent Reinforcement Learning.* ACM BuildSys. https://doi.org/10.1145/3408308

[#5] Nuvoli, A., Oneto, L., Anguita, A. (2023). *MAHTM: A Multi-Agent Framework for Hierarchical Transactive Microgrids.* arXiv preprint. https://arxiv.org/abs/2303.08447

[#6] Pinto, S. et al. (2022). *GridLearn: Multiagent Reinforcement Learning for Grid-Aware Building Energy Management.* Electric Power Systems Research. https://arxiv.org/abs/2110.06396

[#7] Lehna, M., Holzer, C., Scholz, C. (2025). *Centrally Coordinated Multi-Agent Reinforcement Learning for Power Grid Topology Control.* ACM e-Energy. https://arxiv.org/abs/2502.08681

[#8] Pigott, C. et al. (2024). *Multi-Agent Reinforcement Learning for Energy Networks: Computational Challenges, Progress and Open Problems.* arXiv preprint. https://arxiv.org/abs/2404.15583

[#9] Qiu, Y., Yang, Y., Lan, T. (2024). *Peer-to-Peer Energy Trading of Solar and Energy Storage: A Networked Multiagent RL Approach.* arXiv preprint. https://arxiv.org/abs/2401.13947

[#10] Ghasempour, A. (2020). *Smart Grid for Industry Using Multi-Agent Reinforcement Learning.* Applied Sciences. https://doi.org/10.3390/app10196900

[#11] Ma, Y. et al. (2022). *Energy Management Based on Multi-Agent Deep RL for a Multi-Energy Industrial Park.* arXiv preprint. https://arxiv.org/abs/2202.03771

[#12] Singh, S., Kumar, A. (2025). *AI-Powered Demand Response Optimization in Smart Grids: A Multi-Objective RL Framework.* IJSRED. https://doi.org/10.17577/IJSRED

[#13] Olusegun, A., Smith, E. (2025). *Multi-Agent Reinforcement Learning for Coordinated Smart Grid and Demand Response.* Computer Life Journal. https://computer-life.org/index.php/ojs/article/view/22

[#14] Shawon, M.R. et al. (2019). *Multi-Agent Systems in ICT-Enabled Smart Grids: A Review.* IEEE Access. https://doi.org/10.1109/ACCESS

[#15] Zhang, Z. et al. (2025). *Multi-Agent Deep RL Based Demand Response for Discrete Manufacturing Systems.* Applied Energy. https://doi.org/10.1016/j.apenergy

[#16] Roesch, M. et al. (2023). *MARL-iDR: Multi-Agent Reinforcement Learning for Incentive-based Residential Demand Response.* arXiv preprint. https://arxiv.org/abs/2304.04086

[#17] Review Authors (2024). *A Review of Safe Reinforcement Learning Methods for Modern Power Systems.* arXiv preprint. https://arxiv.org/abs/2407.00304

[#18] Review Authors (2024). *Safe Reinforcement Learning for Power System Control: A Review.* arXiv preprint. https://arxiv.org/abs/2407.00681

[#19] Multi-Objective RL Authors (2025). *Multi-Objective Reinforcement Learning for Power Grid Topology Control.* arXiv preprint. https://arxiv.org/abs/2502.00040

[#20] RL2Grid Authors (2025). *RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations.* OpenReview. https://arxiv.org/abs/2503.23101

[#21] MARL Battery Authors (2025). *Multi-Agent Deep RL for Scheduling of Energy Storage System in Microgrids.* Mathematics (MDPI). https://doi.org/10.3390/math13121999

[#22] AI-Powered Microgrid Authors (2024). *AI-Powered Microgrid Networks: Multi-Agent Deep RL for Optimized Energy Trading.* Arabian J. Sci. Eng.. https://doi.org/10.1007/s13369-024-09754-4

[#23] CTDE EV Authors (2024). *Centralized vs. Decentralized MARL for Enhanced Control of EV Charging Networks.* arXiv preprint. https://arxiv.org/abs/2404.12520

[#24] Decentralized REM Authors (2024). *Multi-agent-based Decentralized Residential Energy Management Using Deep RL.* J. Building Engineering. https://doi.org/10.1016/j.jobe.2024

[#25] Thurner, L. et al. (2018). *pandapower — An Open-Source Python Tool for Power System Analysis.* IEEE Trans. Power Systems. https://arxiv.org/abs/1709.06743

[#26] Ahmad, T. et al. (2024). *Ethical Considerations in AI Applications in Smart Grid.* Preprints/Zayed Univ.. https://doi.org/10.20944/preprints

[#27] Ahmad, T. et al. (2024). *Ethical Considerations in Artificial Intelligence Applications in Smart Grid.* Zayed University.

[#28] Jiang, Z. et al. (2025). *An Ethical Assessment Framework for AI Security and Ethics in Smart Grid.* Eng. Appl. of AI. https://doi.org/10.1016/j.engappai

[#29] Brunel Authors (2023). *An Ethical Assessment Framework for AI Security and Ethics in Smart Grids.* Brunel University Tech. Rep..

[#30] Buchner, E., Hühne, C. (2024). *Being Accountable is Smart: Navigating the Technical and Regulatory Challenges of AI in Smart Grids.* arXiv preprint. https://arxiv.org/abs/2408.01121

[#31] Hassan, M. et al. (2025). *Design of Ethical AI Frameworks for Sustainable and Adaptive Energy Management Systems.* IJESTY. https://doi.org/10.26555/ijesty

[#32] Sohil, S. et al. (2024). *The AI Act: Shaping Europe's Digital Future and Transforming the Energy Sector.* Energy and Env. Policy.

[#33] The Thinking Company (2026). *AI Governance in Energy & Utilities 2026.* Industry Report.

[#34] Ofgem (2025). *Ethical AI Use in the Energy Sector (OFG1164).* UK Regulatory Guidance. https://www.ofgem.gov.uk

[#35] European Commission JRC (2023). *Energy Justice for AI: A Framework.* JRC Technical Report.

[#36] Trustworthy AI Authors (2024). *Trustworthy AI in the Energy Sector: Landscape Analysis and Evaluation Framework.* arXiv preprint. https://arxiv.org/abs/2412.07782

[#37] XAI Governance Authors (2023). *AI Explainability and Governance in Smart Energy Systems: A Review.* Frontiers in Energy Research. https://doi.org/10.3389/fenrg.2023.1071291

[#38] Microsoft (2026). *Agent Governance Toolkit: Open-source Runtime Security for AI Agents.* Microsoft Open Source Blog. https://opensource.microsoft.com/blog/2026/04/02/introducing-the-agent-governance-toolkit

[#39] Human-AI Authors (2025). *Human-AI Interaction in Safety-Critical Network Infrastructures.* ScienceDirect. https://doi.org/10.1016/j.comnet

[#40] HITL Energy Authors (2026). *Human-in-the-Loop AI in the Energy Sector: A Systematic Review of Paradigm Shifts toward Industry 5.0.* ScienceDirect. https://doi.org/10.1016/j.seta

[#41] ISO/IEC (2023). *ISO/IEC 42001:2023 — AI Management Systems.* ISO Standard. https://www.iso.org/standard/42001

[#42] EU NIS2 Directive (2022). *NIS2 Directive: Cybersecurity Requirements for Critical Infrastructure.* EU Directive. https://digital-strategy.ec.europa.eu

[#43] Model Card Regulatory Authors (2025). *Model Card Regulatory Check.* OECD.AI. https://oecd.ai/en/catalogue/tools/model-card-regulatory-check

[#44] Ethical Smart Grid Authors (2024). *Ethical Considerations in Smart Grid Optimization: Promoting Energy Equity and Fairness.* Preprints.org / ITPHSS. https://doi.org/10.20944/preprints202412.0644

[#45] XAI Compliance Authors (2025). *The Bidirectional Relationship Between XAI and Regulation: Operationalizing XAI for the AI Act.* arXiv preprint. https://arxiv.org/abs/2505.20311

[#46] Bhatt, A. et al. (2026). *Bridging Equity and Efficiency in Smart Energy Systems: A Fairness-First Demand Response Algorithm.* O.P. Jindal Global Univ.. https://pure.jgu.edu.in/id/eprint/10953/

[#47] Prism Sustainability (2025). *Algorithmic Energy Justice.* Prism Sustainability Directory.

[#48] Equity Load Shedding Authors (2024). *Equity-aware Load Shedding Optimization.* arXiv preprint. https://arxiv.org/abs/2406.17877

[#49] Fairness Emergency DR Authors (2021). *Fairness Algorithm for Emergency Demand Response Operation in Distribution Networks.* Int. J. Electrical Power. https://doi.org/10.1016/j.ijepes

[#50] PV Curtailment Authors (2024). *Analysis of Fairness-promoting Optimization Schemes of PV Curtailments for Voltage Regulation.* arXiv preprint. https://arxiv.org/abs/2404.00394

[#51] Fairness RL Survey Authors (2024). *Fairness in Reinforcement Learning: A Survey.* arXiv preprint. https://arxiv.org/abs/2405.06909

[#52] MORL Fairness Authors (2022). *Welfare and Fairness in Multi-objective Reinforcement Learning.* arXiv preprint. https://arxiv.org/abs/2212.01382

[#53] Fair PPO Authors (2025). *Fairness Aware Reinforcement Learning via Proximal Policy Optimization.* arXiv preprint. https://arxiv.org/abs/2502.03953

[#54] Group Fair MTRL Authors (2025). *Group Fairness in Multi-Task Reinforcement Learning.* arXiv preprint. https://arxiv.org/abs/2503.07817

[#55] ML Bias Energy Authors (2024). *Addressing Machine Learning Bias to Foster Energy Justice.* Energy Research & Social Science. https://doi.org/10.1016/j.erss

[#56] Just AI Transition Authors (2024). *Towards a Just AI-Assisted Energy Transitions for Vulnerable Communities.* Energy Research & Social Science. https://doi.org/10.1016/j.erss

[#57] AI Energy Justice Authors (2026). *Artificial Intelligence Adoption for Advancing Energy Justice: A Multidimensional Perspective.* Scientific Reports. https://doi.org/10.1038/s41598-026-38421-x

[#58] Energy Equity XAI Authors (2025). *Energy Equity, Infrastructure and Demographic Analysis with XAI Methods.* arXiv preprint. https://arxiv.org/abs/2509.16279

[#59] DR Fairness Tradeoff Authors (2017). *Demand Side Management in the Smart Grid: An Efficiency and Fairness Tradeoff.* arXiv preprint. https://arxiv.org/abs/1711.11129

[#60] Fair Load Shedding ML Authors (2024). *Machine Learning for Fairness-Aware Load Shedding: A Real-Time Solution.* arXiv preprint. https://arxiv.org/abs/2407.18989

[#61] Energy Justice Framework Authors (2025). *Measuring the Invisible: A Framework for Quantifying Energy Justice in Urban Communities.* Energy Research & Social Science. https://doi.org/10.1016/j.erss

[#62] Chen, L. et al. (2024). *Federated Learning for Smart Grid: A Survey on Applications and Privacy Challenges.* arXiv preprint. https://arxiv.org/abs/2409.10764

[#63] Rafique, A. et al. (2025). *Federated Learning-Based Energy Management Systems for Privacy-Preserving Demand Forecasting in Smart Cities.* IEEE Access. https://doi.org/10.1109/ACCESS

[#64] Li, J. et al. (2025). *Federated Learning Architectures for Privacy-Preserving Smart Grid Data Processing.* IJESTY. https://doi.org/10.26555/ijesty

[#65] Saad, W. et al. (2021). *A Federated Learning Framework for Smart Grids: Securing Power Traces.* arXiv preprint. https://arxiv.org/abs/2103.11870

[#66] Taïk, S., Cherkaoui, S. (2023). *Crossing Roads of Federated Learning and Smart Grids: Overview, Challenges, and Perspectives.* arXiv preprint. https://arxiv.org/abs/2304.08602

[#67] Kim, D., Moon, J. (2023). *PFDRL: Personalized Federated Deep Reinforcement Learning for Residential Energy Management.* ACM BuildSys. https://doi.org/10.1145/3600100

[#68] PFH Authors (2022). *Personalized Federated Hypernetworks for Privacy Preservation in Multi-Task Reinforcement Learning.* arXiv preprint. https://arxiv.org/abs/2210.06820

[#69] FL-RL Microgrid Authors (2024). *Federated Reinforcement Learning for Smart and Privacy-Preserving Energy Management of Residential Microgrids Clusters.* Eng. Appl. of AI. https://doi.org/10.1016/j.engappai

[#70] Ibrahim, M. et al. (2026). *Towards a Cybersecure and Privacy Enhanced Smart Grid: A Blockchain Enabled FL Framework.* PLOS ONE. https://doi.org/10.1371/journal.pone.0342454

[#71] Occupancy Detection Authors (2019). *Occupancy Detection of Residential Buildings Using Smart Meter Data: A Large-Scale Study.* Energy & Buildings. https://doi.org/10.1016/j.enbuild

[#72] Privacy Smart Meters Authors (2023). *Privacy Preservation in Smart Meters: Current Status, Challenges and Future Directions.* Sensors. https://doi.org/10.3390/s23073697

[#73] FL LSTM Authors (2025). *A Federated LSTM Network for Load Forecasting.* AIMS Energy.

[#74] FL IoT Authors (2025). *Smart Grid IoT Framework for Predicting Energy Consumption Using FL Homomorphic Encryption.* Sensors (MDPI). https://doi.org/10.3390/s25123700

[#75] FL Energy Disaggregation Authors (2025). *Privacy Preserving Federated Learning for Energy Disaggregation of Smart Homes.* IET Cyber-Physical Systems. https://doi.org/10.1049/cps2.70013

[#76] Blockchain FL Power Authors (2024). *A Blockchain-based Framework for Federated Learning with Privacy Preservation in Power Load Forecasting.* Knowledge-Based Systems. https://doi.org/10.1016/j.knosys

[#77] Adaptive DP FL Authors (2025). *Privacy-Preserving Decentralized FL via Explainable Adaptive Differential Privacy.* arXiv preprint. https://arxiv.org/abs/2509.10691

[#78] FL Personalized Grid Authors (2024). *Electrical Load Forecasting in Smart Grid: A Personalized Federated Learning Approach.* arXiv preprint. https://arxiv.org/abs/2411.10619

[#79] DDP-SA Authors (2025). *DDP-SA: Scalable Privacy-Preserving FL via Distributed DP and Secure Aggregation.* arXiv preprint. https://arxiv.org/abs/2604.07125

[#80] FL DRL Buildings Authors (2021). *Privacy-Preserving Energy Management of a Shared Energy Storage System: A Federated Deep RL Approach.* PMC/MDPI. https://pmc.ncbi.nlm.nih.gov/articles/PMC8309780/

[#81] Machlev, R. et al. (2025). *Applications of Explainable AI and Interpretable AI in Smart Buildings and Energy Savings: A Systematic Review.* Charles Darwin Univ..

[#82] Patsnap Eureka (2025). *Explainable AI (XAI) for Transparent Grid Decisions.* Patsnap Eureka.

[#83] XAI Energy Framework Authors (2025). *Explainable AI Framework for Reliable and Transparent Automated Energy Management in Buildings.* Energy & Buildings. https://doi.org/10.1016/j.enbuild

[#84] XAI Maintenance Authors (2025). *Explainable AI for Energy Systems Maintenance: A Review.* Renewable & Sustainable Energy Reviews. https://doi.org/10.1016/j.rser

[#85] XAI Smart Buildings Authors (2025). *Evolution of Building Energy Management Systems for Greater Sustainability through Explainable AI Models.* Eng. Appl. of AI. https://doi.org/10.1016/j.engappai

[#86] XAI Urban Buildings Authors (2025). *A Systematic Review of XAI in Urban Building Energy Modeling.* Sustainable Cities and Society. https://doi.org/10.1016/j.scs

[#87] XAI Energy Forecasting Authors (2025). *A Four-Dimensional Analysis of XAI in Energy Forecasting: A Domain-Specific Systematic Review.* Machine Learning and Knowledge Extraction. https://doi.org/10.3390/make7040153

[#88] Leofante et al. (2025). *Explainable AI, Energy and Critical Infrastructure Systems.* AI Magazine. https://doi.org/10.1002/aaai.70033

[#89] Explainable RL Grid Authors (2024). *Data-driven Active Corrective Control in Power Systems: An Interpretable Deep RL Approach.* Frontiers in Energy Research. https://doi.org/10.3389/fenrg.2024.1389196

[#90] Transparent Microgrid Authors (2025). *Deep RL with Local Interpretability for Transparent Microgrid Resilience Energy Management.* arXiv preprint. https://arxiv.org/abs/2508.08132

[#91] Graph RL Grid Authors (2024). *Graph Reinforcement Learning in Power Grids: A Survey.* arXiv preprint. https://arxiv.org/abs/2407.04522

[#92] LLM Energy Survey Authors (2025). *A Comprehensive Survey of LLMs for Sustainable and Renewable Energy Systems.* Information (MDPI). https://doi.org/10.3390/info17030271

[#93] RE-LLM Authors (2025). *RE-LLM: Integrating Large Language Models into Renewable Energy Systems.* arXiv preprint. https://arxiv.org/abs/2512.01392

[#94] SHAP Grid Anomaly Authors (2026). *Detection of Disturbances and Cyber-attacks in Smart Grids Using Explainable ML.* Scientific Reports. https://doi.org/10.1038/s41598-026-35449-x

[#95] SHAP Energy Anomaly Authors (2025). *Explaining Deep Learning-based Anomaly Detection in Energy Consumption Data.* arXiv preprint. https://arxiv.org/abs/2501.06099

[#96] XAI Renewable Review Authors (2024). *Advancing Renewable Energy Systems through Explainable AI: A Comprehensive Review.* J. Computer Science and Technology Studies.

[#97] Adversarial Grid RL Authors (2024). *Attacking Learning-based Models in Smart Grids: Current Challenges and New Frontiers.* ACM ECAI. https://doi.org/10.1145/3632775

[#98] Adversarial RL Power Authors (2021). *Improving Robustness of RL for Power System Control with Adversarial Training.* arXiv preprint. https://arxiv.org/abs/2110.08956

[#99] Hevner, A. et al. (2004). *Design Science in Information Systems Research.* MIS Quarterly.

[#100] DSR Artefact Authors (2025). *Artifact Validity in Design Science Research: A Comparative Analysis of Three Frameworks.* DESRIST 2025. https://arxiv.org/abs/2502.11199

[#101] Terry, J.K. et al. (2021). *PettingZoo: Gym for Multi-Agent Reinforcement Learning.* NeurIPS. https://arxiv.org/abs/2009.14471

[#102] Yu, C. et al. (2022). *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.* NeurIPS.

[#103] Lagrangian RL Empirical Authors (2025). *An Empirical Study of Lagrangian Methods in Safe Reinforcement Learning.* arXiv preprint. https://arxiv.org/abs/2510.17564

[#104] Cyber Resilience Grid Authors (2025). *Cybersecurity in Smart Microgrids Using Blockchain-Federated Learning and Quantum-Safe Approaches.* Applied Energy. https://doi.org/10.1016/j.apenergy

[#105] Safe RL Low-Carbon Authors (2024). *Safe Reinforcement Learning Based Optimal Low-Carbon Scheduling Strategy for Multi-Energy System.* Int. J. Electrical Power. https://doi.org/10.1016/j.ijepes
