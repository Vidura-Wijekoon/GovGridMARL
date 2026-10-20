"""GovGridMARL: Governance-aware Multi-Agent Reinforcement Learning for
residential smart-grid demand response.

A research framework that couples constrained multi-agent RL (MAPPO and
variants) on power-grid environments (IEEE-14 / IEEE-118 via pandapower &
PettingZoo) with a governance layer: fairness (Jain index), safety
(constraint costs / Lagrangian methods), privacy (federated learning +
differential privacy), explainability (SHAP) and audit logging mapped to
the EU AI Act.

Subpackages
-----------
agents          PPO / IPPO / MAPPO / MOPPO / Lagrangian-constrained learners.
fairness        Equity metrics and fairness-aware reward shaping.
safety          Safety constraints, costs, and runtime guardrails.
privacy         Federated aggregation (FedAvg) and differential privacy.
explainability  SHAP/LIME attribution for agent decisions.
governance      Tamper-evident audit logging, model cards, Annex IV mapping.
data            Dataset loaders (REDD/UK-DALE) and synthetic load profiles.
training        Training loops and orchestration.
evaluation      Metrics and evaluation protocol.
utils           Seeding, logging, and shared helpers.
"""

__version__ = "0.3.1"
__author__ = "Vidura"

__all__ = ["__version__", "__author__"]
