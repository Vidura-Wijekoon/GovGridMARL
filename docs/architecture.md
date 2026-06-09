# Architecture

GovGridMARL couples a **multi-agent RL core** with a **governance layer**.

```
                 +-------------------- governance layer --------------------+
                 |  audit log (hash-chained)   guardrails   model card      |
                 |  SHAP explanations          Annex IV mapping (EU AI Act) |
                 +----------------------------------------------------------+
                                   ^            |
                          decisions|            |veto / log
                                   |            v
  load data ---> [ GridMARLEnv ] <--> [ agents: MAPPO / IPPO / MOPPO ] 
   (REDD/UK-DALE   PettingZoo over        constrained via Lagrangian /
    + synthetic)   pandapower IEEE-14      PID-Lagrangian (safety),
                                           fairness-shaped reward (Jain)
                                   ^
                                   | federated rounds + DP noise
                          [ FedAvg + DP-SGD + privacy accountant ]
```

## Components

- **Environment** (`envs/govgrid_envs`): PettingZoo `ParallelEnv` over an
  IEEE bus system simulated with pandapower; household / aggregator / DSO agents.
- **Agents** (`govgridmarl.agents`): PPO, IPPO, MAPPO (baseline), MOPPO and
  Lagrangian-constrained variants.
- **Fairness** (`govgridmarl.fairness`): Jain index and fairness reward terms.
- **Safety** (`govgridmarl.safety`): constraint costs + runtime guardrails.
- **Privacy** (`govgridmarl.privacy`): FedAvg aggregation + DP-SGD accounting.
- **Explainability** (`govgridmarl.explainability`): SHAP attributions.
- **Governance** (`govgridmarl.governance`): tamper-evident audit log, model
  card, EU AI Act Annex IV mapping.
