# BHOOmiKAVACH AI — LangGraph Multi-Agent Architecture

## 1. Overview
The Agentic AI layer is implemented using **LangGraph**, providing a cyclic, stateful decision-support workflow for underground mine safety. The graph coordinates eight specialized agents to evaluate prospective mine risks, query official DGMS accident records, generate SHAP attributions, retrieve regulatory citations, formulate role-based protocols, and enforce safety policies.

---

## 2. Agent Graph Workflow Diagram

```
                              [USER INQUIRY / TELEMETRY]
                                           │
                                           ▼
                                    ┌──────────────┐
                                    │ ROUTER AGENT │
                                    └──────┬───────┘
                                           │ (Classifies Intent)
               ┌───────────────────────────┼───────────────────────────┐
               ▼                           ▼                           ▼
      ┌─────────────────┐         ┌─────────────────┐         ┌─────────────────┐
      │ HISTORICAL AGNT │         │ SIMILARITY AGNT │         │   RISK AGENT    │
      │(SQLite Analytics│         │(FAISS Precedents│         │ (ML & Agreement)│
      └────────┬────────┘         └────────┬────────┘         └────────┬────────┘
               │                           │                           │
               │                           │                           ▼
               │                           │                  ┌─────────────────┐
               │                           │                  │EXPLANATION AGENT│
               │                           │                  │(SHAP Attribution│
               │                           │                  └────────┬────────┘
               │                           │                           │
               │                           └───────────┬───────────────┘
               │                                       ▼
               │                              ┌─────────────────┐
               │                              │ EVIDENCE AGENT  │
               │                              │  (FAISS RAG)    │
               │                              └────────┬────────┘
               │                                       │
               │                                       ▼
               │                              ┌─────────────────┐
               │                              │ RECOMMENDATION  │
               │                              │ (7 Mining Roles)│
               │                              └────────┬────────┘
               │                                       │
               └───────────────────┬───────────────────┘
                                   ▼
                            ┌──────────────┐
                            │ CRITIC AGENT │ (Policy Verification & Audit)
                            └──────┬───────┘
                                   │
                                   ▼
                         [DECISION-SUPPORT PACKET]
```

---

## 3. Specialized Sub-Agents

1. **Router Agent (`router_node`):** Classifies the natural language query into 7 intents: `risk_assessment`, `risk_explanation`, `historical_query`, `similarity_search`, `recommendation`, `rag_evidence`, or `general_query`.
2. **Historical Analytics Agent (`historical_node`):** Queries SQLite database `bhoomikavach.db` to extract exact, unhallucinated DGMS statistics, fatality distributions by sector, and regional patterns.
3. **Similarity Agent (`similarity_node`):** Executes semantic cosine similarity over 41 primary fatal accidents using FAISS, retrieving historical incidents matching the scenario.
4. **Risk Assessment Agent (`risk_node`):** Invokes the 84-feature preprocessing pipeline, checks physical boundaries via the OOD detector, evaluates XGBoost/LightGBM/CatBoost, calculates the multi-model agreement consensus, and applies deterministic statutory rules.
5. **Explanation Agent (`explanation_node`):** Uses SHAP TreeExplainer to determine the top 5 directional feature contributors shifting the model toward the predicted hazard level.
6. **Evidence Agent (`evidence_node`):** Retrieves exact sections from Coal Mines Regulations 2017, Metalliferous Mines Regulations 1961, and DGMS Technical Circulars via the local FAISS RAG index.
7. **Recommendation Agent (`recommendation_node`):** Stratifies immediate, engineering, and monitoring actions tailored to the user's specific statutory role (Worker, Supervisor, Safety Officer, Mining Engineer, Geotechnical Engineer, Mine Manager, Executive).
8. **Critic Agent (`critic_node`):** Enforces the BHOOmiKAVACH Safety Policy: validates zero hallucination, strips uncalibrated claims, verifies non-causal language, formats the final response, and appends statutory safety disclaimers.

---

## 4. Local LLM Strategy (Ollama & Fallback)
The system checks for local Ollama server availability (`http://localhost:11434`). If Ollama is offline or uninstalled, the Critic Agent seamlessly falls back to rigorous, pre-compiled deterministic synthesis templates, guaranteeing that the decision-support system **NEVER crashes**.
