# BHOOmiKAVACH AI — System Architecture Specification

## 1. Executive Summary
**BHOOmiKAVACH AI** is an explainable AI and multi-agent decision-support platform engineered specifically for **DGMS Code 0111: Fall of Roof** hazard mitigation across Indian underground coal and metalliferous mines. The platform operates on two segregated datasets:
- **Dataset A (Historical Intelligence):** 41 fatal roof fall inquiry records (2016–2022) extracted from official DGMS SANKET publications with zero data hallucination.
- **Dataset B (Operational Prospective Risk):** 1,500 chronologically sequenced mine shift inspection records grounded in CMRI-ISM Rock Mass Rating (RMR) and statutory parameters (CMR 2017 / MMR 1961).

---

## 2. Multi-Tier System Architecture

```
                                  MINE OPERATIONAL TELEMETRY & INSPECTIONS
                                                    │
                                                    ▼
                                          FASTAPI REST BACKEND
                                          (Port 8000 /api/v1)
                                                    │
                ┌───────────────────────────────────┼───────────────────────────────────┐
                ▼                                   ▼                                   ▼
        ISOLATION FOREST OOD               PREPROCESSING PIPELINE             SAFETY RULE ENGINE
      (Physical Bounds & Envelopes)          (84 Encoded Features)             (Deterministic CMR/MMR)
                │                                   │                                   │
                └─────────────────┬─────────────────┘                                   │
                                  ▼                                                     │
                     GRADIENT BOOSTED CLASSIFIERS                                       │
                ┌─────────────────┼─────────────────┐                                   │
                ▼                 ▼                 ▼                                   │
             XGBoost           LightGBM          CatBoost                               │
                └─────────────────┼─────────────────┘                                   │
                                  ▼                                                     │
                       SOFT & WEIGHTED ENSEMBLES                                        │
                                  │                                                     │
                                  ▼                                                     │
                        MODEL AGREEMENT ENGINE                                          │
                         (Consensus & Uncertainty)                                      │
                                  │                                                     │
                                  ▼                                                     │
                         SHAP TREE EXPLAINER                                            │
                        (Local Feature Weights)                                         │
                                  │                                                     │
                                  ▼                                                     │
                     LANGGRAPH MULTI-AGENT ORCHESTRATOR ◄───────────────────────────────┘
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
          FAISS RAG            FAISS CASE         ROLE-BASED ACTIONS
        (Regulations)         SIMILARITY          (7 Statutory Roles)
                └─────────────────┼─────────────────┘
                                  ▼
                             CRITIC AGENT
                       (Safety Policy & Audit)
                                  │
                                  ▼
                     STREAMLIT DECISION SUPPORT UI
```

---

## 3. Subsystem Breakdown

### 3.1 Preprocessing & Validation
- **Temporal Splitting:** Data is strictly ordered chronologically: Train (70%), Validation (15%), Test (15%). Prevents temporal information leakage.
- **Feature Space:** 84 dimensions combining geotechnical ratings (RMR, Q, GSI, RQD), excavation geometry (gallery width, height), support condition, blasting vibration, and convergence dynamics.

### 3.2 Machine Learning & Ensembles
- **Classifiers:** Baseline Logistic Regression, XGBoost (`XGBClassifier`), LightGBM (`LGBMClassifier`), CatBoost (`CatBoostClassifier`).
- **Ensemble Schemes:** Soft Voting, Weighted Soft Voting (weights optimized strictly on validation data), and Stacking Classifier.
- **Champion Selection:** Ranked via composite safety score prioritizing HIGH and CRITICAL hazard sensitivity and minimizing catastrophic False Negatives.

### 3.3 Interpretability & Agreement
- **SHAP TreeExplainer:** Computes local feature attribution for each prediction. Directional language is strictly non-causal ("contributed to the prediction score", NOT "caused the incident").
- **Model Agreement Metric:** Calculates agreement ratio (e.g. 3/3, 2/3) across gradient boosted models as an operational uncertainty indicator.

### 3.4 Geotechnical Safety Rule Engine
- Deterministic rules enforcing Coal Mines Regulations (CMR 2017 Reg 111, 112, 123, 125, 129), MMR 1961 Reg 112, and DGMS Circulars (SCAMP Circular 3/2010, Bolt Testing Circular 6/2013). Operates independently of ML models.

### 3.5 Out-of-Distribution (OOD) Detector
- Combines physical bounding envelopes with an Isolation Forest fitted on the training distribution. Abnormal sensor values or unseen minerals trigger confidence penalties and technical assessment notices.

### 3.6 Local RAG & Similarity Search
- **FAISS Flat Inner Product Index:** Normalized cosine similarity over SentenceTransformer (`all-MiniLM-L6-v2`) embeddings.
- **Knowledge Base:** Curates official regulations, technical circulars, safety alerts, and verbatim historical DGMS enquiry precedents.

### 3.7 LangGraph Multi-Agent Workflow
- State-managed execution through Router -> Risk/Historical/Similarity -> Explanation -> Evidence -> Recommendation -> Critic -> Response.
- Supported with automated SQLite audit logging across all decisions.
