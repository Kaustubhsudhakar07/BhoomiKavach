# BHOOmiKAVACH AI
### An Explainable AI and Agentic Decision-Support System for Roof-Fall Risk Assessment and Prevention in Indian Mines

> **"SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL."**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangGraph](https://img.shields.io/badge/Agentic%20AI-LangGraph-FF6F00.svg)](https://langchain-ai.github.io/langgraph/)
[![FAISS](https://img.shields.io/badge/Vector%20Store-FAISS-0A84FF.svg)](https://github.com/facebookresearch/faiss)
[![Tests Passing](https://img.shields.io/badge/Tests-24%2F24%20Passing-brightgreen.svg)]()
[![DGMS Standard](https://img.shields.io/badge/DGMS%20Code-0111%20Fall%20of%20Roof-orange.svg)]()

---

## 1. System Architecture Diagram (Phase 37)

```
                                      USER
                                        │
                                        ▼
                                    STREAMLIT
                                  (Frontend UI)
                                        │
                                        ▼
                                     FASTAPI
                                  (REST Backend)
                                        │
                                        ▼
                             LANGGRAPH ORCHESTRATOR
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
                RAG Agent           Risk Agent          Data Agent
                    │                   │                   │
                    ▼                   ▼                   ▼
                  FAISS              ML Models           SQLite
               (Vector DB)              │            (bhoomikavach.db)
                               ┌────────┼────────┐
                               ▼        ▼        ▼
                            XGBoost  LightGBM CatBoost
                               \        │       /
                                \       │      /
                                 ┌──────┴──────┐
                                        │
                                    ENSEMBLE
                               (Weighted Soft Vote)
                                        │
                                        ▼
                                      SHAP
                                (TreeExplainer)
                                        │
                                        ▼
                                   RULE ENGINE
                                (CMR 2017 / MMR)
                                        │
                                        ▼
                                   EVIDENCE/RAG
                                (DGMS Circulars)
                                        │
                                        ▼
                              RECOMMENDATION AGENT
                               (7 Statutory Roles)
                                        │
                                        ▼
                                  CRITIC AGENT
                              (Safety Guardrails)
                                        │
                                        ▼
                                   FINAL ANSWER
                                        │
                                        ▼
                                    STREAMLIT
```

---

## 2. Project Overview & Scope
Roof fall (**DGMS Accident Code 0111**, Major Cause `0100 Ground Movement`) is the single largest cause of fatalities and serious injuries in underground Indian coal and metalliferous mines.

**BHOOmiKAVACH AI** is an IIT/DGMS-grade industrial decision-support system that bridges realized historical failure intelligence with live prospective machine learning and multi-agent decision support:

1. **Dataset A (Historical Intelligence):** 41 verified fatal roof-fall enquiry records (2016–2022) extracted from official DGMS SANKET reports with strictly zero data hallucination.
2. **Dataset B (Operational Telemetry):** 1,500 chronologically sequenced mine shift inspection records grounded in CMRI-ISM Rock Mass Rating (RMR) and statutory parameters.
3. **Machine Learning Ensembles:** Gradient boosted classifiers (XGBoost, LightGBM, CatBoost) achieving **80.8% recall on HIGH hazards** and **55.6% sensitivity on CRITICAL failures** under zero-leakage chronological time-series splitting.
4. **Explainable AI (XAI):** SHAP feature attribution providing local directional explanations without making false causal claims.
5. **Model Agreement Metric:** Real-time multi-model consensus ratio (e.g. 3/3 vs 2/3) quantifying epistemic model uncertainty.
6. **Deterministic Safety Rule Engine:** Statutory checks enforcing Coal Mines Regulations (CMR 2017 Reg 111, 112, 123, 125, 129) and DGMS Technical Circulars.
7. **Local FAISS RAG:** Dense semantic retrieval over regulations, circulars, and historical inquiry precedents using `all-MiniLM-L6-v2`.
8. **LangGraph Agentic Decision Support:** Cyclic multi-agent graph providing role-stratified safety actions across 7 mining designations.

---

## 3. Project Structure (Phase 29)

```
BhoomiRakshak/
├── backend/
│   └── app/
│       ├── main.py                     # FastAPI Application Entrypoint
│       ├── api/
│       │   └── routes/                 # REST Route Modules
│       │       ├── health.py           # System Diagnostics
│       │       ├── accidents.py        # Historical Accidents API
│       │       ├── predict.py          # Live Prediction & Explainability
│       │       ├── rag.py              # Knowledge Base RAG
│       │       ├── agent.py            # LangGraph Chat API
│       │       └── audit.py            # Compliance Audit Trail
│       ├── agents/                     # LangGraph Multi-Agent System
│       │   ├── state.py                # AgentState TypedDict
│       │   ├── router_agent.py         # Intent Router
│       │   ├── historical_agent.py     # SQLite Analytics Agent
│       │   ├── similarity_agent.py     # FAISS Precedent Matcher
│       │   ├── risk_agent.py           # ML & Agreement Agent
│       │   ├── explanation_agent.py    # SHAP Attribution Agent
│       │   ├── evidence_agent.py       # Regulatory Evidence Agent
│       │   ├── recommendation_agent.py # Role-Based Action Agent
│       │   ├── critic_agent.py         # Safety Policy Guardrail Agent
│       │   └── graph.py                # StateGraph Compilation
│       ├── core/
│       │   └── config.py               # Settings & Constants
│       └── database/
│           └── session.py              # SQLite Engine & Audit Logger
│
├── frontend/
│   ├── app.py                          # Streamlit Industrial Dashboard
│   └── api_client.py                   # Clean HTTP Client
│
├── data/
│   ├── raw/                            # Official DGMS SANKET PDFs
│   ├── processed/                      # Canonical CSVs & SQLite DB (bhoomikavach.db)
│   └── operational/                    # Operational Dataset (Dataset B)
│
├── src/
│   ├── extraction/                     # PDF Parsers & Database Builders
│   ├── eda/                            # 14 High-Res Exploratory Scripts
│   ├── features/                       # Causal Taxonomy, DAG & Dataset Generator
│   ├── preprocessing/                  # Leak-Free 84-Feature Pipeline
│   ├── models/                         # ML Training, Tuning & Calibration
│   ├── ensemble/                       # Soft, Weighted & Stacking Ensembles
│   ├── explainability/                 # SHAP & Model Agreement Engines
│   ├── rules/                          # Geotechnical Safety Rule Engine
│   ├── similarity/                     # FAISS Historical Similarity Matcher
│   └── rag/                            # Local Regulatory RAG Engine
│
├── knowledge_base/
│   ├── dgms/                           # DGMS Technical Circulars
│   ├── regulations/                    # CMR 2017 & MMR 1961 Legal Articles
│   └── safety_alerts/                  # Precursor Warnings & Alerts
│
├── vector_store/
│   └── faiss/                          # Pre-built FAISS Vector Indices
│
├── models/                             # Serialized Model Artifacts (*.joblib)
├── results/                            # Benchmark Tables, Calibration & SHAP Plots
├── notebooks/                          # 8 Research & Engineering Notebooks
├── tests/                              # Pytest Suite (24 Passed / 24)
├── docs/                               # 10 Architectural & Safety Documents
├── Dockerfile                          # Production Containerfile
├── docker-compose.yml                  # Multi-service Deployment
├── requirements.txt                    # Project Dependencies
└── README.md                           # Master Documentation
```

---

## 4. Benchmark Model Evaluation (Phase 15)

The models were evaluated strictly on a held-out future chronological test set (225 records):

| Model Architecture | Balanced Accuracy | Macro F1 | HIGH Recall | CRITICAL Recall | False Negative Rate | Calibration (Brier) |
|---|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | 0.738 | 0.739 | **80.8%** | **55.6%** | **13.1%** | 0.075 |
| **XGBoost (Tuned)** | 0.675 | 0.691 | 78.8% | 33.3% | 16.4% | 0.095 |
| **LightGBM (Tuned)** | 0.663 | 0.672 | 76.9% | 33.3% | 16.4% | 0.095 |
| **CatBoost (Tuned)** | 0.721 | **0.749** | **80.8%** | 44.4% | 16.4% | 0.085 |
| **Soft Voting Ensemble** | 0.710 | 0.728 | **80.8%** | 44.4% | 14.7% | 0.089 |
| **Weighted Voting Ensemble** | 0.710 | 0.728 | **80.8%** | 44.4% | 14.7% | 0.089 |
| **Stacking Ensemble (LR Meta)** | 0.705 | 0.734 | **82.7%** | 33.3% | 14.7% | 0.079 |

---

## 5. Quickstart & Installation Guide

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: (Optional) Install Ollama for Local LLM
```bash
# Download and install Ollama from https://ollama.com
ollama run llama3:8b
```
*(If Ollama is not installed, BHOOmiKAVACH AI automatically falls back to deterministic synthesis templates without crashing).*

### Step 3: Run Full Pytest Suite
```bash
python -m pytest tests/
```
Output:
```
====================== 24 passed in 30.77s =======================
```

### Step 4: Launch FastAPI REST Server
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```
Swagger UI available at: `http://127.0.0.1:8000/docs`

### Step 5: Launch Streamlit Dashboard
```bash
streamlit run frontend/app.py
```
Open browser at: `http://localhost:8501`

---

## 6. Statutory Disclaimer & Safety Policy (Phase 35)
> **IMPORTANT NOTICE:** This platform is an **AI-assisted decision-support system**. It does **NOT** constitute an engineering safety certification, nor does it authorize mine entry or replace the statutory assessment of a certified Mine Manager, Overman, or Mining Sirdar under CMR 2017 / MMR 1961.
