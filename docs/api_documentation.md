# BHOOmiKAVACH AI — REST API Documentation

**Base URL:** `http://127.0.0.1:8000/api/v1`  
**Interactive Docs:** `http://127.0.0.1:8000/docs` (Swagger UI)  
**Alternative Docs:** `http://127.0.0.1:8000/redoc` (ReDoc)

---

## 1. System Health
### `GET /health`
Returns system status, active database counts, model version, and component health.
```json
{
  "status": "HEALTHY",
  "project": "BHOOmiKAVACH AI",
  "model_version": "v1.2.0-champion",
  "dataset_version": "DGMS-2016-2022-CANONICAL-V1",
  "components": {
    "database": {"status": "ONLINE", "historical_accidents_loaded": 41},
    "ml_models": {"status": "ONLINE"},
    "faiss_vector_stores": {"status": "ONLINE"}
  }
}
```

---

## 2. Historical Accident Intelligence
### `GET /accidents`
Lists canonical DGMS fatal roof-fall records with filtering options:
- Query Parameters: `year`, `coal_noncoal`, `state`, `mineral`, `limit`, `offset`.

### `GET /accidents/stats`
Returns aggregated historical distributions (accidents by year, sector, state, and mineral).

### `POST /accidents/similar`
Searches nearest historical inquiry cases via FAISS vector similarity.
```json
{
  "query": "Depillaring stook extraction near active goaf edge with sandstone parting",
  "top_k": 3,
  "filter_coal_noncoal": "Coal"
}
```

---

## 3. Live Risk Prediction & Explainability
### `POST /predict/risk`
Evaluates live mine telemetry across Preprocessing, OOD, Models, Agreement, SHAP, and Rules.
```json
{
  "mine": "Moonidih Colliery",
  "coal_noncoal": "Coal",
  "mineral": "Coal",
  "depth_m": 420.0,
  "gallery_width_m": 4.5,
  "rmr_rating": 45.0,
  "roof_convergence_mm_day": 3.8,
  "bed_separation_mm": 11.5,
  "support_density_bolts_per_m2": 1.1,
  "bolt_capacity_tonnes": 8.5,
  "user_role": "Safety Officer"
}
```
**Response Key Fields:**
- `assessed_risk`: "MODERATE" | "HIGH" | "CRITICAL" | "LOW"
- `confidence`: "HIGH" | "MEDIUM" | "LOW"
- `model_agreement`: `{ "agreement_ratio": "3/3", "consensus_score": 1.0 }`
- `shap_explanation`: Top 5 directional contributors.
- `rule_engine_findings`: Triggered statutory rules (CMR 111, 125, etc.).
- `ood_diagnostics`: In-bounds verification.
- `audit_id`: Immutable audit record ID.

---

## 4. Regulatory Knowledge Base (RAG)
### `POST /rag/query`
Queries official regulations, circulars, and inquiry lessons.
```json
{
  "query": "What statutory regulations mandate support setting at the goaf edge?",
  "top_k": 3
}
```

---

## 5. Agentic AI Decision Advisor
### `POST /agent/chat`
Invokes the LangGraph multi-agent decision support system.
```json
{
  "query": "Why is my current roof-fall risk HIGH after blasting in depillaring stook?",
  "user_role": "Safety Officer",
  "mine_context": { ... },
  "session_id": "session-001"
}
```

---

## 6. Audit Trail
### `GET /audit/logs`
Retrieves chronological audit log entries containing input snapshots, model agreement, and decision metadata.
