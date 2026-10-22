"""
BHOOmiKAVACH AI — System Health & Diagnostics Endpoint (Phase 23)
=================================================================
"""

import os
import requests
from fastapi import APIRouter
from backend.app.core.config import settings
from backend.app.database.session import get_db_connection

router = APIRouter()


@router.get("/health")
def get_system_health():
    # 1. Database check
    db_ok = False
    acc_count = 0
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT count(*) FROM accidents;")
        acc_count = c.fetchone()[0]
        conn.close()
        db_ok = True
    except Exception:
        db_ok = False

    # 2. Vector Store check
    faiss_ok = os.path.exists("models/similarity_index/faiss_accidents.bin") and os.path.exists("vector_store/faiss/rag_index.bin")

    # 3. Model files check
    models_ok = (
        os.path.exists("models/xgb_model.joblib") and
        os.path.exists("models/lgb_model.joblib") and
        os.path.exists("models/cat_model.joblib") and
        os.path.exists("models/preprocessor.joblib")
    )

    # 4. Ollama check (Phase 31: check whether Ollama is available, graceful fallback)
    ollama_ok = False
    try:
        resp = requests.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=0.2)
        ollama_ok = (resp.status_code == 200)
    except Exception:
        ollama_ok = False

    return {
        "status": "HEALTHY" if (db_ok and models_ok and faiss_ok) else "DEGRADED",
        "project": settings.PROJECT_NAME,
        "tagline": "SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL.",
        "model_version": settings.MODEL_VERSION,
        "dataset_version": settings.DATASET_VERSION,
        "feature_version": settings.FEATURE_VERSION,
        "components": {
            "database": {"status": "ONLINE" if db_ok else "OFFLINE", "historical_accidents_loaded": acc_count},
            "ml_models": {"status": "ONLINE" if models_ok else "OFFLINE"},
            "faiss_vector_stores": {"status": "ONLINE" if faiss_ok else "OFFLINE"},
            "local_llm_ollama": {
                "status": "ONLINE" if ollama_ok else "OFFLINE",
                "notice": "LOCAL LLM AVAILABLE" if ollama_ok else "LOCAL LLM UNAVAILABLE — AI narrative generation operates in template fallback mode."
            }
        }
    }
