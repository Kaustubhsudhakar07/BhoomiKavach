"""
BHOOmiKAVACH AI — Statutory Evidence & RAG Agent (Phase 22)
===========================================================
Retrieves legal and technical regulatory evidence from FAISS
(CMR 2017, MMR 1961, DGMS Circulars, and inquiry precedents).
"""

from typing import Dict, Any, List
from backend.app.agents.state import AgentState
from src.rag.rag_engine import LocalRagEngine

rag_engine = LocalRagEngine()


def evidence_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    risk_data = state.get("risk_data") or {}
    assessed_risk = risk_data.get("assessed_risk", "HIGH")
    
    # Formulate targeted RAG query
    rag_query = f"{query} {assessed_risk} roof fall prevention CMR 2017 DGMS circular Systematic Support Rules"
    
    try:
        evidence = rag_engine.query_evidence(rag_query, top_k=3)
    except Exception as e:
        evidence = [{"error": str(e)}]

    return {"rag_evidence": evidence}
