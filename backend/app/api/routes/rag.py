"""
BHOOmiKAVACH AI — Regulatory & Case Precedent RAG Query Route (Phase 23)
========================================================================
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from src.rag.rag_engine import LocalRagEngine

router = APIRouter()
rag_engine = LocalRagEngine()


class RagQueryRequest(BaseModel):
    query: str
    top_k: int = 4


@router.post("/query")
def query_regulatory_rag(req: RagQueryRequest):
    try:
        evidence = rag_engine.query_evidence(req.query, top_k=req.top_k)
        return {
            "query": req.query,
            "evidence_count": len(evidence),
            "evidence": evidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
