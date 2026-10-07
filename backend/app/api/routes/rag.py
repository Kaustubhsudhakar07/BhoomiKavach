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


class RagAskRequest(BaseModel):
    query: str
    user_role: Optional[str] = "Safety Officer"
    top_k: int = 4
    gemini_api_key: Optional[str] = None


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


@router.post("/ask")
def ask_regulatory_safety_advisor(req: RagAskRequest):
    try:
        evidence = rag_engine.query_evidence(req.query, top_k=req.top_k)

        # Try Gemini reasoning
        from backend.app.services.gemini_service import generate_gemini_safety_advisory, is_gemini_available
        gemini_answer = None
        gemini_powered = False

        if is_gemini_available(req.gemini_api_key):
            gemini_answer = generate_gemini_safety_advisory(
                query=req.query,
                user_role=req.user_role or "Safety Officer",
                rag_evidence=evidence,
                api_key=req.gemini_api_key
            )
            if gemini_answer:
                gemini_powered = True

        if not gemini_answer:
            # Deterministic fallback synthesis
            blocks = [f"STATUTORY REGULATORY ADVISORY (ROLE: {req.user_role})", f"QUERY: {req.query}\n"]
            if evidence:
                blocks.append(f"Top {len(evidence)} Relevant Provisions Identified from DGMS Knowledge Base:")
                for i, ev in enumerate(evidence, 1):
                    blocks.append(f"{i}. {ev.get('title')} [{ev.get('category')}]")
                    blocks.append(f"   Source: {ev.get('source')}, {ev.get('page')}")
                    blocks.append(f"   Provision Excerpt: {ev.get('content')}\n")
            else:
                blocks.append("No specific regulatory clauses directly matched this phrasing.")
            blocks.append("\n[NOTE: Generative AI is in template fallback mode. Set GEMINI_API_KEY to activate Google Gemini reasoning.]")
            gemini_answer = "\n".join(blocks)

        citations = [
            {"title": e.get("title", ""), "source": e.get("source", ""), "page": e.get("page", "")}
            for e in evidence if "source" in e
        ]

        return {
            "query": req.query,
            "user_role": req.user_role,
            "gemini_powered": gemini_powered,
            "answer": gemini_answer,
            "citations": citations,
            "evidence": evidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Regulatory safety advisory failed: {str(e)}")
