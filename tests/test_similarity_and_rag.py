"""
BHOOmiKAVACH AI — FAISS Similarity & RAG Test Suite (Phase 28)
==============================================================
Tests:
  - FAISS index loading and vector dimensionality
  - Historical accident similarity ranking
  - Regulatory RAG query and source attribution integrity
"""

import pytest
from src.similarity.accident_matcher import AccidentSimilarityMatcher
from src.rag.rag_engine import LocalRagEngine


def test_similarity_matcher():
    matcher = AccidentSimilarityMatcher()
    matches = matcher.query_similar_cases(
        "Depillaring stook extraction goaf edge roof fall props dislodged",
        top_k=2
    )
    assert len(matches) == 2
    assert matches[0]["similarity_score"] > 0.0
    assert "mine" in matches[0]
    assert "root_cause" in matches[0]


def test_rag_retrieval():
    rag = LocalRagEngine()
    results = rag.query_evidence("Regulation 111 gallery width CMR 2017", top_k=2)
    assert len(results) >= 1
    top_doc = results[0]
    assert "CMR" in top_doc["title"] or "111" in top_doc["title"] or "Regulation" in top_doc["title"]
    assert "source" in top_doc
    assert "page" in top_doc
