"""
BHOOmiKAVACH AI — Historical Similarity Matching Agent (Phase 22)
=================================================================
Connects to FAISS Vector Search over official DGMS fatal accident cases
to retrieve precedent inquiry findings and preventive recommendations.
"""

from typing import Dict, Any, List
from backend.app.agents.state import AgentState
from src.similarity.accident_matcher import AccidentSimilarityMatcher

matcher = AccidentSimilarityMatcher()


def similarity_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "")
    mine_ctx = state.get("mine_context") or {}

    # Build search context
    search_prompt = query
    if mine_ctx:
        search_prompt += (
            f" Context: Mine {mine_ctx.get('mine', '')}, Mineral {mine_ctx.get('mineral', '')}, "
            f"Method {mine_ctx.get('mining_method', '')}, Depth {mine_ctx.get('depth_m', '')}m, "
            f"RMR {mine_ctx.get('rmr_rating', '')}, Width {mine_ctx.get('gallery_width_m', '')}m, "
            f"Support {mine_ctx.get('support_condition', '')}."
        )

    try:
        matches = matcher.query_similar_cases(search_prompt, top_k=3)
    except Exception as e:
        matches = [{"error": str(e)}]

    return {"similar_accidents": matches}
