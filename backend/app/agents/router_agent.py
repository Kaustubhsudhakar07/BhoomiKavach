"""
BHOOmiKAVACH AI — Intent Router & Orchestrator Agent (Phase 22)
===============================================================
Classifies the user query into operational mining safety intents:
  1. `risk_assessment`: evaluate current or supplied mine parameters
  2. `risk_explanation`: explain why risk is High/Critical using SHAP & agreement
  3. `historical_query`: analyze DGMS accident statistics, trends, and causes
  4. `similarity_search`: find previous 0111 fatal fall cases matching current scenario
  5. `recommendation`: what should a specific role do given mine conditions
  6. `rag_evidence`: statutory regulation or DGMS circular lookup
  7. `general_query`: mining safety definitions and overview
"""

import re
from typing import Dict, Any
from backend.app.agents.state import AgentState


def router_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "").lower()
    mine_ctx = state.get("mine_context")

    intent = "general_query"

    if any(k in query for k in ["why is", "explain risk", "explain the", "contributor", "shap", "factor"]):
        intent = "risk_explanation"
    elif any(k in query for k in ["predict", "assess", "risk score", "current risk", "hazard level"]) or (mine_ctx and len(mine_ctx) > 3):
        intent = "risk_assessment"
    elif any(k in query for k in ["similar", "case", "precedent", "like this", "happened before", "past accident"]):
        intent = "similarity_search"
    elif any(k in query for k in ["should i do", "what should", "action", "recommend", "protocol", "procedure", "sop"]):
        intent = "recommendation"
    elif any(k in query for k in ["statute", "regulation", "cmr", "mmr", "circular", "rule", "dgms order", "legal"]):
        intent = "rag_evidence"
    elif any(k in query for k in ["how many", "trend", "statistics", "fatalities", "between 20", "coal vs non-coal", "state with most", "causes"]):
        intent = "historical_query"

    return {"intent": intent}
