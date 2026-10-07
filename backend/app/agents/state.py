"""
BHOOmiKAVACH AI — Agent State Definition (Phase 22)
===================================================
Defines the shared state schema flowing through the LangGraph multi-agent workflow.
"""

from typing import TypedDict, List, Dict, Any, Optional


class AgentState(TypedDict):
    # User Input & Session Context
    query: str
    user_role: str # "Worker", "Supervisor", "Safety Officer", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Executive"
    session_id: str
    mine_context: Optional[Dict[str, Any]]

    # Routing
    intent: str # "risk_assessment", "risk_explanation", "historical_query", "similarity_search", "recommendation", "rag_evidence", "general_query"
    confidence_level: str

    # Agent Artifacts
    risk_data: Optional[Dict[str, Any]]
    model_agreement: Optional[Dict[str, Any]]
    shap_explanation: Optional[Dict[str, Any]]
    rules_triggered: Optional[Dict[str, Any]]
    ood_status: Optional[Dict[str, Any]]
    similar_accidents: Optional[List[Dict[str, Any]]]
    historical_stats: Optional[Dict[str, Any]]
    rag_evidence: Optional[List[Dict[str, Any]]]
    role_recommendations: Optional[Dict[str, Any]]

    # Critic & Safety
    critic_passed: bool
    critic_notes: List[str]
    hallucination_warning: Optional[str]

    # Gemini Integration
    gemini_api_key: Optional[str]
    gemini_powered: Optional[bool]

    # Final Output Response
    final_response: str
    citations: List[Dict[str, str]]
