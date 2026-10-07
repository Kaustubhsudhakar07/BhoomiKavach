"""
BHOOmiKAVACH AI — LangGraph Multi-Agent Orchestrator (Phase 22 & 24)
===================================================================
Constructs the full LangGraph state graph linking:
  - Router Agent
  - Historical Analytics Agent
  - Similarity Agent
  - Risk Assessment Agent
  - Explanation Agent
  - Evidence Agent
  - Recommendation Agent
  - Critic & Safety Agent
"""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from backend.app.agents.state import AgentState
from backend.app.agents.router_agent import router_node
from backend.app.agents.historical_agent import historical_node
from backend.app.agents.similarity_agent import similarity_node
from backend.app.agents.risk_agent import risk_node
from backend.app.agents.explanation_agent import explanation_node
from backend.app.agents.evidence_agent import evidence_node
from backend.app.agents.recommendation_agent import recommendation_node
from backend.app.agents.critic_agent import critic_node


def route_decision(state: AgentState) -> str:
    intent = state.get("intent", "general_query")
    if intent in ["risk_assessment", "risk_explanation"]:
        return "risk_node"
    elif intent == "historical_query":
        return "historical_node"
    elif intent == "similarity_search":
        return "similarity_node"
    elif intent == "recommendation":
        return "risk_node"
    elif intent == "rag_evidence":
        return "evidence_node"
    else:
        return "critic_node"


def build_bhoomikavach_graph():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router_node", router_node)
    workflow.add_node("historical_node", historical_node)
    workflow.add_node("similarity_node", similarity_node)
    workflow.add_node("risk_node", risk_node)
    workflow.add_node("explanation_node", explanation_node)
    workflow.add_node("evidence_node", evidence_node)
    workflow.add_node("recommendation_node", recommendation_node)
    workflow.add_node("critic_node", critic_node)

    # Set Entry Point
    workflow.set_entry_point("router_node")

    # Conditional Routing from Router
    workflow.add_conditional_edges(
        "router_node",
        route_decision,
        {
            "risk_node": "risk_node",
            "historical_node": "historical_node",
            "similarity_node": "similarity_node",
            "evidence_node": "evidence_node",
            "critic_node": "critic_node"
        }
    )

    # Workflow paths
    # Risk Assessment Pipeline: risk -> explanation -> similarity -> evidence -> recommendation -> critic
    workflow.add_edge("risk_node", "explanation_node")
    workflow.add_edge("explanation_node", "similarity_node")
    workflow.add_edge("similarity_node", "evidence_node")
    workflow.add_edge("evidence_node", "recommendation_node")
    workflow.add_edge("recommendation_node", "critic_node")

    # Historical Path
    workflow.add_edge("historical_node", "critic_node")

    # Terminal Edge
    workflow.add_edge("critic_node", END)

    app = workflow.compile()
    return app


# Pre-compiled workflow singleton
bhoomikavach_agent_app = build_bhoomikavach_graph()


def run_agentic_workflow(
    query: str,
    user_role: str = "Supervisor",
    mine_context: Dict[str, Any] = None,
    session_id: str = "default_session"
) -> Dict[str, Any]:
    """
    Executes the multi-agent graph with user inputs and returns complete decision-support packet.
    """
    initial_state: AgentState = {
        "query": query,
        "user_role": user_role,
        "session_id": session_id,
        "mine_context": mine_context,
        "intent": "",
        "confidence_level": "",
        "risk_data": None,
        "model_agreement": None,
        "shap_explanation": None,
        "rules_triggered": None,
        "ood_status": None,
        "similar_accidents": None,
        "historical_stats": None,
        "rag_evidence": None,
        "role_recommendations": None,
        "critic_passed": False,
        "critic_notes": [],
        "hallucination_warning": None,
        "final_response": "",
        "citations": []
    }

    final_output = bhoomikavach_agent_app.invoke(initial_state)
    return final_output


if __name__ == "__main__":
    print("Testing LangGraph Multi-Agent Workflow on Risk Query...")
    sample_res = run_agentic_workflow(
        query="Why is my current roof-fall risk HIGH after blasting in depillaring stook?",
        user_role="Safety Officer"
    )
    print("\n--- FINAL WORKFLOW RESPONSE ---")
    print(sample_res["final_response"])
