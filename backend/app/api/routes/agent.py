"""
BHOOmiKAVACH AI — Agent Chat API Route (Phase 24)
=================================================
Exposes the multi-agent decision support system through:
  `POST /api/v1/agent/chat`
"""

import uuid
import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from backend.app.agents.graph import run_agentic_workflow
from backend.app.database.session import get_db_connection, log_audit_trail

router = APIRouter()


class AgentChatRequest(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Why is my current roof-fall risk HIGH?"})
    user_role: Optional[str] = Field(default="Supervisor", json_schema_extra={"example": "Safety Officer"})
    mine_context: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = Field(default_factory=lambda: f"sess-{uuid.uuid4().hex[:8]}")
    gemini_api_key: Optional[str] = Field(default=None, description="Optional Google Gemini API key for generative safety reasoning")


@router.post("/chat")
def chat_with_agent(req: AgentChatRequest):
    sess_id = req.session_id or f"sess-{uuid.uuid4().hex[:8]}"

    try:
        # 1. Execute LangGraph multi-agent workflow
        result = run_agentic_workflow(
            query=req.query,
            user_role=req.user_role or "Supervisor",
            mine_context=req.mine_context,
            session_id=sess_id,
            gemini_api_key=req.gemini_api_key
        )

        # 2. Record Session & Message in Database (Phase 26)
        conn = get_db_connection()
        c = conn.cursor()
        now_str = datetime.datetime.now().isoformat()
        
        c.execute("""
        INSERT OR IGNORE INTO agent_sessions (session_id, started_at, user_role, last_active)
        VALUES (?, ?, ?, ?);
        """, (sess_id, now_str, req.user_role or "Supervisor", now_str))

        c.execute("""
        UPDATE agent_sessions SET last_active = ? WHERE session_id = ?;
        """, (now_str, sess_id))

        # Store user message
        c.execute("""
        INSERT INTO agent_messages (session_id, timestamp, sender, intent, content)
        VALUES (?, ?, 'USER', ?, ?);
        """, (sess_id, now_str, result.get("intent", "query"), req.query))

        # Store assistant response
        c.execute("""
        INSERT INTO agent_messages (session_id, timestamp, sender, intent, content)
        VALUES (?, ?, 'ASSISTANT', ?, ?);
        """, (sess_id, now_str, result.get("intent", "response"), result.get("final_response", "")))

        conn.commit()
        conn.close()

        # 3. Log Audit Trail
        risk_info = result.get("risk_data") or {}
        model_agreement_val = (result.get("model_agreement") or {}).get("agreement_ratio", "N/A")
        audit_id = log_audit_trail(
            user_role=req.user_role or "Supervisor",
            input_snapshot={"query": req.query, "context": req.mine_context},
            risk_output=risk_info.get("assessed_risk", "N/A"),
            confidence=risk_info.get("confidence", "HIGH"),
            model_agreement=str(model_agreement_val),
            recommendation=result.get("final_response", "")[:300],
            tool_calls=["Router", "RiskEngine", "SHAP", "FAISS_Similarity", "FAISS_RAG", "Critic"]
        )

        return {
            "session_id": sess_id,
            "intent": result.get("intent"),
            "confidence_level": result.get("confidence_level"),
            "risk_assessment": result.get("risk_data"),
            "model_agreement": result.get("model_agreement"),
            "shap_explanation": result.get("shap_explanation"),
            "rules_triggered": result.get("rules_triggered"),
            "ood_status": result.get("ood_status"),
            "similar_accidents": result.get("similar_accidents"),
            "historical_stats": result.get("historical_stats"),
            "rag_evidence": result.get("rag_evidence"),
            "role_recommendations": result.get("role_recommendations"),
            "critic_passed": result.get("critic_passed", True),
            "gemini_powered": result.get("gemini_powered", False),
            "final_response": result.get("final_response"),
            "citations": result.get("citations", []),
            "audit_id": audit_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent workflow failed: {str(e)}")
