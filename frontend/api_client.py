"""
BHOOmiKAVACH AI — Streamlit API Client (Phase 25)
=================================================
Encapsulates all HTTP communication between Streamlit frontend and FastAPI backend.
No machine learning or heavy business logic is executed in Streamlit.
"""

import requests
from typing import Dict, Any, List, Optional

DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/api/v1"


class BhoomiApiClient:
    def __init__(self, base_url: str = DEFAULT_BACKEND_URL):
        self.base_url = base_url.rstrip("/")

    def get_health(self) -> Dict[str, Any]:
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=3.0)
            return resp.json() if resp.status_code == 200 else {"status": "OFFLINE", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "OFFLINE", "error": str(e)}

    def get_accident_stats(self) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/accidents/stats", timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def list_accidents(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/accidents", params=params or {}, timeout=5.0)
        resp.raise_for_status()
        return resp.json()

    def search_similar_accidents(self, query: str, top_k: int = 5, filter_coal_noncoal: Optional[str] = None) -> Dict[str, Any]:
        payload = {"query": query, "top_k": top_k, "filter_coal_noncoal": filter_coal_noncoal}
        resp = requests.post(f"{self.base_url}/accidents/similar", json=payload, timeout=45.0)
        resp.raise_for_status()
        return resp.json()

    def predict_risk(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(f"{self.base_url}/predict/risk", json=observation, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def query_rag(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        payload = {"query": query, "top_k": top_k}
        resp = requests.post(f"{self.base_url}/rag/query", json=payload, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    def ask_rag_gemini(self, query: str, user_role: str = "Safety Officer", top_k: int = 4, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        payload = {
            "query": query,
            "user_role": user_role,
            "top_k": top_k,
            "gemini_api_key": gemini_api_key
        }
        resp = requests.post(f"{self.base_url}/rag/ask", json=payload, timeout=60.0)
        resp.raise_for_status()
        return resp.json()

    def chat_with_agent(
        self,
        query: str,
        user_role: str = "Supervisor",
        mine_context: Dict[str, Any] = None,
        session_id: str = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        payload = {
            "query": query,
            "user_role": user_role,
            "mine_context": mine_context,
            "session_id": session_id,
            "gemini_api_key": gemini_api_key
        }
        resp = requests.post(f"{self.base_url}/agent/chat", json=payload, timeout=90.0)
        resp.raise_for_status()
        return resp.json()

    def get_audit_logs(self, limit: int = 50) -> Dict[str, Any]:
        resp = requests.get(f"{self.base_url}/audit/logs", params={"limit": limit}, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
