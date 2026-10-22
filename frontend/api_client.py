"""
BHOOmiKAVACH AI — Streamlit API Client (Phase 25)
=================================================
Encapsulates all communication between Streamlit frontend and the backend.
Supports:
1. Direct in-process execution (default for Streamlit Community Cloud and standalone app).
   Guarantees 0ms latency, zero socket errors, and eliminates port binding issues.
2. HTTP requests to external remote backends (when BACKEND_URL is explicitly set to an external host).
"""

import os
import requests
from typing import Dict, Any, List, Optional


def resolve_backend_url() -> str:
    """
    Resolves the backend URL from Streamlit secrets or environment variables.
    Defaults to empty string (direct in-process mode).
    """
    env_url = os.environ.get("BACKEND_URL") or os.environ.get("BHOOMI_BACKEND_URL")
    if env_url:
        return env_url.rstrip("/")

    try:
        import streamlit as st
        if hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
            return str(st.secrets["BACKEND_URL"]).rstrip("/")
    except Exception:
        pass

    return ""


class BhoomiApiClient:
    def __init__(self, base_url: Optional[str] = None):
        if base_url is None:
            base_url = resolve_backend_url()
        self.base_url = (base_url or "").rstrip("/")
        # A backend is considered external remote ONLY if explicitly set to an http/https URL that is NOT localhost/127.0.0.1
        self.is_remote = bool(
            self.base_url and
            "127.0.0.1" not in self.base_url and
            "localhost" not in self.base_url
        )

    # 1. Health
    def get_health(self) -> Dict[str, Any]:
        if self.is_remote:
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=3.0)
                if resp.status_code == 200:
                    return resp.json()
            except Exception as e:
                return {"status": "OFFLINE", "error": str(e)}

        try:
            from backend.app.api.routes.health import get_system_health
            return get_system_health()
        except Exception as e:
            return {"status": "OFFLINE", "error": str(e)}

    # 2. Stats
    def get_accident_stats(self) -> Dict[str, Any]:
        if self.is_remote:
            resp = requests.get(f"{self.base_url}/accidents/stats", timeout=5.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.accidents import get_accident_statistics
        return get_accident_statistics()

    # 3. List Accidents
    def list_accidents(self, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if self.is_remote:
            resp = requests.get(f"{self.base_url}/accidents", params=params or {}, timeout=5.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.accidents import list_historical_accidents
        clean_params = params or {}
        return list_historical_accidents(
            year=clean_params.get("year"),
            coal_noncoal=clean_params.get("coal_noncoal"),
            state=clean_params.get("state"),
            mineral=clean_params.get("mineral"),
            limit=clean_params.get("limit", 50),
            offset=clean_params.get("offset", 0)
        )

    # 4. Similar Accidents Search
    def search_similar_accidents(self, query: str, top_k: int = 5, filter_coal_noncoal: Optional[str] = None) -> Dict[str, Any]:
        if self.is_remote:
            payload = {"query": query, "top_k": top_k, "filter_coal_noncoal": filter_coal_noncoal}
            resp = requests.post(f"{self.base_url}/accidents/similar", json=payload, timeout=45.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.accidents import search_similar_accidents as search_sim_fn, SimilarAccidentRequest
        req = SimilarAccidentRequest(query=query, top_k=top_k, filter_coal_noncoal=filter_coal_noncoal)
        return search_sim_fn(req)

    # 5. Predict Risk
    def predict_risk(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_remote:
            resp = requests.post(f"{self.base_url}/predict/risk", json=observation, timeout=30.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.predict import assess_mine_risk, MineObservationInput
        obs_obj = MineObservationInput(**observation)
        return assess_mine_risk(obs_obj)

    # 6. Regulatory RAG Query
    def query_rag(self, query: str, top_k: int = 4) -> Dict[str, Any]:
        if self.is_remote:
            payload = {"query": query, "top_k": top_k}
            resp = requests.post(f"{self.base_url}/rag/query", json=payload, timeout=30.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.rag import query_regulatory_rag, RagQueryRequest
        req = RagQueryRequest(query=query, top_k=top_k)
        return query_regulatory_rag(req)

    # 7. Regulatory RAG Ask (Gemini)
    def ask_rag_gemini(self, query: str, user_role: str = "Safety Officer", top_k: int = 4, gemini_api_key: Optional[str] = None) -> Dict[str, Any]:
        if self.is_remote:
            payload = {
                "query": query,
                "user_role": user_role,
                "top_k": top_k,
                "gemini_api_key": gemini_api_key
            }
            resp = requests.post(f"{self.base_url}/rag/ask", json=payload, timeout=60.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.rag import ask_regulatory_safety_advisor, RagAskRequest
        req = RagAskRequest(query=query, user_role=user_role, top_k=top_k, gemini_api_key=gemini_api_key)
        return ask_regulatory_safety_advisor(req)

    # 8. Agent Chat
    def chat_with_agent(
        self,
        query: str,
        user_role: str = "Supervisor",
        mine_context: Dict[str, Any] = None,
        session_id: str = None,
        gemini_api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        if self.is_remote:
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

        from backend.app.api.routes.agent import chat_with_agent as chat_fn, AgentChatRequest
        req = AgentChatRequest(
            query=query,
            user_role=user_role,
            mine_context=mine_context,
            session_id=session_id,
            gemini_api_key=gemini_api_key
        )
        return chat_fn(req)

    # 9. Audit Logs
    def get_audit_logs(self, limit: int = 50) -> Dict[str, Any]:
        if self.is_remote:
            resp = requests.get(f"{self.base_url}/audit/logs", params={"limit": limit}, timeout=10.0)
            resp.raise_for_status()
            return resp.json()

        from backend.app.api.routes.audit import get_audit_logs as audit_fn
        return audit_fn(limit=limit)

    # 10. Upload RAG Document
    def upload_rag_document(self, file_name: str, file_bytes: bytes) -> Dict[str, Any]:
        if self.is_remote:
            files = {"file": (file_name, file_bytes)}
            resp = requests.post(f"{self.base_url}/rag/upload", files=files, timeout=120.0)
            resp.raise_for_status()
            return resp.json()

        import tempfile
        from backend.app.api.routes.rag import rag_engine, _extract_text_from_file, _chunk_text
        ext = os.path.splitext(file_name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            raw_text = _extract_text_from_file(tmp_path, file_name)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        chunks = _chunk_text(raw_text, chunk_size=800, overlap=150)
        new_docs = []
        for i, chunk in enumerate(chunks):
            new_docs.append({
                "doc_id": f"UPLOAD-{file_name}-CHUNK-{i+1}",
                "title": f"{file_name} (Chunk {i+1}/{len(chunks)})",
                "category": "User Uploaded Document",
                "content": chunk,
                "source": file_name,
                "page": f"Chunk {i+1}",
                "topics": "User Upload"
            })
        added_count = rag_engine.add_documents(new_docs)
        return {
            "status": "success",
            "filename": file_name,
            "chunks_added": added_count,
            "total_characters": len(raw_text),
            "total_index_size": len(rag_engine.documents),
            "message": f"Successfully indexed {added_count} chunks from '{file_name}' into the RAG knowledge base."
        }
