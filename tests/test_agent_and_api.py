"""
BHOOmiKAVACH AI — FastAPI Endpoints & Agent Workflow Test Suite (Phase 28)
==========================================================================
Tests:
  - FastAPI /health endpoint
  - FastAPI /accidents/stats endpoint
  - FastAPI /predict/risk endpoint
  - FastAPI /agent/chat multi-agent decision support
"""

import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)


def test_api_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["HEALTHY", "DEGRADED"]
    assert "model_version" in data


def test_api_accidents_stats():
    response = client.get("/api/v1/accidents/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["total_accidents"] == 41
    assert data["total_fatalities"] == 58


def test_api_predict_risk():
    payload = {
        "mine": "Moonidih Colliery",
        "coal_noncoal": "Coal",
        "mineral": "Coal",
        "mining_method": "Bord and Pillar (Depillaring)",
        "depth_m": 420.0,
        "working_height_m": 2.8,
        "gallery_width_m": 4.5,
        "gallery_height_m": 2.8,
        "is_junction": 0,
        "face_position": "Development Face",
        "roof_lithology": "Sandstone",
        "floor_lithology": "Shale",
        "rmr_rating": 45.0,
        "roof_convergence_mm_day": 3.2,
        "bed_separation_mm": 8.0,
        "support_density_bolts_per_m2": 1.1,
        "bolt_capacity_tonnes": 8.0,
        "support_condition": "Intact & Adequate",
        "recent_blasting": "No Blasting (<24h)"
    }
    response = client.post("/api/v1/predict/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "assessed_risk" in data
    assert "model_agreement" in data
    assert "audit_id" in data


def test_api_agent_chat():
    payload = {
        "query": "What are the historical roof fall fatality figures in Coal vs Non-Coal mines?",
        "user_role": "Mining Engineer"
    }
    response = client.post("/api/v1/agent/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "historical_query"
    assert "STATUTORY SAFETY DISCLAIMER" in data["final_response"]
