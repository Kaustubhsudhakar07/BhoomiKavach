"""
BHOOmiKAVACH AI — Safety Rules, OOD & Critic Policy Test Suite (Phase 28)
=========================================================================
Tests:
  - Deterministic safety rules (CMR 111 over-span, CMR 125 support density, etc.)
  - Insufficient data handling when critical fields are missing
  - OOD boundary detection and isolation forest anomaly flags
  - Critic safety policy compliance (no invented regulations, mandatory disclaimers)
"""

import pytest
from src.rules.rule_engine import GeotechnicalSafetyRuleEngine
from src.models.ood_detector import OutOfDistributionDetector
from backend.app.agents.critic_agent import critic_node
from backend.app.agents.state import AgentState


def test_rule_engine_statutory_triggers():
    engine = GeotechnicalSafetyRuleEngine()
    obs = {
        "depth_m": 400,
        "gallery_width_m": 5.2, # Over CMR 111 statutory limit (4.8m)
        "rmr_rating": 38,
        "support_density_bolts_per_m2": 0.75, # Substandard density
        "bolt_capacity_tonnes": 5.0, # Compromised anchorage
        "roof_convergence_mm_day": 7.5,
        "bed_separation_mm": 19.0,
        "support_condition": "Excessive Load/Yielding"
    }

    res = engine.evaluate_rules(obs)
    assert res["status"] == "EVALUATED"
    assert res["rule_severity"] in ["HIGH", "CRITICAL"]
    rule_ids = [r["rule_id"] for r in res["triggered_rules"]]
    assert "RULE_CMR_111_SPAN" in rule_ids
    assert "RULE_CMR_125_DENSITY" in rule_ids


def test_rule_engine_insufficient_data():
    engine = GeotechnicalSafetyRuleEngine()
    incomplete_obs = {"mine": "Moonidih Colliery"}
    res = engine.evaluate_rules(incomplete_obs)
    assert res["status"] == "INSUFFICIENT_DATA"
    assert "INSUFFICIENT DATA" in res["recommended_action"]


def test_ood_detector():
    detector = OutOfDistributionDetector()
    
    # In-bounds observation
    normal_obs = {"depth_m": 350, "gallery_width_m": 4.2, "rmr_rating": 50, "mineral": "Coal"}
    res_normal = detector.evaluate_ood(normal_obs)
    assert res_normal["is_ood"] is False

    # Out-of-bounds observation (depth 1800m, unknown mineral)
    abnormal_obs = {"depth_m": 1800, "gallery_width_m": 4.2, "rmr_rating": 50, "mineral": "Kryptonite"}
    res_abnormal = detector.evaluate_ood(abnormal_obs)
    assert res_abnormal["is_ood"] is True
    assert "Kryptonite" in res_abnormal["reasons"][0] or "1800" in res_abnormal["reasons"][0]


def test_critic_agent_disclaimer_and_no_hallucination():
    state: AgentState = {
        "query": "Assess mine risk",
        "user_role": "Supervisor",
        "session_id": "test_sess",
        "mine_context": {},
        "intent": "risk_assessment",
        "confidence_level": "HIGH",
        "risk_data": {"assessed_risk": "HIGH", "confidence": "HIGH"},
        "model_agreement": {"agreement_ratio": "3/3"},
        "shap_explanation": {"top_contributors": []},
        "rules_triggered": {"triggered_rules": []},
        "ood_status": None,
        "similar_accidents": [],
        "historical_stats": None,
        "rag_evidence": [],
        "role_recommendations": {"immediate_actions": "Evacuate"},
        "critic_passed": False,
        "critic_notes": [],
        "hallucination_warning": None,
        "final_response": "",
        "citations": []
    }

    res = critic_node(state)
    assert res["critic_passed"] is True
    assert "STATUTORY SAFETY DISCLAIMER" in res["final_response"]
    assert "Competent Person under CMR 2017" in res["final_response"]
