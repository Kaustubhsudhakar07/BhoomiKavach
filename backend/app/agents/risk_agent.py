"""
BHOOmiKAVACH AI — Prospective Risk Assessment Agent (Phase 22)
==============================================================
Orchestrates:
  - Preprocessing transformation
  - OOD verification
  - Gradient boosted model predictions (XGBoost, LightGBM, CatBoost)
  - Multi-model agreement consensus
  - Geotechnical safety rule engine
"""

import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any
from backend.app.agents.state import AgentState
from src.preprocessing.pipeline import PreprocessingManager, CLASS_NAMES
from src.models.ood_detector import OutOfDistributionDetector
from src.explainability.shap_explainer import ModelAgreementEngine
from src.rules.rule_engine import GeotechnicalSafetyRuleEngine

MODELS_DIR = "models"
prep_mgr = PreprocessingManager.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
ood_detector = OutOfDistributionDetector()
agreement_engine = ModelAgreementEngine()
rule_engine = GeotechnicalSafetyRuleEngine()


def risk_node(state: AgentState) -> Dict[str, Any]:
    mine_ctx = state.get("mine_context") or {}

    # If no mine context is provided, return default baseline demo observation
    if not mine_ctx or len(mine_ctx) < 3:
        mine_ctx = {
            "mine": "Moonidih Colliery",
            "coal_noncoal": "Coal",
            "mineral": "Coal",
            "mining_method": "Bord and Pillar (Depillaring)",
            "depth_m": 450,
            "working_height_m": 3.0,
            "gallery_width_m": 4.6,
            "gallery_height_m": 3.0,
            "is_junction": 1,
            "face_position": "Depillaring Edge / Goaf",
            "roof_lithology": "Sandstone-Shale Intercalation",
            "floor_lithology": "Shale",
            "rmr_rating": 44,
            "q_value": 0.5,
            "gsi": 40,
            "rqd": 45,
            "rock_strength_mpa": 35,
            "joint_spacing_cm": 25,
            "joint_orientation": "Unfavorable (Parallel to Drive)",
            "joint_condition": "Open/Slickensided",
            "water_condition": "Wet/Dripping",
            "water_seepage_lpm": 8.5,
            "roof_convergence_mm_day": 6.2,
            "roof_to_floor_convergence_mm_day": 8.8,
            "bed_separation_mm": 17.5,
            "support_density_bolts_per_m2": 0.85,
            "bolt_spacing_m": 1.2,
            "bolt_length_m": 1.8,
            "bolt_capacity_tonnes": 5.8,
            "cable_bolts_installed": 0,
            "props_installed": 1,
            "support_condition": "Excessive Load/Yielding",
            "recent_blasting": "Immediate (<4h)",
            "blast_distance_m": 15,
            "blast_vibration_ppv": 14.5,
            "production_tpd": 600,
            "persons_exposed": 6,
            "temperature_c": 32,
            "humidity_pct": 88,
            "previous_roof_fall": 1,
            "recent_geological_disturbance": 1
        }

    df_sample = pd.DataFrame([mine_ctx])
    X_encoded = prep_mgr.transform(df_sample)

    # 1. OOD check
    ood_res = ood_detector.evaluate_ood(mine_ctx, X_encoded)

    # 2. Rule Engine check
    rule_res = rule_engine.evaluate_rules(mine_ctx)

    # 3. Model Agreement & Probability scoring
    ag_res = agreement_engine.evaluate_agreement(X_encoded)

    # Reconcile confidence if OOD or Disagreement
    final_confidence = ag_res["confidence_level"]
    if ood_res["is_ood"]:
        final_confidence = "LOW (OOD Warning)"

    risk_result = {
        "assessed_risk": ag_res["ensemble_prediction"],
        "confidence": final_confidence,
        "probabilities": ag_res["risk_probabilities"],
        "observation_snapshot": mine_ctx
    }

    return {
        "risk_data": risk_result,
        "model_agreement": ag_res,
        "rules_triggered": rule_res,
        "ood_status": ood_res
    }
