"""
BHOOmiKAVACH AI — Live Risk Prediction & Explainability Route (Phase 23)
========================================================================
"""

import os
import json
import datetime
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from backend.app.core.config import settings
from backend.app.database.session import get_db_connection, log_audit_trail
from src.preprocessing.pipeline import PreprocessingManager, TARGET_MAP, CLASS_NAMES
from src.models.ood_detector import OutOfDistributionDetector
from src.explainability.shap_explainer import ModelAgreementEngine, BhoomiShapExplainer
from src.rules.rule_engine import GeotechnicalSafetyRuleEngine

router = APIRouter()

MODELS_DIR = "models"
prep_mgr = PreprocessingManager.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
ood_detector = OutOfDistributionDetector()
agreement_engine = ModelAgreementEngine()
explainer = BhoomiShapExplainer("XGBoost")
rule_engine = GeotechnicalSafetyRuleEngine()


class MineObservationInput(BaseModel):
    mine: str = Field(default="Moonidih Colliery", description="Colliery or mine name")
    coal_noncoal: str = Field(default="Coal", description="'Coal' or 'Non-Coal'")
    mineral: str = Field(default="Coal", description="Mineral extracted")
    mining_method: str = Field(default="Bord and Pillar (Depillaring)")
    depth_m: float = Field(default=350.0, ge=10.0, le=1500.0)
    working_height_m: float = Field(default=2.8, ge=1.0, le=7.0)
    gallery_width_m: float = Field(default=4.5, ge=2.0, le=8.0)
    gallery_height_m: float = Field(default=2.8, ge=1.0, le=7.0)
    is_junction: int = Field(default=0, ge=0, le=1)
    face_position: str = Field(default="Development Face")
    roof_lithology: str = Field(default="Sandstone")
    floor_lithology: str = Field(default="Shale")
    rmr_rating: float = Field(default=48.0, ge=10.0, le=95.0)
    q_value: Optional[float] = Field(default=1.5)
    gsi: Optional[float] = Field(default=45.0)
    rqd: Optional[float] = Field(default=50.0)
    rock_strength_mpa: Optional[float] = Field(default=40.0)
    joint_spacing_cm: Optional[float] = Field(default=30.0)
    joint_orientation: Optional[str] = Field(default="Fair")
    joint_condition: Optional[str] = Field(default="Slightly Weathered")
    water_condition: Optional[str] = Field(default="Damp")
    water_seepage_lpm: Optional[float] = Field(default=2.5)
    roof_convergence_mm_day: float = Field(default=2.2, ge=0.0)
    roof_to_floor_convergence_mm_day: Optional[float] = Field(default=3.1)
    bed_separation_mm: float = Field(default=6.5, ge=0.0)
    support_density_bolts_per_m2: float = Field(default=1.2, ge=0.1)
    bolt_spacing_m: Optional[float] = Field(default=1.0)
    bolt_length_m: Optional[float] = Field(default=1.8)
    bolt_capacity_tonnes: float = Field(default=8.5, ge=1.0)
    cable_bolts_installed: Optional[int] = Field(default=0)
    props_installed: Optional[int] = Field(default=1)
    support_condition: str = Field(default="Intact & Adequate")
    recent_blasting: str = Field(default="No Blasting (<24h)")
    blast_distance_m: Optional[float] = Field(default=30.0)
    blast_vibration_ppv: Optional[float] = Field(default=4.5)
    production_tpd: Optional[float] = Field(default=600.0)
    persons_exposed: Optional[int] = Field(default=4)
    temperature_c: Optional[float] = Field(default=28.0)
    humidity_pct: Optional[float] = Field(default=80.0)
    previous_roof_fall: Optional[int] = Field(default=0)
    recent_geological_disturbance: Optional[int] = Field(default=0)
    user_role: Optional[str] = Field(default="Safety Officer")


@router.post("/risk")
def assess_mine_risk(obs: MineObservationInput):
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else obs.dict()
    df_sample = pd.DataFrame([obs_dict])

    try:
        X_encoded = prep_mgr.transform(df_sample)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")

    # 1. OOD Verification
    ood_res = ood_detector.evaluate_ood(obs_dict, X_encoded)

    # 2. Rule Engine Check
    rule_res = rule_engine.evaluate_rules(obs_dict)

    # 3. Model Agreement & Probability Consensus
    ag_res = agreement_engine.evaluate_agreement(X_encoded)
    assessed_risk = ag_res["ensemble_prediction"]
    cls_idx = TARGET_MAP.get(assessed_risk, 1)

    # 4. SHAP Local Explanation
    shap_res = explainer.explain_instance(X_encoded, predicted_class=cls_idx)

    # Reconcile confidence if OOD detected
    final_confidence = ag_res["confidence_level"]
    if ood_res["is_ood"]:
        final_confidence = "LOW (OOD Warning Triggered)"

    # Persist prediction in SQLite
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    INSERT INTO predictions (
        timestamp, mine, predicted_risk, confidence, model_agreement,
        model_version, probabilities, input_snapshot
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        datetime.datetime.now().isoformat(),
        obs.mine,
        assessed_risk,
        final_confidence,
        ag_res["agreement_ratio"],
        settings.MODEL_VERSION,
        json.dumps(ag_res["risk_probabilities"]),
        json.dumps(obs_dict)
    ))
    pred_id = c.lastrowid
    conn.commit()
    conn.close()

    # Log immutable audit trail entry (Phase 27)
    audit_id = log_audit_trail(
        user_role=obs.user_role or "Safety Officer",
        input_snapshot=obs_dict,
        risk_output=assessed_risk,
        confidence=final_confidence,
        model_agreement=ag_res["agreement_ratio"],
        recommendation=f"{len(rule_res.get('triggered_rules', []))} statutory rules triggered; assessed risk: {assessed_risk}"
    )

    return {
        "prediction_id": pred_id,
        "audit_id": audit_id,
        "assessed_risk": assessed_risk,
        "confidence": final_confidence,
        "model_agreement": ag_res,
        "probabilities": ag_res["risk_probabilities"],
        "shap_explanation": shap_res,
        "rule_engine_findings": rule_res,
        "ood_diagnostics": ood_res,
        "safety_disclaimer": "AI-assisted risk assessment only. Does not replace statutory inspection by Competent Person under CMR 2017 / MMR 1961."
    }
