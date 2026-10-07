"""
BHOOmiKAVACH AI — Explanation Agent (Phase 22)
==============================================
Generates local SHAP feature attribution explaining why the model assigned
the given risk score. Strictly enforces non-causal explanatory language.
"""

import os
import pandas as pd
from typing import Dict, Any
from backend.app.agents.state import AgentState
from src.preprocessing.pipeline import PreprocessingManager, TARGET_MAP
from src.explainability.shap_explainer import BhoomiShapExplainer

MODELS_DIR = "models"
prep_mgr = PreprocessingManager.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
explainer = BhoomiShapExplainer("XGBoost")


def explanation_node(state: AgentState) -> Dict[str, Any]:
    risk_data = state.get("risk_data") or {}
    obs = risk_data.get("observation_snapshot") or state.get("mine_context") or {}

    if not obs:
        return {"shap_explanation": None}

    df_sample = pd.DataFrame([obs])
    X_encoded = prep_mgr.transform(df_sample)
    
    assessed_risk = risk_data.get("assessed_risk", "HIGH")
    cls_idx = TARGET_MAP.get(assessed_risk, 2)

    try:
        explanation = explainer.explain_instance(X_encoded, predicted_class=cls_idx)
    except Exception as e:
        explanation = {"error": str(e)}

    return {"shap_explanation": explanation}
