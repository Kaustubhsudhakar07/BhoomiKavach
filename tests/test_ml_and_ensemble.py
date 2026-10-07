"""
BHOOmiKAVACH AI — ML & Ensemble Test Suite (Phase 28)
=====================================================
Tests:
  - Feature engineering and PreprocessingManager transformation
  - Zero temporal leakage (strictly chronologically ordered splits)
  - Model serialization and predict_proba on XGBoost, LightGBM, CatBoost
  - Ensemble Soft and Weighted voting functionality
  - SHAP local instance attribution integrity
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.pipeline import PreprocessingManager, prepare_temporal_datasets, CLASS_NAMES
from src.explainability.shap_explainer import BhoomiShapExplainer, ModelAgreementEngine


def test_preprocessing_and_temporal_splits():
    data = prepare_temporal_datasets()
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]

    # Verify chronological order (zero temporal leakage)
    assert train_df["date"].max() <= val_df["date"].min(), "Temporal leakage detected between Train and Val"
    assert val_df["date"].max() <= test_df["date"].min(), "Temporal leakage detected between Val and Test"

    # Verify feature dimension
    assert data["X_train"].shape[1] == 84, f"Expected 84 features, got {data['X_train'].shape[1]}"
    assert len(data["feature_names"]) == 84


def test_models_exist_and_predict():
    prep = PreprocessingManager.load("models/preprocessor.joblib")
    xgb = joblib.load("models/xgb_model.joblib")
    lgb = joblib.load("models/lgb_model.joblib")
    cat = joblib.load("models/cat_model.joblib")

    # Generate a dummy test sample
    df_sample = pd.DataFrame([{
        "depth_m": 300, "working_height_m": 2.5, "gallery_width_m": 4.2, "gallery_height_m": 2.5,
        "rmr_rating": 50, "q_value": 1.0, "gsi": 45, "rqd": 50, "rock_strength_mpa": 40,
        "joint_spacing_cm": 30, "water_seepage_lpm": 2.0, "roof_convergence_mm_day": 2.0,
        "roof_to_floor_convergence_mm_day": 3.0, "bed_separation_mm": 5.0,
        "support_density_bolts_per_m2": 1.2, "bolt_spacing_m": 1.0, "bolt_length_m": 1.8,
        "bolt_capacity_tonnes": 8.0, "blast_distance_m": 25.0, "blast_vibration_ppv": 5.0,
        "production_tpd": 500, "persons_exposed": 4, "temperature_c": 28, "humidity_pct": 80,
        "coal_noncoal": "Coal", "mineral": "Coal", "mining_method": "Bord and Pillar (Development)",
        "is_junction": 0, "face_position": "Development Face", "roof_lithology": "Sandstone",
        "floor_lithology": "Shale", "joint_orientation": "Fair", "joint_condition": "Tight/Rough",
        "water_condition": "Damp", "cable_bolts_installed": 0, "props_installed": 1,
        "support_condition": "Intact & Adequate", "recent_blasting": "No Blasting (<24h)",
        "previous_roof_fall": 0, "recent_geological_disturbance": 0
    }])

    X = prep.transform(df_sample)
    p_xgb = xgb.predict_proba(X)
    p_lgb = lgb.predict_proba(X)
    p_cat = cat.predict_proba(X)

    assert p_xgb.shape == (1, 4), "XGBoost proba shape mismatch"
    assert p_lgb.shape == (1, 4), "LightGBM proba shape mismatch"
    assert p_cat.shape == (1, 4), "CatBoost proba shape mismatch"
    assert np.isclose(np.sum(p_xgb), 1.0, atol=1e-3), "XGBoost probabilities do not sum to 1"


def test_model_agreement_engine():
    prep = PreprocessingManager.load("models/preprocessor.joblib")
    ag_engine = ModelAgreementEngine()

    df_sample = pd.DataFrame([{
        "depth_m": 300, "working_height_m": 2.5, "gallery_width_m": 4.2, "gallery_height_m": 2.5,
        "rmr_rating": 50, "q_value": 1.0, "gsi": 45, "rqd": 50, "rock_strength_mpa": 40,
        "joint_spacing_cm": 30, "water_seepage_lpm": 2.0, "roof_convergence_mm_day": 2.0,
        "roof_to_floor_convergence_mm_day": 3.0, "bed_separation_mm": 5.0,
        "support_density_bolts_per_m2": 1.2, "bolt_spacing_m": 1.0, "bolt_length_m": 1.8,
        "bolt_capacity_tonnes": 8.0, "blast_distance_m": 25.0, "blast_vibration_ppv": 5.0,
        "production_tpd": 500, "persons_exposed": 4, "temperature_c": 28, "humidity_pct": 80,
        "coal_noncoal": "Coal", "mineral": "Coal", "mining_method": "Bord and Pillar (Development)",
        "is_junction": 0, "face_position": "Development Face", "roof_lithology": "Sandstone",
        "floor_lithology": "Shale", "joint_orientation": "Fair", "joint_condition": "Tight/Rough",
        "water_condition": "Damp", "cable_bolts_installed": 0, "props_installed": 1,
        "support_condition": "Intact & Adequate", "recent_blasting": "No Blasting (<24h)",
        "previous_roof_fall": 0, "recent_geological_disturbance": 0
    }])

    X = prep.transform(df_sample)
    ag_res = ag_engine.evaluate_agreement(X)
    assert ag_res["ensemble_prediction"] in CLASS_NAMES
    assert ag_res["agreement_ratio"] in ["2/3", "3/3"]


def test_shap_explainer():
    prep = PreprocessingManager.load("models/preprocessor.joblib")
    explainer = BhoomiShapExplainer("XGBoost")

    df_sample = pd.DataFrame([{
        "depth_m": 450, "working_height_m": 3.0, "gallery_width_m": 5.0, "gallery_height_m": 3.0,
        "rmr_rating": 35, "q_value": 0.4, "gsi": 30, "rqd": 35, "rock_strength_mpa": 25,
        "joint_spacing_cm": 15, "water_seepage_lpm": 10.0, "roof_convergence_mm_day": 7.0,
        "roof_to_floor_convergence_mm_day": 9.5, "bed_separation_mm": 18.0,
        "support_density_bolts_per_m2": 0.8, "bolt_spacing_m": 1.3, "bolt_length_m": 1.8,
        "bolt_capacity_tonnes": 5.0, "blast_distance_m": 15.0, "blast_vibration_ppv": 15.0,
        "production_tpd": 500, "persons_exposed": 4, "temperature_c": 30, "humidity_pct": 90,
        "coal_noncoal": "Coal", "mineral": "Coal", "mining_method": "Bord and Pillar (Depillaring)",
        "is_junction": 1, "face_position": "Depillaring Edge / Goaf", "roof_lithology": "Shale",
        "floor_lithology": "Shale", "joint_orientation": "Unfavorable (Parallel to Drive)", "joint_condition": "Open/Slickensided",
        "water_condition": "Wet/Dripping", "cable_bolts_installed": 0, "props_installed": 1,
        "support_condition": "Excessive Load/Yielding", "recent_blasting": "Immediate (<4h)",
        "previous_roof_fall": 1, "recent_geological_disturbance": 1
    }])

    X = prep.transform(df_sample)
    shap_res = explainer.explain_instance(X, predicted_class=2)
    assert len(shap_res["top_contributors"]) == 5
    assert "contributed to the model prediction" in shap_res["safety_note"]
