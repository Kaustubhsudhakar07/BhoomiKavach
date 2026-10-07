"""
BHOOmiKAVACH AI — SHAP Explainability & Model Agreement Engine (Phases 17 & 18)
================================================================================
Capabilities:
  1. TreeExplainer / KernelExplainer for XGBoost, LightGBM, and CatBoost models.
  2. Global feature importance generation (bar plot, summary plot) -> `results/explainability/`
  3. Local single-sample explanation with top 5 contributors and positive/negative impact.
  4. Multi-model Agreement & Uncertainty Metric (3/3 agreement vs disagreement alert).

CRITICAL SCIENTIFIC & SAFETY PRINCIPLE:
  SHAP values explain "why the mathematical model assigned this prediction score".
  They must NEVER be described as "physical causes of an accident".
  Language: "contributed to the model prediction", NOT "caused the incident".

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Tuple, Optional

from src.preprocessing.pipeline import PreprocessingManager, CLASS_NAMES, TARGET_MAP

MODELS_DIR = "models"
EXPLAIN_DIR = "results/explainability"
os.makedirs(EXPLAIN_DIR, exist_ok=True)


class BhoomiShapExplainer:
    def __init__(self, model_name: str = "XGBoost"):
        self.model_name = model_name
        self.model = None
        self.explainer = None
        self.feature_names = []
        self._load()

    def _load(self):
        # Load Preprocessor to get feature names
        prep_mgr = PreprocessingManager.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
        self.feature_names = prep_mgr.feature_names_out

        if self.model_name == "XGBoost":
            self.model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_name == "LightGBM":
            self.model = joblib.load(os.path.join(MODELS_DIR, "lgb_model.joblib"))
            self.explainer = shap.TreeExplainer(self.model)
        elif self.model_name == "CatBoost":
            self.model = joblib.load(os.path.join(MODELS_DIR, "cat_model.joblib"))
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.model = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
            self.explainer = shap.TreeExplainer(self.model)

    def explain_instance(self, X_sample: np.ndarray, predicted_class: int = 2) -> Dict[str, Any]:
        """
        Generates local explanation for a single mine condition feature vector.
        predicted_class: index (0=LOW, 1=MODERATE, 2=HIGH, 3=CRITICAL)
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        shap_values = self.explainer.shap_values(X_sample)
        
        # Handle multiclass output structure
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, predicted_class]
        else:
            sv = shap_values[0]

        # Top 5 most influential features
        abs_weights = np.abs(sv)
        top_indices = np.argsort(abs_weights)[::-1][:5]

        contributors = []
        for idx in top_indices:
            feat_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
            val = float(sv[idx])
            direction = "INCREASED risk score" if val > 0 else "DECREASED risk score"
            contributors.append({
                "feature": feat_name,
                "shap_value": round(val, 4),
                "impact_direction": direction,
                "importance_magnitude": round(float(abs_weights[idx]), 4),
                "explanation_text": f"{feat_name} ({direction} by {abs(val):.3f})"
            })

        return {
            "model_explained": self.model_name,
            "explained_class": CLASS_NAMES[predicted_class],
            "top_contributors": contributors,
            "safety_note": "Features listed contributed to the model prediction score. Physical causality must be independently verified."
        }

    def generate_global_plots(self, X_background: np.ndarray, max_samples: int = 150):
        """Generates and saves global SHAP summary and bar plots."""
        X_sub = X_background[:max_samples]
        shap_values = self.explainer.shap_values(X_sub)

        # For multiclass, focus on HIGH Risk (class 2)
        target_cls = 2
        if isinstance(shap_values, list):
            sv_target = shap_values[target_cls]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv_target = shap_values[:, :, target_cls]
        else:
            sv_target = shap_values

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_target, X_sub,
            feature_names=self.feature_names,
            max_display=12,
            plot_type="bar",
            show=False
        )
        plt.title(f"SHAP Global Feature Importance ({self.model_name} — HIGH Risk)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        bar_path = os.path.join(EXPLAIN_DIR, f"{self.model_name.lower()}_shap_bar.png")
        plt.savefig(bar_path, dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv_target, X_sub,
            feature_names=self.feature_names,
            max_display=12,
            show=False
        )
        plt.title(f"SHAP Beeswarm Distribution ({self.model_name} — HIGH Risk)", fontsize=12, fontweight="bold")
        plt.tight_layout()
        beeswarm_path = os.path.join(EXPLAIN_DIR, f"{self.model_name.lower()}_shap_beeswarm.png")
        plt.savefig(beeswarm_path, dpi=200)
        plt.close()

        print(f"SHAP plots saved -> {bar_path} & {beeswarm_path}")


class ModelAgreementEngine:
    def __init__(self):
        self.xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
        self.lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_model.joblib"))
        self.cat = joblib.load(os.path.join(MODELS_DIR, "cat_model.joblib"))
        self.ensemble = joblib.load(os.path.join(MODELS_DIR, "weighted_ensemble.joblib"))

    def evaluate_agreement(self, X_sample: np.ndarray) -> Dict[str, Any]:
        """
        Computes individual predictions across the 3 primary models + ensemble
        and calculates agreement consensus.
        """
        if X_sample.ndim == 1:
            X_sample = X_sample.reshape(1, -1)

        p_xgb = int(self.xgb.predict(X_sample)[0])
        p_lgb = int(self.lgb.predict(X_sample)[0])
        p_cat_raw = self.cat.predict(X_sample)
        p_cat = int(p_cat_raw[0] if p_cat_raw.ndim == 1 else p_cat_raw[0, 0])
        p_ens = int(self.ensemble.predict(X_sample)[0])

        predictions = [p_xgb, p_lgb, p_cat]
        class_predictions = {
            "XGBoost": CLASS_NAMES[p_xgb],
            "LightGBM": CLASS_NAMES[p_lgb],
            "CatBoost": CLASS_NAMES[p_cat],
            "Ensemble": CLASS_NAMES[p_ens]
        }

        # Count modal agreement
        from collections import Counter
        counts = Counter(predictions)
        most_common_class, freq = counts.most_common(1)[0]
        agreement_ratio = f"{freq}/3"
        consensus_score = float(freq / 3.0)

        is_unanimous = (freq == 3)
        disagreement_warning = None
        confidence_level = "HIGH" if is_unanimous else ("MEDIUM" if freq == 2 else "LOW")

        if not is_unanimous:
            disagreement_warning = "MODEL DISAGREEMENT — CONFIDENCE REDUCED. Divergence detected among gradient boosted models."

        # Average class probabilities
        probas = self.ensemble.predict_proba(X_sample)[0]
        prob_dict = {CLASS_NAMES[i]: round(float(probas[i]), 4) for i in range(4)}

        return {
            "individual_predictions": class_predictions,
            "ensemble_prediction": CLASS_NAMES[p_ens],
            "agreement_ratio": agreement_ratio,
            "consensus_score": round(consensus_score, 2),
            "is_unanimous": is_unanimous,
            "confidence_level": confidence_level,
            "disagreement_warning": disagreement_warning,
            "risk_probabilities": prob_dict
        }


if __name__ == "__main__":
    from src.preprocessing.pipeline import prepare_temporal_datasets
    data = prepare_temporal_datasets()
    X_test = data["X_test"]

    print("Building global SHAP plots for XGBoost...")
    explainer = BhoomiShapExplainer("XGBoost")
    explainer.generate_global_plots(X_test, max_samples=100)

    print("\nTesting Model Agreement Engine on sample 0...")
    ag_engine = ModelAgreementEngine()
    ag_res = ag_engine.evaluate_agreement(X_test[0])
    print("Agreement result:", ag_res)
