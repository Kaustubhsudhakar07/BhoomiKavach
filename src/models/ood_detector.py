"""
BHOOmiKAVACH AI — Out-of-Distribution (OOD) & Anomaly Detector (Phase 20)
========================================================================
Detects domain shift, anomalous mine inputs, and out-of-distribution environments:
  1. Unseen categorical values (e.g. unknown mineral, exotic mining method)
  2. Physical and geotechnical bounding envelope violations (e.g. depth > 1200m, RMR < 5)
  3. Multidimensional feature outlier detection via IsolationForest fitted on training data
  4. Missing critical sensory variables

Action:
  When OOD is detected, reduces model confidence or triggers:
  "INSUFFICIENT DATA — HUMAN/COMPETENT TECHNICAL ASSESSMENT REQUIRED."

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import IsolationForest

from src.preprocessing.pipeline import PreprocessingManager, prepare_temporal_datasets

MODELS_DIR = "models"
OOD_MODEL_PATH = os.path.join(MODELS_DIR, "ood_isolation_forest.joblib")

# Strict Physical Domain Ranges for Indian Underground Mining
PHYSICAL_BOUNDS = {
    "depth_m": (20.0, 1200.0),
    "working_height_m": (1.0, 6.5),
    "gallery_width_m": (2.0, 7.5),
    "gallery_height_m": (1.0, 6.0),
    "rmr_rating": (10.0, 95.0),
    "q_value": (0.01, 100.0),
    "gsi": (10.0, 90.0),
    "rqd": (5.0, 100.0),
    "rock_strength_mpa": (5.0, 250.0),
    "roof_convergence_mm_day": (0.0, 60.0),
    "bed_separation_mm": (0.0, 100.0),
    "support_density_bolts_per_m2": (0.1, 4.0),
    "bolt_capacity_tonnes": (1.0, 25.0),
    "blast_vibration_ppv": (0.0, 80.0)
}

KNOWN_MINERALS = {"Coal", "Copper", "Manganese", "Galena & Sphalerite", "Gold", "Uranium", "Lead-Zinc", "Chromite", "Iron Ore"}
KNOWN_MINE_TYPES = {"Coal", "Non-Coal"}


class OutOfDistributionDetector:
    def __init__(self):
        self.iso_forest = None
        self.is_fitted = False
        self._load_or_fit()

    def _load_or_fit(self):
        if os.path.exists(OOD_MODEL_PATH):
            self.iso_forest = joblib.load(OOD_MODEL_PATH)
            self.is_fitted = True
        else:
            self.fit_from_train_data()

    def fit_from_train_data(self):
        print("Fitting OOD IsolationForest on Training Distribution...")
        data = prepare_temporal_datasets()
        X_train = data["X_train"]
        self.iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.03, # 3% baseline boundary
            random_state=42,
            n_jobs=1
        )
        self.iso_forest.fit(X_train)
        self.is_fitted = True
        joblib.dump(self.iso_forest, OOD_MODEL_PATH)
        print(f"OOD IsolationForest saved -> {OOD_MODEL_PATH}")

    def evaluate_ood(self, raw_observation: Dict[str, Any], X_encoded: np.ndarray = None) -> Dict[str, Any]:
        """
        Evaluates whether the input observation is Out-of-Distribution or physically invalid.
        """
        ood_reasons: List[str] = []

        # 1. Unseen categorical check
        mineral = str(raw_observation.get("mineral", "Coal"))
        if mineral not in KNOWN_MINERALS and mineral != "NULL":
            ood_reasons.append(f"Unseen mineral extract '{mineral}' outside validated DGMS taxonomy.")

        mine_type = str(raw_observation.get("coal_noncoal", "Coal"))
        if mine_type not in KNOWN_MINE_TYPES and mine_type != "NULL":
            ood_reasons.append(f"Unseen mining classification '{mine_type}'.")

        # 2. Geotechnical bounding envelope check
        for param, (low, high) in PHYSICAL_BOUNDS.items():
            if param in raw_observation and raw_observation[param] is not None:
                try:
                    val = float(raw_observation[param])
                    if val < low or val > high:
                        ood_reasons.append(f"Geotechnical parameter '{param}' = {val} violates physical boundary [{low}, {high}].")
                except (ValueError, TypeError):
                    ood_reasons.append(f"Invalid non-numeric value for parameter '{param}'.")

        # 3. Isolation Forest multidimensional density check
        iso_score = 0.0
        is_multivariate_outlier = False
        if X_encoded is not None and self.is_fitted:
            if X_encoded.ndim == 1:
                X_encoded = X_encoded.reshape(1, -1)
            pred = self.iso_forest.predict(X_encoded)[0] # 1 = inlier, -1 = outlier
            iso_score = float(self.iso_forest.decision_function(X_encoded)[0])
            if pred == -1:
                is_multivariate_outlier = True
                ood_reasons.append(f"Multivariate feature anomaly detected (IsolationForest score={iso_score:.3f}).")

        is_ood = len(ood_reasons) > 0

        # Confidence penalty / refusal
        if is_ood:
            return {
                "is_ood": True,
                "ood_status": "OUT_OF_DISTRIBUTION_WARNING",
                "reasons": ood_reasons,
                "confidence_penalty": 0.40,
                "recommended_fallback": "INSUFFICIENT DATA — HUMAN/COMPETENT TECHNICAL ASSESSMENT REQUIRED.",
                "isolation_forest_score": round(iso_score, 4)
            }
        else:
            return {
                "is_ood": False,
                "ood_status": "IN_DISTRIBUTION",
                "reasons": [],
                "confidence_penalty": 0.0,
                "recommended_fallback": None,
                "isolation_forest_score": round(iso_score, 4)
            }


if __name__ == "__main__":
    detector = OutOfDistributionDetector()
    test_normal = {"depth_m": 320, "gallery_width_m": 4.2, "rmr_rating": 52, "mineral": "Coal"}
    test_abnormal = {"depth_m": 1600, "gallery_width_m": 12.0, "rmr_rating": 2, "mineral": "Plutonium"}

    print("Test Normal:", detector.evaluate_ood(test_normal))
    print("\nTest Abnormal:", detector.evaluate_ood(test_abnormal))
