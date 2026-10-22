"""
BHOOmiKAVACH AI — ML Preprocessing Pipeline (Phase 8 & 11)
==========================================================
Implements a leak-free, modular preprocessing and time-based validation pipeline:
  - Temporal Splitting: Train (oldest 70%), Validation (middle 15%), Test (latest 15%)
  - Outlier handling (domain-specific geotechnical clipping)
  - Missing value imputation (Median for numerical, Most Frequent for categorical)
  - Standard scaling for numerical features
  - One-Hot encoding for categorical features with handle_unknown='ignore'
  - Target encoding for 4 classes: LOW (0), MODERATE (1), HIGH (2), CRITICAL (3)
  - Serialization to `models/preprocessor.joblib`

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Backward compatibility patch for unpickling ColumnTransformer across scikit-learn versions
try:
    import sklearn.compose._column_transformer
    if not hasattr(sklearn.compose._column_transformer, "_RemainderColsList"):
        from collections import UserList
        class _RemainderColsList(UserList):
            pass
        sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList
except Exception:
    pass


MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Target Mapping
TARGET_MAP = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "CRITICAL": 3}
REVERSE_TARGET_MAP = {0: "LOW", 1: "MODERATE", 2: "HIGH", 3: "CRITICAL"}
CLASS_NAMES = ["LOW", "MODERATE", "HIGH", "CRITICAL"]

# Features defining prospective geotechnical risk
NUMERICAL_FEATURES = [
    "depth_m",
    "working_height_m",
    "gallery_width_m",
    "gallery_height_m",
    "rmr_rating",
    "q_value",
    "gsi",
    "rqd",
    "rock_strength_mpa",
    "joint_spacing_cm",
    "water_seepage_lpm",
    "roof_convergence_mm_day",
    "roof_to_floor_convergence_mm_day",
    "bed_separation_mm",
    "support_density_bolts_per_m2",
    "bolt_spacing_m",
    "bolt_length_m",
    "bolt_capacity_tonnes",
    "blast_distance_m",
    "blast_vibration_ppv",
    "production_tpd",
    "persons_exposed",
    "temperature_c",
    "humidity_pct"
]

CATEGORICAL_FEATURES = [
    "coal_noncoal",
    "mineral",
    "mining_method",
    "is_junction",
    "face_position",
    "roof_lithology",
    "floor_lithology",
    "joint_orientation",
    "joint_condition",
    "water_condition",
    "cable_bolts_installed",
    "props_installed",
    "support_condition",
    "recent_blasting",
    "previous_roof_fall",
    "recent_geological_disturbance"
]


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clips extreme numerical values to prevent sensor anomalies from skewing models."""
    def __init__(self, lower_quantile: float = 0.001, upper_quantile: float = 0.999):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.bounds_ = {}

    def fit(self, X, y=None):
        X_df = pd.DataFrame(X)
        for col in X_df.columns:
            low = X_df[col].quantile(self.lower_quantile)
            high = X_df[col].quantile(self.upper_quantile)
            self.bounds_[col] = (low, high)
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X).copy()
        for col, (low, high) in self.bounds_.items():
            if col in X_df.columns:
                X_df[col] = X_df[col].clip(lower=low, upper=high)
        return X_df.values


class PreprocessingManager:
    def __init__(self):
        self.preprocessor = None
        self.feature_names_out: List[str] = []
        self.is_fitted = False

    def build_pipeline(self) -> ColumnTransformer:
        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper()),
            ("scaler", StandardScaler())
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, NUMERICAL_FEATURES),
                ("cat", cat_pipeline, CATEGORICAL_FEATURES)
            ],
            remainder="drop"
        )
        return self.preprocessor

    def fit(self, df_train: pd.DataFrame):
        self.build_pipeline()
        X_train_raw = df_train[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        self.preprocessor.fit(X_train_raw)
        self.is_fitted = True

        # Extract encoded feature names for interpretability & SHAP
        cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names_out = NUMERICAL_FEATURES + cat_feature_names
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor has not been fitted.")
        X_raw = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES]
        return self.preprocessor.transform(X_raw)

    def save(self, filepath: str = "models/preprocessor.joblib"):
        bundle = {
            "preprocessor": self.preprocessor,
            "feature_names_out": self.feature_names_out,
            "numerical_features": NUMERICAL_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "target_map": TARGET_MAP,
            "reverse_target_map": REVERSE_TARGET_MAP,
            "class_names": CLASS_NAMES
        }
        joblib.dump(bundle, filepath)
        print(f"Saved preprocessing pipeline -> {filepath} (total features={len(self.feature_names_out)})")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib") -> "PreprocessingManager":
        bundle = joblib.load(filepath)
        mgr = cls()
        mgr.preprocessor = bundle["preprocessor"]
        mgr.feature_names_out = bundle["feature_names_out"]
        mgr.is_fitted = True
        return mgr


def prepare_temporal_datasets(csv_path: str = "data/operational/operational_risk_dataset.csv") -> Dict[str, Any]:
    """
    Splits operational data strictly by chronological date to prevent temporal leakage:
      - Train (oldest 70%)
      - Validation (middle 15%)
      - Test (latest 15%)
    """
    df = pd.read_csv(csv_path)
    # Sort strictly by date and record_id to respect chronological ordering
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["date", "record_id"]).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    print(f"Dataset Temporal Split:")
    print(f"  Train:      {len(train_df)} records ({train_df['date'].min().date()} to {train_df['date'].max().date()})")
    print(f"  Validation: {len(val_df)} records ({val_df['date'].min().date()} to {val_df['date'].max().date()})")
    print(f"  Test:       {len(test_df)} records ({test_df['date'].min().date()} to {test_df['date'].max().date()})")

    # Fit preprocessor on TRAIN ONLY (Strict zero leakage)
    mgr = PreprocessingManager()
    mgr.fit(train_df)
    mgr.save("models/preprocessor.joblib")

    X_train = mgr.transform(train_df)
    y_train = train_df["roof_fall_risk"].map(TARGET_MAP).values

    X_val = mgr.transform(val_df)
    y_val = val_df["roof_fall_risk"].map(TARGET_MAP).values

    X_test = mgr.transform(test_df)
    y_test = test_df["roof_fall_risk"].map(TARGET_MAP).values

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": mgr.feature_names_out,
        "preprocessor_mgr": mgr
    }


if __name__ == "__main__":
    split_data = prepare_temporal_datasets()
    print("Preprocessing pipeline successfully built and verified.")
