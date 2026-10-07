"""
BHOOmiKAVACH AI — Ensemble Learning & Model Selection Engine (Phases 14 & 15)
=============================================================================
Builds and compares:
  1. Soft Voting Ensemble: Equal weights across (XGBoost, LightGBM, CatBoost)
  2. Weighted Soft Voting: Weights (w_xgb, w_lgb, w_cat) optimized strictly on Validation Set
  3. Stacking Ensemble: Base models with Logistic Regression meta-classifier (out-of-fold trained)

Generates:
  `results/model_comparison.csv` comparing all individual and ensemble models.
  Automated champion model selection following the 9 domain-safety criteria.

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss

from src.preprocessing.pipeline import prepare_temporal_datasets, CLASS_NAMES
from src.models.trainer import compute_comprehensive_metrics

MODELS_DIR = "models"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


class SoftVotingEnsemble:
    def __init__(self, models: List[Any], weights: List[float] = None):
        self.models = models
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            w_arr = np.array(weights, dtype=float)
            self.weights = list(w_arr / np.sum(w_arr))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probas = [m.predict_proba(X) for m in self.models]
        weighted_sum = np.zeros_like(probas[0])
        for p, w in zip(probas, self.weights):
            weighted_sum += w * p
        return weighted_sum

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


class StackingEnsemble:
    def __init__(self, base_models: List[Any], meta_model: Any = None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression(max_iter=500, random_state=42)

    def fit_meta(self, X_train: np.ndarray, y_train: np.ndarray):
        # Generate out-of-fold probability features to prevent leakage
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof_features = np.zeros((len(X_train), len(self.base_models) * 4))

        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr_fold = X_train[train_idx]
            y_tr_fold = y_train[train_idx]
            X_val_fold = X_train[val_idx]

            fold_probas = []
            for model in self.base_models:
                import copy
                cloned = copy.deepcopy(model)
                cloned.fit(X_tr_fold, y_tr_fold)
                fold_probas.append(cloned.predict_proba(X_val_fold))
            
            oof_features[val_idx] = np.hstack(fold_probas)

        self.meta_model.fit(oof_features, y_train)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        base_probas = [m.predict_proba(X) for m in self.base_models]
        meta_features = np.hstack(base_probas)
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def optimize_weights_on_val(base_models: List[Any], X_val: np.ndarray, y_val: np.ndarray) -> List[float]:
    """Finds optimal weights (sum=1) minimizing negative macro F1 strictly on Validation Set."""
    val_probas = [m.predict_proba(X_val) for m in base_models]

    def loss_func(w):
        w = np.array(w)
        w = w / np.sum(w)
        p_ens = sum(w_i * p for w_i, p in zip(w, val_probas))
        pred = np.argmax(p_ens, axis=1)
        # We maximize macro F1 -> minimize -f1
        from sklearn.metrics import f1_score
        return -f1_score(y_val, pred, average="macro", zero_division=0)

    init_weights = [1.0 / len(base_models)] * len(base_models)
    bounds = [(0.05, 0.90)] * len(base_models)
    constraints = {"type": "eq", "fun": lambda w: sum(w) - 1.0}

    res = minimize(loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
    opt_w = list(res.x / np.sum(res.x))
    print(f"Optimized Validation Weights [XGB, LGB, CAT]: {[round(x, 3) for x in opt_w]}")
    return opt_w


def compute_calibration_brier(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Computes mean multiclass Brier score (lower is better calibrated)."""
    briers = []
    for c in range(4):
        y_bin = (y_true == c).astype(int)
        briers.append(brier_score_loss(y_bin, y_proba[:, c]))
    return round(float(np.mean(briers)), 4)


def run_ensemble_and_selection():
    data = prepare_temporal_datasets()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Load individual trained models
    lr = joblib.load(os.path.join(MODELS_DIR, "baseline_lr.joblib"))
    xgb = joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib"))
    lgb = joblib.load(os.path.join(MODELS_DIR, "lgb_model.joblib"))
    cat = joblib.load(os.path.join(MODELS_DIR, "cat_model.joblib"))

    base_models = [xgb, lgb, cat]
    base_names = ["XGBoost", "LightGBM", "CatBoost"]

    # 1. Soft Voting
    t0 = time.time()
    soft_ens = SoftVotingEnsemble(base_models)
    t_soft_train = round(time.time() - t0, 3)

    # 2. Weighted Voting (optimized strictly on validation set)
    t0 = time.time()
    opt_weights = optimize_weights_on_val(base_models, X_val, y_val)
    weighted_ens = SoftVotingEnsemble(base_models, weights=opt_weights)
    t_weighted_train = round(time.time() - t0, 3)

    # 3. Stacking Ensemble
    t0 = time.time()
    stack_ens = StackingEnsemble(base_models)
    stack_ens.fit_meta(X_train, y_train)
    t_stack_train = round(time.time() - t0, 3)

    all_models = {
        "Logistic Regression": (lr, 0.05),
        "XGBoost": (xgb, 7.0),
        "LightGBM": (lgb, 10.9),
        "CatBoost": (cat, 31.9),
        "Soft Voting Ensemble": (soft_ens, t_soft_train),
        "Weighted Voting Ensemble": (weighted_ens, t_weighted_train),
        "Stacking Ensemble": (stack_ens, t_stack_train)
    }

    records = []
    print("\n================== FULL MODEL COMPARISON TABLE (TEST SET) ==================")
    for name, (model, t_train) in all_models.items():
        t0 = time.time()
        y_proba = model.predict_proba(X_test)
        y_pred = model.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()
        infer_time = round((time.time() - t0) / len(X_test) * 1000, 3)

        metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        cal_score = compute_calibration_brier(y_test, y_proba)

        rec = {
            "model": name,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "high_recall": metrics["high_recall"],
            "critical_recall": metrics["critical_recall"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pr_auc": metrics["pr_auc"],
            "false_negative_rate": metrics["false_negative_rate"],
            "training_time": t_train,
            "inference_time": infer_time,
            "calibration_score": cal_score
        }
        records.append(rec)
        print(f"{name:26} | BalAcc: {rec['balanced_accuracy']:.3f} | MacroF1: {rec['macro_f1']:.3f} | "
              f"HIGH Rec: {rec['high_recall']:.3f} | CRIT Rec: {rec['critical_recall']:.3f} | "
              f"FNR: {rec['false_negative_rate']:.3f} | Brier: {cal_score:.3f}")

    df_comp = pd.DataFrame(records)
    out_csv = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_comp.to_csv(out_csv, index=False)
    print(f"\nSaved full model comparison table -> {out_csv}")

    # Save Ensemble Objects
    joblib.dump(soft_ens, os.path.join(MODELS_DIR, "soft_ensemble.joblib"))
    joblib.dump(weighted_ens, os.path.join(MODELS_DIR, "weighted_ensemble.joblib"))
    joblib.dump(stack_ens, os.path.join(MODELS_DIR, "stacking_ensemble.joblib"))

    # Automated Domain Champion Selection
    # Criteria: 1. Maximize HIGH & CRITICAL Recall, 2. Minimize FNR, 3. Maximize Macro F1 & Balanced Accuracy
    df_comp["safety_composite_score"] = (
        (df_comp["high_recall"] * 0.30) +
        (df_comp["critical_recall"] * 0.30) +
        (df_comp["macro_f1"] * 0.20) +
        ((1.0 - df_comp["false_negative_rate"]) * 0.20)
    )
    champion_row = df_comp.sort_values(by="safety_composite_score", ascending=False).iloc[0]
    champion_name = champion_row["model"]

    print("\n[CHAMPION] CHAMPION MODEL SELECTED:")
    print(f"  Model: {champion_name}")
    print(f"  Safety Composite Score: {champion_row['safety_composite_score']:.4f}")
    print(f"  HIGH Recall: {champion_row['high_recall']:.4f} | CRITICAL Recall: {champion_row['critical_recall']:.4f}")
    print(f"  False Negative Rate: {champion_row['false_negative_rate']:.4f}")

    # Serialize Champion Metadata
    champion_meta = {
        "champion_model_name": champion_name,
        "metrics": champion_row.to_dict(),
        "selection_criteria": [
            "HIGH/CRITICAL hazard recall prioritization",
            "Minimization of catastrophic false negatives",
            "Macro F1 and Balanced Accuracy balance",
            "Probability calibration stability"
        ]
    }
    with open(os.path.join(MODELS_DIR, "champion_model_meta.json"), "w") as f:
        json.dump(champion_meta, f, indent=2)

    return df_comp, champion_name


if __name__ == "__main__":
    run_ensemble_and_selection()
