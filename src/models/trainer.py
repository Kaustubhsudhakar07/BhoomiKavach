"""
BHOOmiKAVACH AI — ML Model Training & Hyperparameter Tuning (Phases 9, 10, 12, 13)
===================================================================================
Trains and compares:
  1. Logistic Regression (Baseline)
  2. XGBoost (XGBClassifier)
  3. LightGBM (LGBMClassifier)
  4. CatBoost (CatBoostClassifier)

Features:
  - Class imbalance compensation via sklearn compute_class_weight & balanced weights
  - Local hyperparameter tuning using RandomizedSearchCV & validation tracking
  - Evaluation across all required metrics:
      Accuracy, Balanced Accuracy, Macro F1, Precision, Recall,
      HIGH Recall, CRITICAL Recall, False Negative Rate (FNR), PR-AUC, Confusion Matrix
  - Model serialization to `models/`

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, confusion_matrix
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.preprocessing.pipeline import prepare_temporal_datasets, CLASS_NAMES

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Per-class recalls
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    per_class_recalls = {}
    for idx, cname in enumerate(CLASS_NAMES):
        total_actual = cm[idx].sum()
        correct = cm[idx, idx]
        per_class_recalls[cname] = float(correct / total_actual) if total_actual > 0 else 0.0

    high_recall = per_class_recalls.get("HIGH", 0.0)
    critical_recall = per_class_recalls.get("CRITICAL", 0.0)

    # False negative rate for hazardous classes (HIGH + CRITICAL misclassified as LOW or MODERATE)
    hazardous_actual = np.isin(y_true, [2, 3])
    hazardous_predicted_safe = np.isin(y_pred, [0, 1])
    fn_count = np.sum(hazardous_actual & hazardous_predicted_safe)
    total_haz = np.sum(hazardous_actual)
    fnr = float(fn_count / total_haz) if total_haz > 0 else 0.0

    # Macro PR-AUC approximation
    pr_auc = 0.0
    if y_proba is not None:
        try:
            from sklearn.metrics import precision_recall_curve, auc
            pr_aucs = []
            for c in range(4):
                y_bin = (y_true == c).astype(int)
                if y_bin.sum() > 0:
                    p, r, _ = precision_recall_curve(y_bin, y_proba[:, c])
                    pr_aucs.append(auc(r, p))
            pr_auc = float(np.mean(pr_aucs)) if pr_aucs else 0.0
        except Exception:
            pr_auc = 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "precision": round(prec_macro, 4),
        "recall": round(rec_macro, 4),
        "high_recall": round(high_recall, 4),
        "critical_recall": round(critical_recall, 4),
        "false_negative_rate": round(fnr, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": cm.tolist(),
        "per_class_recalls": per_class_recalls
    }


def train_baseline_lr(X_train, y_train, class_weights_dict):
    print("\n--- Training Baseline (Logistic Regression) ---")
    start_t = time.time()
    lr = LogisticRegression(
        multi_class="multinomial",
        class_weight=class_weights_dict,
        max_iter=1000,
        random_state=42
    )
    lr.fit(X_train, y_train)
    train_time = round(time.time() - start_t, 3)
    joblib.dump(lr, os.path.join(MODELS_DIR, "baseline_lr.joblib"))
    return lr, train_time


def train_tuned_xgboost(X_train, y_train, sample_weights):
    print("\n--- Training & Tuning Model 1: XGBoost ---")
    start_t = time.time()
    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=4,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0,
        n_jobs=1
    )
    param_dist = {
        "n_estimators": [100, 180, 250],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.03, 0.07, 0.12],
        "subsample": [0.75, 0.9, 1.0],
        "colsample_bytree": [0.75, 0.85, 1.0],
        "min_child_weight": [1, 3, 5],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.5, 1.0, 2.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        xgb, param_distributions=param_dist,
        n_iter=6, cv=cv, scoring="f1_macro",
        random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train, sample_weight=sample_weights)
    best_xgb = search.best_estimator_
    train_time = round(time.time() - start_t, 3)
    print(f"  Best XGB Params: {search.best_params_} (Train Time: {train_time}s)")
    joblib.dump(best_xgb, os.path.join(MODELS_DIR, "xgb_model.joblib"))
    return best_xgb, train_time, search.best_params_


def train_tuned_lightgbm(X_train, y_train, class_weights_dict):
    print("\n--- Training & Tuning Model 2: LightGBM ---")
    start_t = time.time()
    lgb = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight=class_weights_dict,
        random_state=42,
        verbosity=-1,
        n_jobs=1
    )
    param_dist = {
        "n_estimators": [100, 180, 250],
        "max_depth": [-1, 6, 9],
        "num_leaves": [20, 31, 50],
        "learning_rate": [0.03, 0.08, 0.12],
        "min_child_samples": [10, 20, 35],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.75, 0.9],
        "reg_alpha": [0.0, 0.2, 0.8],
        "reg_lambda": [0.2, 1.0, 2.0]
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    search = RandomizedSearchCV(
        lgb, param_distributions=param_dist,
        n_iter=6, cv=cv, scoring="f1_macro",
        random_state=42, n_jobs=1
    )
    search.fit(X_train, y_train)
    best_lgb = search.best_estimator_
    train_time = round(time.time() - start_t, 3)
    print(f"  Best LightGBM Params: {search.best_params_} (Train Time: {train_time}s)")
    joblib.dump(best_lgb, os.path.join(MODELS_DIR, "lgb_model.joblib"))
    return best_lgb, train_time, search.best_params_


def train_tuned_catboost(X_train, y_train, X_val, y_val):
    print("\n--- Training & Tuning Model 3: CatBoost ---")
    start_t = time.time()
    
    # Custom validation search over key parameters to avoid clone/constructor issues
    param_candidates = [
        {"iterations": 150, "depth": 4, "learning_rate": 0.08, "l2_leaf_reg": 3.0},
        {"iterations": 200, "depth": 6, "learning_rate": 0.06, "l2_leaf_reg": 5.0},
        {"iterations": 220, "depth": 7, "learning_rate": 0.05, "l2_leaf_reg": 4.0},
        {"iterations": 180, "depth": 5, "learning_rate": 0.10, "l2_leaf_reg": 2.0},
    ]

    best_score = -1.0
    best_cat = None
    best_params = {}

    for params in param_candidates:
        cat = CatBoostClassifier(
            loss_function="MultiClass",
            auto_class_weights="Balanced",
            random_seed=42,
            verbose=False,
            thread_count=1,
            **params
        )
        cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)
        val_pred = cat.predict(X_val)
        if val_pred.ndim > 1:
            val_pred = val_pred.ravel()
        val_f1 = f1_score(y_val, val_pred, average="macro", zero_division=0)
        
        if val_f1 > best_score:
            best_score = val_f1
            best_cat = cat
            best_params = params

    train_time = round(time.time() - start_t, 3)
    print(f"  Best CatBoost Params: {best_params} (Val Macro F1: {best_score:.4f}, Train Time: {train_time}s)")
    joblib.dump(best_cat, os.path.join(MODELS_DIR, "cat_model.joblib"))
    return best_cat, train_time, best_params


def train_and_evaluate_all():
    data = prepare_temporal_datasets()
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # Compute class weights to counter imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_dict = {c: float(w) for c, w in zip(classes, weights)}
    sample_weights = np.array([class_weights_dict[y] for y in y_train])

    print("\nBalanced Class Weights:")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name} (Class {c}): {class_weights_dict.get(c, 1.0):.2f}")

    models = {}
    train_times = {}

    # 1. Baseline LR
    lr, t_lr = train_baseline_lr(X_train, y_train, class_weights_dict)
    models["Logistic Regression"] = lr
    train_times["Logistic Regression"] = t_lr

    # 2. XGBoost
    xgb, t_xgb, _ = train_tuned_xgboost(X_train, y_train, sample_weights)
    models["XGBoost"] = xgb
    train_times["XGBoost"] = t_xgb

    # 3. LightGBM
    lgb, t_lgb, _ = train_tuned_lightgbm(X_train, y_train, class_weights_dict)
    models["LightGBM"] = lgb
    train_times["LightGBM"] = t_lgb

    # 4. CatBoost
    cat, t_cat, _ = train_tuned_catboost(X_train, y_train, X_val, y_val)
    models["CatBoost"] = cat
    train_times["CatBoost"] = t_cat

    # Evaluate on Validation & Test Sets
    evaluation_records = []
    print("\n================== TEST SET EVALUATION ==================")
    for name, model in models.items():
        t0 = time.time()
        y_pred = model.predict(X_test)
        infer_time = round((time.time() - t0) / len(X_test) * 1000, 3) # ms per sample
        
        # Format 1D prediction array for catboost
        if y_pred.ndim > 1:
            y_pred = y_pred.ravel()

        y_proba = model.predict_proba(X_test)
        metrics = compute_comprehensive_metrics(y_test, y_pred, y_proba)
        
        print(f"\n[{name}]")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f} | Macro F1: {metrics['macro_f1']:.4f}")
        print(f"  HIGH Recall: {metrics['high_recall']:.4f} | CRITICAL Recall: {metrics['critical_recall']:.4f}")
        print(f"  False Negative Rate: {metrics['false_negative_rate']:.4f} | Inference Time: {infer_time} ms/sample")

        evaluation_records.append({
            "model": name,
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "high_recall": metrics["high_recall"],
            "critical_recall": metrics["critical_recall"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "pr_auc": metrics["pr_auc"],
            "false_negative_rate": metrics["false_negative_rate"],
            "training_time": train_times[name],
            "inference_time": infer_time
        })

    eval_df = pd.DataFrame(evaluation_records)
    eval_df.to_csv(os.path.join(RESULTS_DIR, "individual_models_evaluation.csv"), index=False)
    print(f"\nSaved evaluation metrics to {os.path.join(RESULTS_DIR, 'individual_models_evaluation.csv')}")
    return models, eval_df


if __name__ == "__main__":
    train_and_evaluate_all()
