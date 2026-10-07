"""
BHOOmiKAVACH AI — Probability Calibration Engine (Phase 16)
===========================================================
Evaluates the calibration quality of models across risk classes.
Generates:
  - Multiclass calibration curves (reliability diagrams)
  - Class-wise and overall Brier scores
  - Safety-compliant calibration report

Notice:
  "Model output represents a model-estimated risk score, NOT an absolute physical accident probability."

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from src.ensemble.ensemble_models import SoftVotingEnsemble, StackingEnsemble

from src.preprocessing.pipeline import prepare_temporal_datasets, CLASS_NAMES

MODELS_DIR = "models"
CALIB_DIR = "results/calibration"
os.makedirs(CALIB_DIR, exist_ok=True)


def evaluate_and_plot_calibration():
    data = prepare_temporal_datasets()
    X_test, y_test = data["X_test"], data["y_test"]

    models_to_eval = {
        "Logistic Regression": joblib.load(os.path.join(MODELS_DIR, "baseline_lr.joblib")),
        "XGBoost": joblib.load(os.path.join(MODELS_DIR, "xgb_model.joblib")),
        "LightGBM": joblib.load(os.path.join(MODELS_DIR, "lgb_model.joblib")),
        "CatBoost": joblib.load(os.path.join(MODELS_DIR, "cat_model.joblib")),
        "Weighted Ensemble": joblib.load(os.path.join(MODELS_DIR, "weighted_ensemble.joblib"))
    }

    # Evaluate HIGH & CRITICAL classes (Hazard Focus)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    target_classes = [(2, "HIGH Risk (Class 2)"), (3, "CRITICAL Risk (Class 3)")]

    calib_records = []

    for ax, (cls_idx, cls_label) in zip(axes, target_classes):
        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        y_bin = (y_test == cls_idx).astype(int)

        for name, model in models_to_eval.items():
            proba = model.predict_proba(X_test)[:, cls_idx]
            brier = brier_score_loss(y_bin, proba)
            
            calib_records.append({
                "model": name,
                "class_index": cls_idx,
                "class_name": CLASS_NAMES[cls_idx],
                "brier_score": round(float(brier), 4)
            })

            try:
                frac_pos, mean_pred = calibration_curve(y_bin, proba, n_bins=5, strategy="uniform")
                ax.plot(mean_pred, frac_pos, "s-", label=f"{name} (Brier={brier:.3f})")
            except Exception:
                pass

        ax.set_title(f"Reliability Diagram: {cls_label}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Model-Estimated Risk Score", fontsize=11)
        ax.set_ylabel("Observed Empirical Proportion", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9, loc="upper left")

    plt.suptitle("BHOOmiKAVACH AI — Prospective Model Probability Calibration", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plot_path = os.path.join(CALIB_DIR, "calibration_curves.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()

    cal_df = pd.DataFrame(calib_records)
    report_csv = os.path.join(CALIB_DIR, "brier_scores.csv")
    cal_df.to_csv(report_csv, index=False)

    print(f"Calibration curve saved -> {plot_path}")
    print(f"Brier score report saved -> {report_csv}")
    return cal_df


if __name__ == "__main__":
    evaluate_and_plot_calibration()
