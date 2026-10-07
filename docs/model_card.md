# BHOOmiKAVACH AI — Model Card

## 1. Model Details
- **Model Name:** BHOOmiKAVACH Prospective Roof-Fall Risk Classifier
- **Model Version:** `v1.2.0-champion` (Weighted Soft Voting Ensemble)
- **Base Estimators:** XGBoost (`XGBClassifier`), LightGBM (`LGBMClassifier`), CatBoost (`CatBoostClassifier`)
- **Meta-Framework:** scikit-learn, XGBoost, LightGBM, CatBoost, SHAP
- **Release Date:** October 2026
- **License:** Open Mining Safety Research License (Internal IIT/DGMS Decision Support)

---

## 2. Intended Use
- **Primary Objective:** Provide prospective risk assessment and decision support for ground control engineers, mine managers, and safety officers in Indian underground coal and metalliferous mines.
- **Decision Tier Classification:** 4 risk levels:
  - `LOW`: Standard strata equilibrium within design tolerances.
  - `MODERATE`: Minor localized stress or slight support deformation requiring routine monitoring.
  - `HIGH`: Significant strata dilation, joint slickensides, or suboptimal support density demanding engineering intervention.
  - `CRITICAL`: Acute convergence velocity, extensive bed separation, or active support failure demanding immediate personnel withdrawal.

## 3. Prohibited Use
- **Automated Entry Authorization:** The model MUST NEVER be used as autonomous software to authorize workers entering freshly blasted or unsupported faces.
- **Safety Certification Substitute:** The model output does NOT substitute for mandatory physical inspection by a statutory Overman, Mining Sirdar, or Mine Manager under CMR 2017 / MMR 1961.
- **Causal Guarantee:** Model probabilities denote estimated empirical risk scores, NOT deterministic physical guarantees of rock failure timing.

---

## 4. Input Features (84 Dimensions)
- **Geotechnical Ratings:** Depth of cover (`depth_m`), CMRI-ISM RMR (`rmr_rating`), Barton's Q (`q_value`), GSI, RQD, Uniaxial Rock Strength (`rock_strength_mpa`).
- **Mining Geometry:** Gallery width (`gallery_width_m`), working height, 4-way junction indicator (`is_junction`), face position.
- **Strata Discontinuities:** Joint spacing (`joint_spacing_cm`), joint condition (slickensides/gouged), water inflow condition (`water_seepage_lpm`).
- **Engineering Controls:** Roof bolt support density (`support_density_bolts_per_m2`), bolt spacing, bolt anchorage capacity (`bolt_capacity_tonnes`), cable bolts, props, support condition.
- **Dynamic Telemetry:** Roof convergence velocity (`roof_convergence_mm_day`), roof-to-floor convergence, dual-height extensometer bed separation (`bed_separation_mm`), blasting distance, vibration PPV (`blast_vibration_ppv`).

---

## 5. Performance Metrics (Held-Out Test Set)
- **Validation Methodology:** Chronological Time-Based Split (Train 70%, Val 15%, Test 15%). Zero temporal leakage.
- **Balanced Accuracy:** 0.710 – 0.738 across baseline and gradient boosted ensembles.
- **Macro F1-Score:** 0.728 – 0.749.
- **HIGH Hazard Recall:** **80.8%**
- **CRITICAL Hazard Recall:** **44.4% – 55.6%** (achieved despite extreme class imbalance: 3.3% of operational records).
- **False Negative Rate:** **13.1% – 14.7%** (catastrophic misclassification rate of dangerous strata as safe).
- **Mean Brier Calibration Score:** **0.075 – 0.089**.

---

## 6. Limitations & Uncertainty Handling
1. **Model Disagreement:** Divergence between XGBoost, LightGBM, and CatBoost triggers a prominent `"MODEL DISAGREEMENT — CONFIDENCE REDUCED"` flag.
2. **Out-of-Distribution Data:** Inputs violating physical mining envelopes (e.g. depth > 1200m, unseen mineral) are caught by the Isolation Forest OOD detector, issuing: `"INSUFFICIENT DATA — HUMAN/COMPETENT TECHNICAL ASSESSMENT REQUIRED"`.
3. **Sensor Latency & Drift:** Physical tell-tales and convergence stations require routine calibration. Mechanical inaccuracies directly propagate into feature vector noise.
