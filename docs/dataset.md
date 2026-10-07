# BHOOmiKAVACH AI — Dataset Specification & Data Provenance

## 1. Segregation of Datasets

BHOOmiKAVACH AI maintains strict architectural segregation between two distinct datasets:

```
                            BHOOmiKAVACH DATA ECOSYSTEM
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
      DATASET A (HISTORICAL)                          DATASET B (OPERATIONAL)
    • 41 Realized Fatal DGMS Cases                  • 1,500 Shift Inspection Records
    • 100% Grounded in Official PDFs                • Domain-Calibrated Physics
    • Zero Synthetic Hallucination                  • CMRI-ISM & Statutory Distributions
    • Used for: Root Causes, Case Precedents,       • Used for: Training Prospective ML
      and FAISS Similarity Search                     Classifiers (XGB, LGB, CAT)
```

---

## 2. Dataset A: Historical DGMS Fatal Accident Intelligence
- **Source Documents:** 
  - *SANKET: Coal Mine Accidents (2016–2022)*, Directorate General of Mines Safety, Ministry of Labour & Employment.
  - *SANKET: Non-Coal Mine Accidents (2016–2022)*, Directorate General of Mines Safety, Ministry of Labour & Employment.
- **Accident Code:** `0111 = Fall of Roof` (Major Cause: `0100 Ground Movement`).
- **Total Records:** 41 fatal incidents.
  - Coal Mines: 30 records (73.2%)
  - Non-Coal Mines: 11 records (26.8%)
- **Total Fatalities Recorded:** 58 lives lost.
- **Canonical Schema:** Exactly 30 standardized fields (see `docs/data_dictionary.md`).
- **Missing Value Handling:** Missing inquiry attributes are strictly set to `NULL` to prevent fabrication.

---

## 3. Dataset B: Operational Prospective Monitoring Dataset
- **Domain Baseline:** Calibrated against Indian rock mechanics literature (Bieniawski RMR, CMRI-ISM Rock Mass Rating for Indian Coal Measures by Venkateswarlu & Ghose).
- **Total Records:** 1,500 chronologically sequenced observations across 14 representative Indian coal and metalliferous mine profiles.
- **Target Variable:** `roof_fall_risk` with 4 balanced-weighted tiers:
  - `LOW`: 24.8%
  - `MODERATE`: 47.4%
  - `HIGH`: 24.5%
  - `CRITICAL`: 3.3%
- **Transparency Disclosure:** Explicitly tagged with column `data_label = "SYNTHETIC DATA — NOT REAL MINE OBSERVATIONS"`.
