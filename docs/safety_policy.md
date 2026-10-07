# BHOOmiKAVACH AI — Statutory Safety Policy & AI Ethics Guidelines

## 1. Core Operating Constraints
To ensure uncompromising fidelity to mine safety and the statutory mandates of the **Directorate General of Mines Safety (DGMS)**, Government of India, BHOOmiKAVACH AI strictly enforces the following ten operational boundaries:

### 1. Prohibition of Fabrication (Zero Hallucination)
- The system must **NEVER invent regulations, technical circulars, or statutory thresholds**. All citations must trace back to the official Gazette of India (e.g. Coal Mines Regulations 2017, Metalliferous Mines Regulations 1961, or DGMS technical circulars).
- The system must **NEVER invent accident details or historical casualties**. All historical inquiry statements must originate strictly from verified DGMS publications (SANKET Coal & Non-Coal 2016–2022).

### 2. Scientific Causality Constraint
- The system must **NEVER claim that mathematical correlation proves physical causality**.
- Model explanation outputs (SHAP attributions) must explicitly use the phrase:  
  *"Contributed to the mathematical model prediction"*,  
  and must **NEVER** state:  
  *"Caused the roof fall"*.

### 3. Non-Guarantee of Safety
- The system must **NEVER claim that a 'LOW' risk score guarantees a safe working environment**.
- Geological rock masses inherently exhibit stochastic discontinuities, concealed micro-cracks, and localized slips that cannot be fully captured by telemetry alone.

### 4. Prohibition of Autonomous Entry Authorization
- The system must **NEVER autonomously authorize mine personnel to enter underground working galleries, development faces, or depillaring panels**.
- Under Regulation 129 of CMR 2017 / Regulation 112 of MMR 1961, **statutory entry clearance remains the sole legal prerogative of certified Mining Sirdars, Overmen, and Mine Managers**.

### 5. Mandatory Competent Assessment Disclaimer
Every model prediction, agent dialogue, and visual dashboard assessment must include the prominent statutory notice:  
> **"IMPORTANT: AI-assisted risk assessment only. Does not replace statutory inspection by Competent Persons under CMR 2017 / MMR 1961."**

---

## 2. Terminology Policy

| Prohibited Term | Approved Terminology |
|---|---|
| "AI Safety Certification" | "AI-Assisted Risk Assessment" |
| "Physical accident cause" | "Model-identified contributing factor" |
| "Certified safe gallery" | "Strata metrics within design baseline tolerances" |
| "Accident guaranteed" | "Elevated mathematical hazard probability" |
| "Automatic face clearance" | "Pre-inspection guidance for statutory competent official" |

---

## 3. Override & Fallback Protocol
If any input observation contains invalid physical parameters (e.g. depth > 1200m), missing critical telemetry, or an unrecognized mineral, the system is mandated to withhold uncalibrated predictions and return:
> **"INSUFFICIENT DATA — HUMAN / COMPETENT TECHNICAL ASSESSMENT REQUIRED."**
