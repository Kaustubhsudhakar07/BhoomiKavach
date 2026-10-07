# BHOOmiKAVACH AI — Research Paper Outline

**Proposed Title:**  
*BHOOmiKAVACH AI: An Explainable AI and Agentic Decision-Support Framework for Mitigating Ground Movement and Roof-Fall Hazards in Indian Underground Mines*

**Target Venues:**  
- *Journal of Rock Mechanics and Geotechnical Engineering (Elsevier)*
- *Safety Science (Elsevier)*
- *International Journal of Mining Science and Technology*

---

## 1. Abstract
Roof falls account for over 40% of underground mining fatalities globally and remain the single leading cause of ground movement casualties in Indian collieries (DGMS Code 0111). Existing strata management workflows rely predominantly on retrospective inquiry or static empirical ratings (CMRI-ISM RMR). In this paper, we propose BHOOmiKAVACH AI, an explainable machine learning and multi-agent decision-support architecture. We extract and standardize 41 realized fatal roof-fall inquiry cases from official DGMS publications (2016–2022) with zero synthetic fabrication, and evaluate prospective gradient-boosted ensembles (XGBoost, LightGBM, CatBoost) across an 84-dimensional geotechnical feature space under strict zero-leakage temporal evaluation. The champion weighted soft-voting ensemble achieves 80.8% recall on high-hazard strata and 55.6% on critical failures. To address black-box opacity and regulatory compliance, we integrate SHAP feature attribution, an independent deterministic safety rule engine, and a LangGraph multi-agent orchestrator with FAISS vector retrieval. Our framework demonstrates that combining prospective machine learning with deterministic statutory guardrails provides an auditable, high-sensitivity decision-support tool for underground mine safety.

---

## 2. Structure of Proposed Manuscript
1. **Introduction:** Underground mining ground control challenges in India; DGMS statutory framework; limitation of static rating methods.
2. **Related Work:** Strata mechanics and empirical RMR; machine learning in rock mass characterization; explainable AI and multi-agent architectures in high-consequence industries.
3. **Data Provenance & Engineering:**
   - Official DGMS inquiry extraction methodology (Dataset A, 41 records).
   - Domain-grounded operational calibration (Dataset B, 1,500 records).
   - Zero-leakage chronological time-series splitting.
4. **Machine Learning & Ensemble Architecture:**
   - 84-dimensional geotechnical feature space.
   - Addressing severe class imbalance in critical failure states.
   - Comparative evaluation: Logistic Regression, XGBoost, LightGBM, CatBoost, Soft/Weighted/Stacking Ensembles.
5. **Explainability & Uncertainty Quantification:**
   - SHAP TreeExplainer feature attributions.
   - Multi-model agreement as an uncertainty metric.
   - Out-of-Distribution (OOD) boundary detection via Isolation Forests.
6. **Agentic Decision Support & Statutory RAG:**
   - LangGraph cyclic multi-agent workflow.
   - Grounding in Coal Mines Regulations 2017 & DGMS Technical Circulars.
   - Role-stratified actionable protocols across 7 mining designations.
7. **Experimental Results & Discussion:**
   - Hazard recall vs false negative tradeoffs.
   - Probability calibration reliability curves.
   - Comparison with traditional empirical trigger action response plans (TARP).
8. **Conclusion & Future Directions:** Summary of findings, colliery pilot integration, IoT microseismic stream ingestion.
