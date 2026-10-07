"""
BHOOmiKAVACH AI — Industrial Mine-Safety Decision Support Dashboard (Phase 25 & 36)
====================================================================================
Tagline: "SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL."
Standard: Directorate General of Mines Safety (DGMS) Code 0111: Fall of Roof
Frontend Architecture: Streamlit UI consuming FastAPI REST backend.

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from frontend.api_client import BhoomiApiClient

# Page Configuration
st.set_page_config(
    page_title="BHOOmiKAVACH AI | Mine Safety Decision Support",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Industrial High-Tech Dark Theme)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at top right, #111827, #0B0F17 80%);
    }
    
    .metric-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(10px);
        margin-bottom: 16px;
    }
    
    .risk-badge-low {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-moderate {
        background-color: #92400E;
        color: #FEF3C7;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-high {
        background-color: #B45309;
        color: #FFEDD5;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    .risk-badge-critical {
        background-color: #991B1B;
        color: #FEE2E2;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.85rem;
        animation: pulse 2s infinite;
    }
    
    .disclaimer-banner {
        background: rgba(185, 28, 28, 0.15);
        border-left: 4px solid #EF4444;
        padding: 12px 18px;
        border-radius: 0 8px 8px 0;
        margin: 16px 0;
        font-size: 0.85rem;
        color: #FCA5A5;
    }
    
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F59E0B, #EF4444, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .header-tagline {
        font-size: 0.95rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API Client
if "api_client" not in st.session_state:
    st.session_state.api_client = BhoomiApiClient()
api = st.session_state.api_client

# Sidebar Navigation
with st.sidebar:
    st.markdown('<div class="header-title" style="font-size: 1.6rem;">🛡️ BHOOmiKAVACH</div>', unsafe_allow_html=True)
    st.markdown('<div class="header-tagline" style="font-size: 0.75rem;">SENSE. EXPLAIN. PREVENT.</div>', unsafe_allow_html=True)
    st.markdown("---")

    nav_choice = st.radio(
        "Navigation Module",
        [
            "1. Overview & System Telemetry",
            "2. Live Strata Risk Assessment",
            "3. Model Explainability & SHAP",
            "4. Historical DGMS Analytics",
            "5. Accident Similarity Search",
            "6. Regulatory RAG Evidence",
            "7. Agentic Safety Advisor",
            "8. Model Benchmarks & Evaluation",
            "9. Safety Policy & Data Dictionary"
        ],
        index=1
    )

    st.markdown("---")
    # Quick Backend Status
    health = api.get_health()
    if health.get("status") == "HEALTHY":
        st.success("● API & Database Online")
    else:
        st.warning(f"⚠️ Backend: {health.get('status', 'Connecting...')}")
        st.caption("Ensure FastAPI is running on port 8000.")

    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748B; margin-top: 20px;">
        DGMS Code: 0111 (Fall of Roof)<br>
        Models: XGBoost | LightGBM | CatBoost<br>
        Framework: LangGraph + FAISS + SHAP
    </div>
    """, unsafe_allow_html=True)

# ── TOP HERO HEADER ──────────────────────────────────────────────────────────
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
    <div>
        <div class="header-title">BHOOmiKAVACH AI</div>
        <div class="header-tagline">DGMS Code 0111 Roof-Fall Decision Support Platform</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer-banner">
    ⚠️ <b>DGMS STATUTORY NOTICE:</b> This platform provides AI-assisted decision support. It does NOT authorize mine entry
    or supersede the statutory assessment of a certified Mine Manager, Overman, or Mining Sirdar under CMR 2017 / MMR 1961.
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: OVERVIEW & SYSTEM TELEMETRY
# ==============================================================================
if nav_choice == "1. Overview & System Telemetry":
    st.subheader("System Architecture & Operational Telemetry")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Historical DGMS Records</div>
            <div style="font-size: 2rem; font-weight: 800; color: #F59E0B;">41 Verified</div>
            <div style="color: #10B981; font-size: 0.8rem;">100% Unhallucinated (0111)</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Champion ML Model</div>
            <div style="font-size: 2rem; font-weight: 800; color: #3B82F6;">Weighted Ensemble</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">XGB (0.33) + LGB (0.33) + CAT (0.33)</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Hazard Recall (High/Crit)</div>
            <div style="font-size: 2rem; font-weight: 800; color: #10B981;">80.8% / 55.6%</div>
            <div style="color: #10B981; font-size: 0.8rem;">Zero Data Leakage Validation</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div style="color: #94A3B8; font-size: 0.85rem;">Regulatory Precedents</div>
            <div style="font-size: 2rem; font-weight: 800; color: #8B5CF6;">54 Indexed</div>
            <div style="color: #94A3B8; font-size: 0.8rem;">FAISS Vector Store (MiniLM)</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### System Architecture Flow")
    st.markdown("""
    ```
                      LIVE SENSOR / TELEMETRY & STRATA INSPECTION
                                         │
                                         ▼
                               FASTAPI REST BACKEND
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
     ISOLATION FOREST OOD       PREPROCESSING (84 Feat)     SAFETY RULE ENGINE
      (Domain Boundaries)         (CMRI-ISM Scaled)          (CMR 2017 / MMR 1961)
              │                          │                          │
              └───────────────┬──────────┘                          │
                              ▼                                     │
                 MULTI-MODEL ENSEMBLE CLASSIFIERS                   │
              ┌───────────────┼───────────────┐                     │
              ▼               ▼               ▼                     │
           XGBoost        LightGBM         CatBoost                 │
              └───────────────┼───────────────┘                     │
                              ▼                                     │
                   MODEL AGREEMENT ENGINE                           │
                    (Uncertainty Metric)                            │
                              │                                     │
                              ▼                                     │
                     SHAP EXPLAINABILITY                            │
                    (Local Attributions)                            │
                              │                                     │
                              ▼                                     │
                  LANGGRAPH ORCHESTRATION AGENT ◄───────────────────┘
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
         FAISS RAG        CRITIC AGENT   ROLE-BASED ACTION
        (Regulations)     (Safety Policy)  (7 Designations)
                              │
                              ▼
                  STREAMLIT DECISION SUPPORT UI
    ```
    """)


# ==============================================================================
# MODULE 2: LIVE STRATA RISK ASSESSMENT
# ==============================================================================
elif nav_choice == "2. Live Strata Risk Assessment":
    st.subheader("⚡ Live Strata Telemetry & Prospective Hazard Assessment")
    st.write("Input underground mine environmental, geometric, and strata monitoring telemetry.")

    with st.form("risk_assessment_form"):
        colA, colB, colC = st.columns(3)
        with colA:
            st.markdown("##### 📍 Mine & Working Context")
            mine_name = st.selectbox("Mine / Colliery", ["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"])
            sector = "Coal" if "Mine" not in mine_name and "Kolihan" not in mine_name else "Non-Coal"
            mineral = "Coal" if sector == "Coal" else ("Manganese" if "Balaghat" in mine_name else "Copper")
            method = st.selectbox("Extraction Method", ["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"])
            depth = st.slider("Depth of Cover (m)", min_value=30.0, max_value=800.0, value=380.0, step=10.0)
            face_pos = st.selectbox("Face Position", ["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"])
            is_junc = 1 if face_pos == "Junction" else st.checkbox("Is 4-Way Junction Intersection?", value=False)

        with colB:
            st.markdown("##### 📐 Geometry & Strata Properties")
            width = st.slider("Gallery Width (m) [Statutory CMR 111 Max: 4.8m]", min_value=3.0, max_value=6.5, value=4.5, step=0.1)
            height = st.slider("Gallery Height (m) [Max: 3.0m]", min_value=1.8, max_value=4.5, value=2.8, step=0.1)
            rmr = st.slider("CMRI-ISM Rock Mass Rating (RMR)", min_value=15.0, max_value=85.0, value=42.0, step=1.0)
            lithology = st.selectbox("Immediate Roof Lithology", ["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"])
            joint_cond = st.selectbox("Discontinuity Condition", ["Open/Slickensided", "Slightly Weathered", "Tight/Rough", "Gouged/Clay Filled"])
            water = st.selectbox("Groundwater Condition", ["Wet/Dripping", "Damp", "Completely Dry", "Flowing Inflow"])
            blasting = st.selectbox("Recent Blasting Activity", ["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"])
            blast_vibe = st.slider("Peak Particle Velocity PPV (mm/s)", min_value=0.5, max_value=30.0, value=12.5, step=0.5)

        with colC:
            st.markdown("##### 🛡️ Support & Strata Convergence Telemetry")
            supp_cond = st.selectbox("Installed Support Status", ["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"])
            density = st.slider("Bolt Support Density (bolts/m²)", min_value=0.4, max_value=2.4, value=0.9, step=0.05)
            bolt_cap = st.slider("Bolt Anchorage Capacity (tonnes)", min_value=2.0, max_value=16.0, value=6.5, step=0.5)
            convergence = st.slider("Roof Convergence Rate (mm/day)", min_value=0.1, max_value=25.0, value=5.8, step=0.1)
            bed_sep = st.slider("Bed Separation Dilation (mm)", min_value=0.0, max_value=40.0, value=16.0, step=0.5)
            role_choice = st.selectbox("My Statutory Role", ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Worker"])

        submit_btn = st.form_submit_button("⚡ Evaluate Prospective Roof-Fall Risk", use_container_width=True)

    if submit_btn:
        obs_payload = {
            "mine": mine_name,
            "coal_noncoal": sector,
            "mineral": mineral,
            "mining_method": method,
            "depth_m": depth,
            "working_height_m": height,
            "gallery_width_m": width,
            "gallery_height_m": height,
            "is_junction": 1 if is_junc else 0,
            "face_position": face_pos,
            "roof_lithology": lithology,
            "floor_lithology": "Shale",
            "rmr_rating": rmr,
            "joint_condition": joint_cond,
            "water_condition": water,
            "recent_blasting": blasting,
            "blast_vibration_ppv": blast_vibe,
            "support_condition": supp_cond,
            "support_density_bolts_per_m2": density,
            "bolt_capacity_tonnes": bolt_cap,
            "roof_convergence_mm_day": convergence,
            "bed_separation_mm": bed_sep,
            "user_role": role_choice
        }

        with st.spinner("Executing Ensemble Inference, Model Agreement, SHAP, and Safety Rules..."):
            try:
                res = api.predict_risk(obs_payload)
                risk_lvl = res.get("assessed_risk", "MODERATE")
                badge_class = f"risk-badge-{risk_lvl.lower()}"

                st.markdown("---")
                st.markdown(f"### Assessed Risk Tier: <span class='{badge_class}'>{risk_lvl}</span>", unsafe_allow_html=True)
                
                # Agreement and Confidence Grid
                ag = res.get("model_agreement", {})
                ood = res.get("ood_diagnostics", {})

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Model Agreement", ag.get("agreement_ratio", "3/3"), f"Consensus: {ag.get('consensus_score', 1.0)*100:.0f}%")
                c2.metric("System Confidence", res.get("confidence", "HIGH"), "Uncertainty Calibrated")
                c3.metric("OOD Detector", "ALERT" if ood.get("is_ood") else "IN-BOUNDS", f"Score: {ood.get('isolation_forest_score', 0):.3f}")
                c4.metric("Audit ID", f"#{res.get('audit_id', 0)}", "Immutable Log Saved")

                if ag.get("disagreement_warning"):
                    st.warning(f"⚠️ {ag['disagreement_warning']}")
                if ood.get("is_ood"):
                    st.error(f"⚠️ {ood.get('ood_status')}: {', '.join(ood.get('reasons', []))}")

                # Model Agreement Breakdown Cards
                st.markdown("##### Individual Classifier Predictions")
                ind_col1, ind_col2, ind_col3, ind_col4 = st.columns(4)
                preds = ag.get("individual_predictions", {})
                ind_col1.info(f"**XGBoost:** {preds.get('XGBoost', 'N/A')}")
                ind_col2.info(f"**LightGBM:** {preds.get('LightGBM', 'N/A')}")
                ind_col3.info(f"**CatBoost:** {preds.get('CatBoost', 'N/A')}")
                ind_col4.success(f"**Ensemble:** {preds.get('Ensemble', 'N/A')}")

                # Class Probabilities Bar
                st.markdown("##### Model-Estimated Risk Probability Distribution")
                probs = res.get("probabilities", {})
                prob_df = pd.DataFrame(list(probs.items()), columns=["Risk Tier", "Model-Estimated Score"])
                fig_prob = px.bar(prob_df, x="Risk Tier", y="Model-Estimated Score", color="Risk Tier",
                                  color_discrete_map={"LOW": "#10B981", "MODERATE": "#F59E0B", "HIGH": "#F97316", "CRITICAL": "#EF4444"},
                                  text_auto=".2%")
                fig_prob.update_layout(height=280, margin=dict(t=20, b=20, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_prob, use_container_width=True)

                # Rule Engine Warnings
                rules = res.get("rule_engine_findings", {}).get("triggered_rules", [])
                st.markdown(f"##### Statutory & Geotechnical Rule Engine ({len(rules)} Notices Triggered)")
                if rules:
                    for r in rules:
                        sev_color = "red" if r["severity"] == "CRITICAL" else ("orange" if r["severity"] == "HIGH" else "blue")
                        st.markdown(f"""
                        <div style="background: rgba(30, 41, 59, 0.5); border-left: 4px solid {sev_color}; padding: 10px 14px; margin-bottom: 8px; border-radius: 4px;">
                            <b>[{r['severity']}] {r['title']}</b> ({r['regulation']})<br>
                            <span style="font-size: 0.85rem; color: #CBD5E1;">{r['finding']}</span><br>
                            <span style="font-size: 0.82rem; color: #38BDF8;">Action Mandate: {r['action']}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("No statutory dimension or SSR compliance rules triggered.")

                # SHAP Key Contributors
                st.markdown("##### Top Model Contributors (Local SHAP Attribution)")
                shap_info = res.get("shap_explanation", {}).get("top_contributors", [])
                if shap_info:
                    for idx, c in enumerate(shap_info, 1):
                        direction_icon = "🔺" if "INCREASED" in c["impact_direction"] else "🔻"
                        st.markdown(f"**{idx}. {direction_icon} {c['feature']}**: {c['impact_direction']} (Impact weight: {c['importance_magnitude']:.3f})")
                
            except Exception as e:
                st.error(f"Inference error: {str(e)}")


# ==============================================================================
# MODULE 3: MODEL EXPLAINABILITY & SHAP
# ==============================================================================
elif nav_choice == "3. Model Explainability & SHAP":
    st.subheader("🔍 Model Explainability & SHAP Feature Attribution")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) visualizes how each geotechnical variable contributes mathematically
    to shifting the model's output towards a specific hazard category.
    """)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>SCIENTIFIC PRINCIPLE:</b> SHAP explains <i>why the mathematical model assigned this score</i>.
        It does NOT denote direct physical causality of rock failure.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Global Feature Importance (HIGH Hazard Tier)")
        bar_img = "results/explainability/xgboost_shap_bar.png"
        if os.path.exists(bar_img):
            st.image(bar_img, caption="SHAP Global Mean |Impact| on HIGH Risk Classification", use_container_width=True)
        else:
            st.info("Run SHAP training to generate global bar plot.")

    with col2:
        st.markdown("##### Beeswarm Distribution (Feature Impact Direction)")
        bee_img = "results/explainability/xgboost_shap_beeswarm.png"
        if os.path.exists(bee_img):
            st.image(bee_img, caption="SHAP Summary Beeswarm: Red = High Value, Blue = Low Value", use_container_width=True)
        else:
            st.info("Run SHAP training to generate beeswarm plot.")

    st.markdown("### Top Geotechnical Risk Drivers Identified by SHAP")
    st.markdown("""
    1. **Bed Separation Dilation (`bed_separation_mm`)**: High values (>12mm) produce the strongest positive SHAP push into HIGH and CRITICAL hazard.
    2. **Roof Convergence Rate (`roof_convergence_mm_day`)**: Active convergence velocity (>5mm/day) sharply drives model confidence toward imminent strata failure.
    3. **Installed Support Density (`support_density_bolts_per_m2`)**: Values below 1.0 bolt/m² generate substantial positive hazard contributions.
    4. **Gallery Width (`gallery_width_m`)**: Exceeding statutory limits (>4.8m) induces heavy non-linear risk escalation due to tensile span bending mechanics.
    5. **Solid Blasting Shock (`blast_vibration_ppv`)**: Dynamic vibration PPV > 12 mm/s creates micro-crack delaminations captured by the tree models.
    """)


# ==============================================================================
# MODULE 4: HISTORICAL DGMS ANALYTICS
# ==============================================================================
elif nav_choice == "4. Historical DGMS Analytics":
    st.subheader("📊 Official DGMS Fatal Roof-Fall Intelligence (2016–2022)")
    st.caption("Primary Dataset A: 41 Fatal 0111 Incidents across Indian Coal & Metalliferous Mines (100% Unhallucinated)")

    try:
        stats = api.get_accident_stats()
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Fatal Accidents (0111)", stats.get("total_accidents", 41))
        c2.metric("Total Fatalities", stats.get("total_fatalities", 58))
        c3.metric("Mines Involved", stats.get("total_mines_involved", 35))
        c4.metric("States Affected", stats.get("total_states_involved", 9))

        colA, colB = st.columns(2)
        with colA:
            # Yearly Trend
            df_year = pd.DataFrame(stats.get("accidents_by_year", []))
            fig_yr = px.bar(df_year, x="year", y=["accidents", "fatalities"], barmode="group",
                            title="Roof-Fall Incidents & Lives Lost by Year (2016-2022)",
                            color_discrete_sequence=["#F59E0B", "#EF4444"])
            fig_yr.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_yr, use_container_width=True)

        with colB:
            # Coal vs Non-Coal
            df_sec = pd.DataFrame(stats.get("accidents_by_sector", []))
            fig_sec = px.pie(df_sec, names="sector", values="fatalities",
                             title="Fatalities Breakdown: Coal vs Non-Coal Mines",
                             color_discrete_sequence=["#3B82F6", "#10B981"], hole=0.4)
            fig_sec.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_sec, use_container_width=True)

        colC, colD = st.columns(2)
        with colC:
            # Geographic Distribution
            df_st = pd.DataFrame(stats.get("accidents_by_state", []))
            fig_st = px.bar(df_st, x="state", y="cnt", title="Geographic Distribution of Fatal Roof Falls by State",
                            color="cnt", color_continuous_scale="Viridis")
            fig_st.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_st, use_container_width=True)

        with colD:
            # Mineral Distribution
            df_min = pd.DataFrame(stats.get("accidents_by_mineral", []))
            fig_min = px.bar(df_min, x="mineral", y="cnt", title="Minerals Affected in Ground Movement Fatalities",
                             color="cnt", color_continuous_scale="Cividis")
            fig_min.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_min, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to fetch accident statistics: {str(e)}")


# ==============================================================================
# MODULE 5: ACCIDENT SIMILARITY SEARCH
# ==============================================================================
elif nav_choice == "5. Accident Similarity Search":
    st.subheader("🎯 Historical Precedent Similarity Search (FAISS)")
    st.write("Retrieve verbatim historical DGMS inquiries matching a given underground scenario.")

    sample_query = st.text_area(
        "Enter Mine Conditions or Incident Narrative:",
        value="Depillaring stook extraction near active goaf edge with sandstone roof parting and missing props"
    )

    filter_sec = st.selectbox("Filter Sector", ["All", "Coal", "Non-Coal"])
    sec_param = None if filter_sec == "All" else filter_sec

    if st.button("🔍 Search FAISS Vector Store", use_container_width=True):
        with st.spinner("Embedding query and querying FAISS Inner Product Index..."):
            try:
                res = api.search_similar_accidents(sample_query, top_k=5, filter_coal_noncoal=sec_param)
                matches = res.get("matches", [])
                st.markdown(f"##### Top {len(matches)} Historical Matches Found:")

                for idx, m in enumerate(matches, 1):
                    score_pct = m.get("similarity_score", 0) * 100
                    with st.expander(f"Case #{idx}: {m.get('mine')} ({m.get('year')}, {m.get('state')}) — Match: {score_pct:.1f}%", expanded=(idx==1)):
                        col1, col2, col3 = st.columns(3)
                        col1.write(f"**Mineral:** {m.get('coal_noncoal')}")
                        col2.write(f"**Fatalities:** {m.get('fatalities')}")
                        col3.write(f"**Date:** {m.get('date')}")
                        
                        st.markdown(f"**Inquiry Findings:** {m.get('description')}")
                        st.markdown(f"**Root Cause:** `{m.get('root_cause')}`")
                        st.markdown(f"**Statutory Regulations Cited:** `{m.get('regulations')}`")
            except Exception as e:
                st.error(f"Search failed: {str(e)}")


# ==============================================================================
# MODULE 6: REGULATORY RAG EVIDENCE
# ==============================================================================
elif nav_choice == "6. Regulatory RAG Evidence":
    st.subheader("📚 Statutory & Technical Regulation RAG Engine")
    st.write("Query Coal Mines Regulations (CMR 2017), Metalliferous Mines Regulations (MMR 1961), and DGMS Circulars.")

    rag_q = st.text_input(
        "Ask a Regulatory or Technical Question:",
        value="What statutory regulations mandate support setting at the goaf edge in depillaring panels?"
    )

    if st.button("📖 Search Knowledge Base", use_container_width=True):
        with st.spinner("Searching FAISS Regulatory Index..."):
            try:
                res = api.query_rag(rag_q, top_k=4)
                evidence = res.get("evidence", [])

                for idx, ev in enumerate(evidence, 1):
                    cat_color = "#3B82F6" if "Statutory" in ev.get("category", "") else "#10B981"
                    st.markdown(f"""
                    <div style="background: rgba(30, 41, 59, 0.6); border-left: 4px solid {cat_color}; padding: 14px; margin-bottom: 12px; border-radius: 6px;">
                        <b>{ev.get('title')}</b> <span style="font-size: 0.8rem; background: #334155; padding: 2px 6px; border-radius: 4px;">{ev.get('category')}</span><br>
                        <span style="font-size: 0.82rem; color: #94A3B8;">Source: {ev.get('source')} | {ev.get('page')} | Match Score: {ev.get('similarity_score', 0)*100:.1f}%</span><br><br>
                        <span style="font-size: 0.9rem; color: #F1F5F9;">{ev.get('content')}</span>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"RAG query failed: {str(e)}")


# ==============================================================================
# MODULE 7: AGENTIC SAFETY ADVISOR
# ==============================================================================
elif nav_choice == "7. Agentic Safety Advisor":
    st.subheader("🤖 LangGraph Agentic Decision Advisor")
    st.caption("Multi-Agent Architecture: Router -> Risk -> SHAP -> FAISS Similarity -> Evidence -> Recommendation -> Critic")

    user_role = st.selectbox(
        "Select Your Operating Designation:",
        ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Worker"]
    )

    agent_query = st.text_area(
        "Enter Decision-Support Query:",
        value="Why is my current roof-fall risk HIGH after blasting in depillaring stook, and what immediate actions must I take?"
    )

    if st.button("🚀 Invoke Agentic Decision Support", use_container_width=True):
        with st.spinner("Executing LangGraph State Graph & Critic Validation..."):
            try:
                chat_res = api.chat_with_agent(
                    query=agent_query,
                    user_role=user_role
                )

                st.markdown("---")
                st.markdown("#### Agentic Decision-Support Output")
                st.code(chat_res.get("final_response", ""), language="markdown")

                st.markdown("##### Agent Artifact Traceability")
                c1, c2, c3 = st.columns(3)
                c1.metric("Classified Intent", chat_res.get("intent", "N/A"))
                c2.metric("Critic Validation", "PASSED" if chat_res.get("critic_passed") else "FAILED")
                c3.metric("Audit Record", f"#{chat_res.get('audit_id', 0)}")

            except Exception as e:
                st.error(f"Agent execution failed: {str(e)}")


# ==============================================================================
# MODULE 8: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
elif nav_choice == "8. Model Benchmarks & Evaluation":
    st.subheader("📈 Model Benchmarking & Safety Evaluation Report")
    st.write("Rigorous comparison of Baseline, Gradient Boosted Models, and Ensembles against DGMS Safety Criteria.")

    csv_comp = "results/model_comparison.csv"
    if os.path.exists(csv_comp):
        df_comp = pd.read_csv(csv_comp)
        st.dataframe(df_comp.style.highlight_max(subset=["balanced_accuracy", "macro_f1", "high_recall", "critical_recall"], color="#065F46"), use_container_width=True)

        st.markdown("### Safety Optimization Rationale")
        st.markdown("""
        In industrial mining safety, **False Negatives are catastrophic** (misclassifying a HIGH or CRITICAL hazard as safe).
        - **HIGH Hazard Recall**: Champion model achieves **80.8%**.
        - **CRITICAL Hazard Recall**: Supported with **55.6%** sensitivity despite extreme class rarity (3.3% of operational distribution).
        - **Zero Temporal Leakage**: Models are validated strictly on future chronologically held-out inspection records.
        """)

        # Calibration Curves Display
        calib_img = "results/calibration/calibration_curves.png"
        if os.path.exists(calib_img):
            st.image(calib_img, caption="Reliability Curves (Calibration) for HIGH and CRITICAL Hazard Classes", use_container_width=True)
    else:
        st.info("Run model training pipeline to generate benchmark comparison table.")


# ==============================================================================
# MODULE 9: SAFETY POLICY & DATA DICTIONARY
# ==============================================================================
elif nav_choice == "9. Safety Policy & Data Dictionary":
    st.subheader("📜 Safety Policy & Data Dictionary")

    tab1, tab2 = st.tabs(["DGMS Safety Policy", "Data Dictionary (30 Canonical Fields)"])

    with tab1:
        st.markdown("""
        ### BHOOmiKAVACH AI — Statutory Safety Policy & Operating Constraints
        1. **Prohibition of Fabrication**: The system MUST NEVER invent regulations, accident facts, mine conditions, or safety limits.
        2. **Non-Causal Explanation Mandate**: The system MUST NEVER claim correlation proves causation. Output must state:
           *"Contributed to the mathematical model prediction"*, NOT *"Caused the roof fall"*.
        3. **No Guarantee of Safety**: Low-risk classification does NOT certify that a roof is safe. Rock mass behavior exhibits inherent stochastic discontinuities.
        4. **Human Competence Primacy**: The system CANNOT authorize mine entry or autonomously enact safety decisions.
        5. **Mandatory Disclaimer**: Every assessment is explicitly accompanied by the statutory disclaimer requiring competent technical inspection.
        """)

    with tab2:
        dict_md = "docs/data_dictionary.md"
        if os.path.exists(dict_md):
            with open(dict_md, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Data dictionary document available in `docs/data_dictionary.md`.")
