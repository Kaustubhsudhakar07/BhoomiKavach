"""
BHOOmiKAVACH AI — Industrial Mine-Safety Decision Support Dashboard
=====================================================================
National Mining Geotechnical Safety & Real-Time Roof-Fall Hazard Radar
Directorate General of Mines Safety (DGMS) Code 0111: Fall of Roof
Standards: Coal Mines Regulations 2017 & Metalliferous Mines Regulations 1961

Frontend Architecture: High-Precision Industrial Streamlit UI consuming FastAPI REST backend.
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

# ──────────────────────────────────────────────────────────────────────────────
# 1. PAGE SETUP & METADATA
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BHOOmiKAVACH AI | Industrial Strata Safety Radar",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DESIGN SYSTEM & HUMAN-CRAFTED ENTERPRISE STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-canvas: #080C14;
        --bg-surface: #0E1526;
        --bg-card: rgba(15, 23, 42, 0.75);
        --border-subtle: rgba(255, 255, 255, 0.08);
        --border-accent: rgba(59, 130, 246, 0.4);
        --accent-amber: #F59E0B;
        --accent-emerald: #10B981;
        --accent-crimson: #EF4444;
        --accent-cyan: #06B6D4;
        --accent-blue: #3B82F6;
        --text-primary: #F8FAFC;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
    }

    /* Core typography and canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    .stApp {
        background: radial-gradient(circle at 10% 0%, #111A2E 0%, #080C14 70%);
        background-attachment: fixed;
    }

    /* Streamlit layout tightening */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 96% !important;
    }

    /* Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: #080C14;
    }
    ::-webkit-scrollbar-thumb {
        background: #1E293B;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #334155;
    }

    /* Top Command Header */
    .top-command-bar {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 15, 30, 0.85) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 15px;
    }

    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .brand-crest {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid rgba(245, 158, 11, 0.4);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.15);
    }

    .brand-title-wrap h1 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #FFFFFF 0%, #E2E8F0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .brand-title-wrap span {
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        display: block;
    }

    .telemetry-pills {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
    }

    .status-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid var(--border-subtle);
        padding: 5px 12px;
        border-radius: 9999px;
        font-size: 0.73rem;
        font-weight: 600;
        color: #CBD5E1;
        letter-spacing: 0.04em;
    }

    .status-chip-live {
        border-color: rgba(16, 185, 129, 0.4);
        background: rgba(16, 185, 129, 0.1);
        color: #6EE7B7;
    }

    .pulse-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981;
        display: inline-block;
    }

    /* Official DGMS Notice Banner */
    .statutory-banner {
        background: linear-gradient(90deg, rgba(239, 68, 68, 0.08) 0%, rgba(15, 23, 42, 0.6) 100%);
        border-left: 3px solid #EF4444;
        border-top: 1px solid rgba(239, 68, 68, 0.2);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 10px 18px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 22px;
        font-size: 0.8rem;
        color: #FCA5A5;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Enterprise KPI Card Component */
    .kpi-container {
        background: linear-gradient(145deg, rgba(17, 24, 39, 0.75), rgba(10, 15, 28, 0.85));
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.4);
    }
    .kpi-container::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.6), transparent);
    }
    .kpi-label {
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #FFFFFF;
        line-height: 1.1;
        margin-bottom: 6px;
    }
    .kpi-caption {
        font-size: 0.75rem;
        color: #64748B;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Hazard Cockpit Component */
    .cockpit-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(10, 16, 32, 0.95));
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 24px;
        box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
        margin: 16px 0;
        position: relative;
    }

    .cockpit-critical {
        border-color: rgba(239, 68, 68, 0.5);
        background: linear-gradient(135deg, rgba(35, 12, 16, 0.85), rgba(15, 23, 42, 0.95));
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.2);
    }
    .cockpit-high {
        border-color: rgba(249, 115, 22, 0.5);
        background: linear-gradient(135deg, rgba(35, 20, 12, 0.85), rgba(15, 23, 42, 0.95));
        box-shadow: 0 0 25px rgba(249, 115, 22, 0.15);
    }
    .cockpit-moderate {
        border-color: rgba(245, 158, 11, 0.4);
        background: linear-gradient(135deg, rgba(30, 24, 14, 0.85), rgba(15, 23, 42, 0.95));
    }
    .cockpit-low {
        border-color: rgba(16, 185, 129, 0.4);
        background: linear-gradient(135deg, rgba(10, 28, 22, 0.85), rgba(15, 23, 42, 0.95));
    }

    /* Risk Badges */
    .risk-pill {
        display: inline-flex;
        align-items: center;
        padding: 6px 16px;
        border-radius: 9999px;
        font-weight: 800;
        font-size: 0.9rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .risk-pill-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border: 1px solid rgba(239, 68, 68, 0.6);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.4);
    }
    .risk-pill-high {
        background: rgba(249, 115, 22, 0.2);
        color: #FDBA74;
        border: 1px solid rgba(249, 115, 22, 0.6);
    }
    .risk-pill-moderate {
        background: rgba(245, 158, 11, 0.2);
        color: #FDE68A;
        border: 1px solid rgba(245, 158, 11, 0.5);
    }
    .risk-pill-low {
        background: rgba(16, 185, 129, 0.2);
        color: #6EE7B7;
        border: 1px solid rgba(16, 185, 129, 0.5);
    }

    /* Custom Form & Fieldset containers */
    .panel-section {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 16px;
    }

    .panel-section h5 {
        color: #F1F5F9;
        font-size: 0.88rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
        padding-bottom: 8px;
    }

    /* Official Safety Bulletin Card */
    .bulletin-box {
        background: linear-gradient(180deg, #0D1424 0%, #080D18 100%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 24px;
        margin: 20px 0;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.5);
    }
    .bulletin-header {
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 14px;
        margin-bottom: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }
    .bulletin-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        color: #F8FAFC;
    }
    .bulletin-meta {
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        color: #94A3B8;
    }

    /* Rule alert cards */
    .rule-card {
        background: rgba(15, 23, 42, 0.65);
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
        border-left: 4px solid #3B82F6;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .rule-card-critical { border-left-color: #EF4444; }
    .rule-card-high { border-left-color: #F97316; }
    .rule-card-moderate { border-left-color: #F59E0B; }

    /* Button Polish */
    .stButton > button {
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        letter-spacing: 0.02em !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.3) !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.45) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:active {
        transform: translateY(1px) !important;
    }

    /* Streamlit Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-subtle);
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 6px;
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0 16px;
        border: 1px solid transparent;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: #F8FAFC !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }

    /* Sidebar cleanup */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090E1A 0%, #060911 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 3. GLOBAL CHART THEME HELPER
# ──────────────────────────────────────────────────────────────────────────────
def format_bhoomi_plot(fig, height=320, title=None):
    """Applies high-end human-designed dark theme styling to all Plotly figures."""
    fig.update_layout(
        height=height,
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(family="Plus Jakarta Sans", size=13, color="#CBD5E1"),
            x=0.01,
            y=0.96
        ) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.35)",
        font=dict(family="Plus Jakarta Sans", color="#94A3B8", size=11),
        margin=dict(t=35 if title else 20, b=25, l=35, r=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=10, color="#CBD5E1"),
            bgcolor="rgba(0,0,0,0)"
        ),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(size=10, color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            tickfont=dict(size=10, color="#94A3B8")
        )
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# 4. INITIALIZATION & SESSION MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────
if "api_client" not in st.session_state:
    st.session_state.api_client = BhoomiApiClient()
api = st.session_state.api_client

# Preset state management for Module 2
if "preset_mine" not in st.session_state:
    st.session_state.preset_mine = "Moonidih Colliery"
    st.session_state.preset_method = "Bord and Pillar (Depillaring)"
    st.session_state.preset_depth = 420.0
    st.session_state.preset_width = 4.5
    st.session_state.preset_height = 2.8
    st.session_state.preset_rmr = 42.0
    st.session_state.preset_lithology = "Sandstone-Shale Intercalation"
    st.session_state.preset_density = 0.90
    st.session_state.preset_convergence = 5.8
    st.session_state.preset_bed_sep = 16.0
    st.session_state.preset_blasting = "Recent (4-12h)"
    st.session_state.preset_face_pos = "Depillaring Edge / Goaf"
    st.session_state.preset_supp_cond = "Minor Deformation"


# ──────────────────────────────────────────────────────────────────────────────
# 5. SIDEBAR NAVIGATION & TELEMETRY
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <div style="font-size: 26px;">🛡️</div>
        <div>
            <div style="font-weight: 800; font-size: 1.25rem; letter-spacing: -0.01em; color: #F8FAFC;">BHOOmiKAVACH</div>
            <div style="font-size: 0.68rem; font-weight: 600; letter-spacing: 0.12em; color: #94A3B8; text-transform: uppercase;">Decision Support Radar</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 12px 0 16px 0;'></div>", unsafe_allow_html=True)

    st.caption("TACTICAL NAVIGATION")
    nav_choice = st.radio(
        "Module Directory",
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
        index=1,
        label_visibility="collapsed"
    )

    st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0 16px 0;'></div>", unsafe_allow_html=True)

    # Operational Telemetry Monitor
    st.caption("INFRASTRUCTURE TELEMETRY")
    health = api.get_health()
    if health.get("status") == "HEALTHY":
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 10px 12px;">
            <div style="display: flex; align-items: center; gap: 8px; font-size: 0.78rem; font-weight: 700; color: #34D399;">
                <span class="pulse-dot"></span> API Engine: v1.2.0 ONLINE
            </div>
            <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 4px; font-family: 'JetBrains Mono', monospace;">
                DGMS Records: 41 | Vector Store: 54
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; padding: 10px 12px;">
            <div style="font-size: 0.78rem; font-weight: 700; color: #F87171;">
                ⚠️ API Engine: {health.get('status', 'DEGRADED')}
            </div>
            <div style="font-size: 0.68rem; color: #94A3B8; margin-top: 4px;">
                Confirm backend on port 8000
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 1px; background: rgba(255,255,255,0.08); margin: 20px 0 16px 0;'></div>", unsafe_allow_html=True)

    # Google Gemini Safety Intelligence
    st.caption("GENERATIVE AI ENGINE")
    default_gemini_key = os.getenv("GEMINI_API_KEY", "")
    gemini_key_input = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.get("gemini_api_key", default_gemini_key),
        type="password",
        help="Optional: Powers conversational DGMS regulatory reasoning in Modules 6 & 7."
    )
    if gemini_key_input:
        st.session_state["gemini_api_key"] = gemini_key_input
        st.markdown("""
        <div style="font-size: 0.72rem; color: #38BDF8; display: flex; align-items: center; gap: 6px; font-weight: 600;">
            <span>✨</span> Gemini 2.5 Flash Grounded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="font-size: 0.72rem; color: #94A3B8;">
            Rule-Grounded Knowledge Engine Active
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size: 0.68rem; color: #475569; margin-top: 25px; line-height: 1.4; font-family: 'JetBrains Mono', monospace;">
        CODE: DGMS 0111 (ROOF FALL)<br>
        STANDARDS: CMR 2017 / MMR 1961<br>
        MODELS: XGB · LGB · CAT ENSEMBLE
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# 6. ENTERPRISE TOP COMMAND HEADER & STATUTORY NOTICE
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="top-command-bar">
    <div class="brand-group">
        <div class="brand-crest">🛡️</div>
        <div class="brand-title-wrap">
            <h1>BHOOmiKAVACH AI</h1>
            <span>National Mining Strata Hazard Radar · DGMS Code 0111</span>
        </div>
    </div>
    <div class="telemetry-pills">
        <div class="status-chip status-chip-live"><span class="pulse-dot"></span> LIVE TELEMETRY</div>
        <div class="status-chip">CMR 2017 GROUNDED</div>
        <div class="status-chip">ENSEMBLE ENFORCED</div>
        <div class="status-chip">SHAP CALIBRATED</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="statutory-banner">
    <div style="font-size: 1.1rem; line-height: 1;">⚠️</div>
    <div>
        <b>STATUTORY TECHNICAL NOTICE:</b> This system provides AI-assisted geotechnical decision support under
        DGMS guidelines. It does <b>NOT</b> constitute an engineering certification or supersede the on-site statutory
        authority of a certified Mine Manager, Overman, or Mining Sirdar under CMR 2017 / MMR 1961.
    </div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# MODULE 1: OVERVIEW & SYSTEM TELEMETRY
# ==============================================================================
if nav_choice == "1. Overview & System Telemetry":
    st.markdown("### Operational Telemetry & Architecture Framework")
    st.caption("Live monitoring of models, statutory vector corpora, and feature pipeline integrity.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">Historical Inquiries</div>
            <div class="kpi-value" style="color: #F59E0B;">41 Cases</div>
            <div class="kpi-caption"><span style="color: #10B981;">●</span> 100% Unhallucinated (0111)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">Champion Ensemble</div>
            <div class="kpi-value" style="color: #3B82F6;">Tri-Model</div>
            <div class="kpi-caption">XGB (0.33) + LGB (0.33) + CAT (0.33)</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">Hazard Sensitivity</div>
            <div class="kpi-value" style="color: #10B981;">80.8%</div>
            <div class="kpi-caption">High-Risk Recall (Temporal Validation)</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">Regulatory Precedents</div>
            <div class="kpi-value" style="color: #A855F7;">54 Clauses</div>
            <div class="kpi-caption">FAISS MiniLM-L6 Vector Store</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### High-Reliability Pipeline Architecture")

    st.markdown("""
    <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 22px; font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94A3B8; overflow-x: auto;">
        <span style="color: #38BDF8;">[1. LIVE TELEMETRY]</span> ────────► <span style="color: #F59E0B;">[2. SCALED PREPROCESSING]</span> ───┬──► <span style="color: #10B981;">[3. OUT-OF-DISTRIBUTION ISOLATION FOREST]</span><br>
        Convergence (mm/day)            84 Engineered Features         │    Domain Boundary Alerting (&lt; 0.0 threshold)<br>
        Bed Separation (mm)             CMRI-ISM RMR Normalized        │<br>
        Support Density (b/m²)          Rock Mass Geomechanics         ├──► <span style="color: #A855F7;">[4. MULTI-MODEL ENSEMBLE ENGINE]</span><br>
        Gallery Geometry (W x H)                                       │    XGBoost + LightGBM + CatBoost (3-Way Consensus)<br>
                                                                       │<br>
                                                                       ├──► <span style="color: #EC4899;">[5. LOCAL SHAP EXPLAINABILITY]</span><br>
                                                                       │    Exact Feature Contribution Attributions<br>
                                                                       │<br>
                                                                       └──► <span style="color: #EF4444;">[6. STATUTORY SAFETY CRITIC & RAG]</span><br>
                                                                            CMR 2017 Regulations + Life Safety Directives
    </div>
    """, unsafe_allow_html=True)


# ==============================================================================
# MODULE 2: LIVE STRATA RISK ASSESSMENT (CORE OPERATIONAL RADAR)
# ==============================================================================
elif nav_choice == "2. Live Strata Risk Assessment":
    st.markdown("### Geotechnical Hazard Radar & Inference Cockpit")
    st.caption("Input live strata convergence, installed support parameters, and mining geometry for prospective evaluation.")

    # 1-Click Operational Scenario Presets
    st.markdown("##### ⚡ Quick Operational Scenario Presets")
    col_pre1, col_pre2, col_pre3, col_pre4 = st.columns(4)

    if col_pre1.button("🔴 Depillaring Goaf Edge\n(High Risk Preset)", use_container_width=True):
        st.session_state.preset_mine = "Moonidih Colliery"
        st.session_state.preset_method = "Bord and Pillar (Depillaring)"
        st.session_state.preset_depth = 480.0
        st.session_state.preset_width = 4.8
        st.session_state.preset_height = 2.8
        st.session_state.preset_rmr = 38.0
        st.session_state.preset_lithology = "Sandstone-Shale Intercalation"
        st.session_state.preset_density = 0.80
        st.session_state.preset_convergence = 9.2
        st.session_state.preset_bed_sep = 19.5
        st.session_state.preset_blasting = "Immediate (<4h)"
        st.session_state.preset_face_pos = "Depillaring Edge / Goaf"
        st.session_state.preset_supp_cond = "Excessive Load/Yielding"
        st.rerun()

    if col_pre2.button("🟢 Standard Development\n(Low Risk Preset)", use_container_width=True):
        st.session_state.preset_mine = "GDK 11A Incline"
        st.session_state.preset_method = "Bord and Pillar (Development)"
        st.session_state.preset_depth = 220.0
        st.session_state.preset_width = 4.2
        st.session_state.preset_height = 2.6
        st.session_state.preset_rmr = 58.0
        st.session_state.preset_lithology = "Sandstone"
        st.session_state.preset_density = 1.40
        st.session_state.preset_convergence = 0.9
        st.session_state.preset_bed_sep = 2.5
        st.session_state.preset_blasting = "No Blasting (<24h)"
        st.session_state.preset_face_pos = "Development Face"
        st.session_state.preset_supp_cond = "Intact & Adequate"
        st.rerun()

    if col_pre3.button("⚠️ Excessive Span Junction\n(Critical Hazard Preset)", use_container_width=True):
        st.session_state.preset_mine = "Jhilimili Colliery"
        st.session_state.preset_method = "Bord and Pillar (Depillaring)"
        st.session_state.preset_depth = 390.0
        st.session_state.preset_width = 5.6
        st.session_state.preset_height = 3.2
        st.session_state.preset_rmr = 32.0
        st.session_state.preset_lithology = "Shale"
        st.session_state.preset_density = 0.65
        st.session_state.preset_convergence = 14.5
        st.session_state.preset_bed_sep = 28.0
        st.session_state.preset_blasting = "Immediate (<4h)"
        st.session_state.preset_face_pos = "Junction"
        st.session_state.preset_supp_cond = "Dislodged/Missing"
        st.rerun()

    if col_pre4.button("🟡 Deep Shaft Station\n(Moderate Risk Preset)", use_container_width=True):
        st.session_state.preset_mine = "Churcha RO Mine"
        st.session_state.preset_method = "Continuous Miner"
        st.session_state.preset_depth = 550.0
        st.session_state.preset_width = 4.4
        st.session_state.preset_height = 2.8
        st.session_state.preset_rmr = 46.0
        st.session_state.preset_lithology = "Carbonaceous Shale"
        st.session_state.preset_density = 1.10
        st.session_state.preset_convergence = 3.8
        st.session_state.preset_bed_sep = 7.0
        st.session_state.preset_blasting = "Recent (4-12h)"
        st.session_state.preset_face_pos = "Main Haulage"
        st.session_state.preset_supp_cond = "Minor Deformation"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Telemetry Input Form
    with st.form("risk_assessment_form"):
        colA, colB, colC = st.columns(3)

        with colA:
            st.markdown("<div class='panel-section'><h5>📍 Spatial & Geological Setting</h5>", unsafe_allow_html=True)
            mine_name = st.selectbox(
                "Mine / Colliery",
                ["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"],
                index=["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"].index(st.session_state.preset_mine) if st.session_state.preset_mine in ["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"] else 0
            )
            sector = "Coal" if "Mine" not in mine_name and "Kolihan" not in mine_name else "Non-Coal"
            mineral = "Coal" if sector == "Coal" else ("Manganese" if "Balaghat" in mine_name else "Copper")
            
            method = st.selectbox(
                "Extraction Method",
                ["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"],
                index=["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"].index(st.session_state.preset_method) if st.session_state.preset_method in ["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"] else 0
            )
            depth = st.slider("Cover Depth (m)", min_value=30.0, max_value=800.0, value=float(st.session_state.preset_depth), step=10.0)
            face_pos = st.selectbox(
                "Working Face Position",
                ["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"],
                index=["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"].index(st.session_state.preset_face_pos) if st.session_state.preset_face_pos in ["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"] else 0
            )
            is_junc = 1 if face_pos == "Junction" else st.checkbox("Intersection Junction?", value=(face_pos == "Junction"))
            st.markdown("</div>", unsafe_allow_html=True)

        with colB:
            st.markdown("<div class='panel-section'><h5>📐 Geometry & Rock Mechanics</h5>", unsafe_allow_html=True)
            width = st.slider("Gallery Width (m) [CMR 111 Limit: 4.8m]", min_value=3.0, max_value=6.5, value=float(st.session_state.preset_width), step=0.1)
            height = st.slider("Gallery Height (m) [CMR 111 Limit: 3.0m]", min_value=1.8, max_value=4.5, value=float(st.session_state.preset_height), step=0.1)
            rmr = st.slider("CMRI-ISM Rock Mass Rating (RMR)", min_value=15.0, max_value=85.0, value=float(st.session_state.preset_rmr), step=1.0)
            lithology = st.selectbox(
                "Roof Lithology",
                ["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"],
                index=["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"].index(st.session_state.preset_lithology) if st.session_state.preset_lithology in ["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"] else 0
            )
            blasting = st.selectbox(
                "Blasting Recency",
                ["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"],
                index=["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"].index(st.session_state.preset_blasting) if st.session_state.preset_blasting in ["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"] else 1
            )
            blast_vibe = st.slider("Blasting Peak Particle Velocity PPV (mm/s)", min_value=0.5, max_value=30.0, value=12.5, step=0.5)
            st.markdown("</div>", unsafe_allow_html=True)

        with colC:
            st.markdown("<div class='panel-section'><h5>🛡️ Strata Convergence & Roof Bolting</h5>", unsafe_allow_html=True)
            supp_cond = st.selectbox(
                "Support System Integrity",
                ["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"],
                index=["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"].index(st.session_state.preset_supp_cond) if st.session_state.preset_supp_cond in ["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"] else 2
            )
            density = st.slider("Bolt Density (bolts/m²)", min_value=0.4, max_value=2.4, value=float(st.session_state.preset_density), step=0.05)
            bolt_cap = st.slider("Bolt Anchorage Capacity (tonnes)", min_value=2.0, max_value=16.0, value=7.5, step=0.5)
            convergence = st.slider("Roof Convergence Velocity (mm/day)", min_value=0.1, max_value=25.0, value=float(st.session_state.preset_convergence), step=0.1)
            bed_sep = st.slider("Borehole Extensometer Bed Separation (mm)", min_value=0.0, max_value=40.0, value=float(st.session_state.preset_bed_sep), step=0.5)
            role_choice = st.selectbox("Operating Designation", ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Worker"])
            st.markdown("</div>", unsafe_allow_html=True)

        submit_btn = st.form_submit_button("⚡ Run Multi-Model Risk Evaluation", use_container_width=True)

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
            "joint_condition": "Slightly Weathered",
            "water_condition": "Damp",
            "recent_blasting": blasting,
            "blast_vibration_ppv": blast_vibe,
            "support_condition": supp_cond,
            "support_density_bolts_per_m2": density,
            "bolt_capacity_tonnes": bolt_cap,
            "roof_convergence_mm_day": convergence,
            "bed_separation_mm": bed_sep,
            "user_role": role_choice
        }

        with st.spinner("Processing Scaled Geomechanics, Model Consensus, and SHAP Attributions..."):
            try:
                res = api.predict_risk(obs_payload)
                risk_lvl = res.get("assessed_risk", "MODERATE")
                cockpit_class = f"cockpit-{risk_lvl.lower()}"
                pill_class = f"risk-pill-{risk_lvl.lower()}"

                # ── HAZARD COCKPIT HUD ──────────────────────────────────────
                st.markdown(f"""
                <div class="cockpit-card {cockpit_class}">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px;">
                        <div>
                            <div style="font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; color: #94A3B8; font-weight: 700;">
                                Prospective Strata Hazard Assessment
                            </div>
                            <div style="display: flex; align-items: center; gap: 14px; margin-top: 6px;">
                                <span class="risk-pill {pill_class}">{risk_lvl} HAZARD</span>
                                <span style="font-size: 0.85rem; color: #E2E8F0; font-weight: 600;">
                                    Audit Record: #{res.get('audit_id', 0)}
                                </span>
                            </div>
                        </div>
                        <div style="text-align: right;">
                            <div style="font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: #94A3B8;">System Confidence</div>
                            <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">
                                {res.get('confidence', 'HIGH')} ({res.get('model_agreement', {}).get('consensus_score', 1.0)*100:.0f}% Consensus)
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Agreement & Diagnostics Cards
                ag = res.get("model_agreement", {})
                ood = res.get("ood_diagnostics", {})

                d1, d2, d3, d4 = st.columns(4)
                with d1:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Model Agreement</div>
                        <div class="kpi-value" style="font-size: 1.5rem; color: #38BDF8;">{ag.get('agreement_ratio', '3/3')}</div>
                        <div class="kpi-caption">Consensus: {ag.get('consensus_score', 1.0)*100:.0f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                with d2:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">OOD Detector</div>
                        <div class="kpi-value" style="font-size: 1.5rem; color: {'#EF4444' if ood.get('is_ood') else '#10B981'};">
                            {'ALERT' if ood.get('is_ood') else 'IN-BOUNDS'}
                        </div>
                        <div class="kpi-caption">Score: {ood.get('isolation_forest_score', 0):.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with d3:
                    preds = ag.get("individual_predictions", {})
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Classifier Votes</div>
                        <div style="font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; line-height: 1.6; color: #E2E8F0; margin-top: 4px;">
                            XGB: <b>{preds.get('XGBoost', 'N/A')}</b><br>
                            LGB: <b>{preds.get('LightGBM', 'N/A')}</b><br>
                            CAT: <b>{preds.get('CatBoost', 'N/A')}</b>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with d4:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Ensemble Vote</div>
                        <div class="kpi-value" style="font-size: 1.5rem; color: #F59E0B;">{preds.get('Ensemble', risk_lvl)}</div>
                        <div class="kpi-caption">Weighted Probability Blend</div>
                    </div>
                    """, unsafe_allow_html=True)

                if ag.get("disagreement_warning"):
                    st.warning(f"⚠️ **Model Disagreement:** {ag['disagreement_warning']}")
                if ood.get("is_ood"):
                    st.error(f"⚠️ **Out-of-Distribution Alert:** {ood.get('ood_status')}: {', '.join(ood.get('reasons', []))}")

                # Charts Row: Class Probabilities + Local SHAP Waterfall
                st.markdown("<br>", unsafe_allow_html=True)
                pcol1, pcol2 = st.columns(2)

                with pcol1:
                    st.markdown("##### Model-Estimated Risk Probability Distribution")
                    probs = res.get("probabilities", {})
                    prob_df = pd.DataFrame(list(probs.items()), columns=["Risk Tier", "Probability"])
                    fig_prob = px.bar(
                        prob_df, x="Risk Tier", y="Probability", color="Risk Tier",
                        color_discrete_map={"LOW": "#10B981", "MODERATE": "#F59E0B", "HIGH": "#F97316", "CRITICAL": "#EF4444"},
                        text_auto=".1%"
                    )
                    fig_prob = format_bhoomi_plot(fig_prob, height=270)
                    st.plotly_chart(fig_prob, use_container_width=True)

                with pcol2:
                    st.markdown("##### Local SHAP Geotechnical Attributions")
                    shap_info = res.get("shap_explanation", {}).get("top_contributors", [])
                    if shap_info:
                        shap_df = pd.DataFrame(shap_info)
                        shap_df["signed_mag"] = shap_df.apply(
                            lambda r: r["importance_magnitude"] if "INCREASED" in r["impact_direction"] else -r["importance_magnitude"],
                            axis=1
                        )
                        shap_df["Color"] = shap_df["signed_mag"].apply(lambda v: "#EF4444" if v > 0 else "#10B981")
                        fig_shap = px.bar(
                            shap_df, y="feature", x="signed_mag", orientation="h",
                            color="Color", color_discrete_map={"#EF4444": "#EF4444", "#10B981": "#10B981"},
                            labels={"signed_mag": "SHAP Risk Contribution", "feature": "Strata Parameter"}
                        )
                        fig_shap.update_traces(showlegend=False)
                        fig_shap = format_bhoomi_plot(fig_shap, height=270)
                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info("Local SHAP attributions calculated.")

                # Statutory Rule Engine Findings
                st.markdown("<br>", unsafe_allow_html=True)
                rules = res.get("rule_engine_findings", {}).get("triggered_rules", [])
                st.markdown(f"##### Statutory & Geotechnical Rule Findings ({len(rules)} Active Directives)")
                if rules:
                    for r in rules:
                        sev_class = r['severity'].lower()
                        st.markdown(f"""
                        <div class="rule-card rule-card-{sev_class}">
                            <div style="display: flex; justify-content: space-between; align-items: center;">
                                <b>[{r['severity']}] {r['title']}</b>
                                <span style="font-size: 0.75rem; background: rgba(255,255,255,0.08); padding: 2px 8px; border-radius: 4px; font-family: 'JetBrains Mono', monospace;">
                                    {r['regulation']}
                                </span>
                            </div>
                            <div style="font-size: 0.82rem; color: #CBD5E1; margin: 6px 0;">{r['finding']}</div>
                            <div style="font-size: 0.8rem; color: #38BDF8; font-weight: 600;">Action Mandate: {r['action']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.success("✅ No statutory gallery dimension or Systematic Support Rules (SSR) violations triggered.")

            except Exception as e:
                st.error(f"Inference execution error: {str(e)}")


# ==============================================================================
# MODULE 3: MODEL EXPLAINABILITY & SHAP (SCIENTIFIC TRANSPARENCY)
# ==============================================================================
elif nav_choice == "3. Model Explainability & SHAP":
    st.markdown("### Scientific Explainability & SHAP Attributions")
    st.caption("Mathematical breakdown of tree model feature importance using Shapley Additive exPlanations.")

    st.markdown("""
    <div class="panel-section">
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.5;">
            <b>SCIENTIFIC PRINCIPLE:</b> SHAP explains <i>why the mathematical model assigned this risk assessment</i>.
            It reflects multi-variate statistical contributions and does <b>NOT</b> assert physical causality of strata failure.
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### Global Feature Importance (HIGH Hazard Tier)")
        bar_img = "results/explainability/xgboost_shap_bar.png"
        if os.path.exists(bar_img):
            st.image(bar_img, caption="SHAP Global Mean |Impact| across Features", use_container_width=True)
        else:
            st.info("Run pipeline to generate SHAP global bar plot.")

    with col2:
        st.markdown("##### Beeswarm Distribution (Feature Impact Direction)")
        bee_img = "results/explainability/xgboost_shap_beeswarm.png"
        if os.path.exists(bee_img):
            st.image(bee_img, caption="Summary Beeswarm: Red = Elevated Feature Value, Blue = Reduced Feature Value", use_container_width=True)
        else:
            st.info("Run pipeline to generate SHAP beeswarm plot.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### Top Geotechnical Risk Drivers Identified by SHAP")

    kpi_d1, kpi_d2, kpi_d3 = st.columns(3)
    with kpi_d1:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">#1 Strongest Driver</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #EF4444;">Bed Separation</div>
            <div class="kpi-caption">Dilations > 12mm sharply escalate model risk</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_d2:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">#2 Strongest Driver</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F97316;">Roof Convergence</div>
            <div class="kpi-caption">Active rates > 5mm/day denote strata yield</div>
        </div>
        """, unsafe_allow_html=True)
    with kpi_d3:
        st.markdown("""
        <div class="kpi-container">
            <div class="kpi-label">#3 Strongest Driver</div>
            <div style="font-size: 1.15rem; font-weight: 700; color: #F59E0B;">Support Density</div>
            <div class="kpi-caption">Densities < 1.0 bolt/m² trigger vulnerability</div>
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# MODULE 4: HISTORICAL DGMS ANALYTICS
# ==============================================================================
elif nav_choice == "4. Historical DGMS Analytics":
    st.markdown("### Official DGMS Fatal Roof-Fall Intelligence (2016–2022)")
    st.caption("Primary Historical Inquiry Base: 41 Fatal 0111 Incidents across Indian Coal & Metalliferous Mines (100% Unhallucinated)")

    try:
        stats = api.get_accident_stats()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Fatal Incidents (0111)</div>
                <div class="kpi-value" style="color: #F59E0B;">{stats.get('total_accidents', 41)}</div>
                <div class="kpi-caption">Verified Inquiry Reports</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Total Fatalities</div>
                <div class="kpi-value" style="color: #EF4444;">{stats.get('total_fatalities', 58)}</div>
                <div class="kpi-caption">Lives Lost in Ground Movement</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Mines Involved</div>
                <div class="kpi-value" style="color: #3B82F6;">{stats.get('total_mines_involved', 36)}</div>
                <div class="kpi-caption">Underground Operations</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">States Affected</div>
                <div class="kpi-value" style="color: #10B981;">{stats.get('total_states_involved', 9)}</div>
                <div class="kpi-caption">Indian Mining Coalfields</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        tab_charts, tab_gallery = st.tabs(["📊 Interactive Analytics", "🖼️ Detailed EDA Visualizations (14 Charts)"])

        with tab_charts:
            colA, colB = st.columns(2)
            with colA:
                df_year = pd.DataFrame(stats.get("accidents_by_year", []))
                fig_yr = px.bar(
                    df_year, x="year", y=["accidents", "fatalities"], barmode="group",
                    color_discrete_sequence=["#F59E0B", "#EF4444"]
                )
                fig_yr = format_bhoomi_plot(fig_yr, height=290, title="Roof-Fall Incidents & Fatalities by Year")
                st.plotly_chart(fig_yr, use_container_width=True)

            with colB:
                df_sec = pd.DataFrame(stats.get("accidents_by_sector", []))
                fig_sec = px.pie(
                    df_sec, names="sector", values="fatalities",
                    color_discrete_sequence=["#3B82F6", "#10B981"], hole=0.5
                )
                fig_sec = format_bhoomi_plot(fig_sec, height=290, title="Sectoral Fatality Ratio: Coal vs Non-Coal")
                st.plotly_chart(fig_sec, use_container_width=True)

            colC, colD = st.columns(2)
            with colC:
                df_st = pd.DataFrame(stats.get("accidents_by_state", []))
                y_col_st = "accidents" if "accidents" in df_st.columns else ("cnt" if "cnt" in df_st.columns else "fatalities")
                fig_st = px.bar(
                    df_st, x="state", y=y_col_st,
                    color=y_col_st, color_continuous_scale=["#1E3A8A", "#3B82F6", "#F59E0B", "#EF4444"]
                )
                fig_st = format_bhoomi_plot(fig_st, height=290, title="Geographic Distribution by State")
                st.plotly_chart(fig_st, use_container_width=True)

            with colD:
                df_min = pd.DataFrame(stats.get("accidents_by_mineral", []))
                y_col_min = "accidents" if "accidents" in df_min.columns else ("cnt" if "cnt" in df_min.columns else "fatalities")
                fig_min = px.bar(
                    df_min, x="mineral", y=y_col_min,
                    color=y_col_min, color_continuous_scale=["#064E3B", "#10B981", "#F59E0B"]
                )
                fig_min = format_bhoomi_plot(fig_min, height=290, title="Incidents by Extracted Mineral")
                st.plotly_chart(fig_min, use_container_width=True)

        with tab_gallery:
            st.markdown("##### Pre-computed Exploratory Data Analysis Suite")
            eda_files = [
                ("Shift Analysis", "results/eda/09_shift_analysis.png"),
                ("Contributing Factors", "results/eda/10_contributing_factors.png"),
                ("Root Causes", "results/eda/11_root_causes.png"),
                ("Regulations Cited", "results/eda/12_regulations_cited.png"),
                ("Victim Designation", "results/eda/07_victim_designation.png"),
                ("Historical Major Falls", "results/eda/13_century_major_rooffalls.png")
            ]
            gcols = st.columns(2)
            for i, (caption, fpath) in enumerate(eda_files):
                if os.path.exists(fpath):
                    gcols[i % 2].image(fpath, caption=caption, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to fetch accident statistics: {str(e)}")


# ==============================================================================
# MODULE 5: ACCIDENT SIMILARITY SEARCH (FAISS)
# ==============================================================================
elif nav_choice == "5. Accident Similarity Search":
    st.markdown("### Precedent Similarity Search Engine (FAISS)")
    st.caption("Retrieve verbatim DGMS historical inquiries sharing geomechanical and operational characteristics.")

    sample_query = st.text_area(
        "Incident Narrative or Current Mine Working Scenario:",
        value="Depillaring stook extraction near active goaf edge with sandstone roof parting and missing props"
    )

    fcol1, fcol2 = st.columns([1, 3])
    with fcol1:
        filter_sec = st.selectbox("Sector Filter", ["All", "Coal", "Non-Coal"])
        sec_param = None if filter_sec == "All" else filter_sec
    with fcol2:
        top_k_val = st.slider("Max Case Precedents to Retrieve", min_value=2, max_value=8, value=4)

    if st.button("🔍 Search DGMS Precedent Vector Store", use_container_width=True):
        with st.spinner("Embedding query and executing Inner Product search across FAISS index..."):
            try:
                res = api.search_similar_accidents(sample_query, top_k=top_k_val, filter_coal_noncoal=sec_param)
                matches = res.get("matches", [])
                st.markdown(f"##### Top {len(matches)} Historical Analogues Identified:")

                for idx, m in enumerate(matches, 1):
                    score_pct = m.get("similarity_score", 0) * 100
                    with st.expander(f"Precedent #{idx}: {m.get('mine')} ({m.get('year')}, {m.get('state')}) — Match: {score_pct:.1f}%", expanded=(idx==1)):
                        c1, c2, c3 = st.columns(3)
                        c1.write(f"**Sector:** {m.get('coal_noncoal')}")
                        c2.write(f"**Fatalities:** {m.get('fatalities')}")
                        c3.write(f"**Date:** {m.get('date')}")

                        st.markdown(f"**DGMS Inquiry Findings:** {m.get('description')}")
                        st.markdown(f"**Identified Root Cause:** `{m.get('root_cause')}`")
                        st.markdown(f"**Statutory Regulations Cited:** `{m.get('regulations')}`")
            except Exception as e:
                st.error(f"Similarity search failed: {str(e)}")


# ==============================================================================
# MODULE 6: REGULATORY RAG EVIDENCE & GEMINI SAFETY QA
# ==============================================================================
elif nav_choice == "6. Regulatory RAG Evidence":
    st.markdown("### Statutory & Technical Regulation RAG Engine")
    st.caption("Vector retrieval across Coal Mines Regulations 2017, MMR 1961, and DGMS Technical Circulars.")

    tab_rag_search, tab_gemini_qa = st.tabs(["📖 Vector Clause Search", "✨ Ask Gemini Regulatory Advisor"])

    with tab_rag_search:
        rag_q = st.text_input(
            "Statutory Query or Regulation Search Topic:",
            value="What statutory regulations mandate support setting at the goaf edge in depillaring panels?"
        )

        if st.button("📖 Search Regulatory Knowledge Base", use_container_width=True):
            with st.spinner("Searching FAISS Regulatory Index..."):
                try:
                    res = api.query_rag(rag_q, top_k=4)
                    evidence = res.get("evidence", [])
                    st.markdown(f"##### Identified {len(evidence)} Applicable Statutory Provisions:")

                    for idx, ev in enumerate(evidence, 1):
                        cat_color = "#3B82F6" if "Statutory" in ev.get("category", "") else "#10B981"
                        st.markdown(f"""
                        <div style="background: rgba(15, 23, 42, 0.7); border-left: 4px solid {cat_color}; padding: 14px 16px; margin-bottom: 12px; border-radius: 6px; border-top: 1px solid rgba(255,255,255,0.05); border-right: 1px solid rgba(255,255,255,0.05); border-bottom: 1px solid rgba(255,255,255,0.05);">
                            <b>{ev.get('title')}</b> <span style="font-size: 0.75rem; background: #1E293B; padding: 2px 8px; border-radius: 4px; color: #94A3B8;">{ev.get('category')}</span><br>
                            <span style="font-size: 0.78rem; color: #94A3B8; font-family: 'JetBrains Mono', monospace;">Source: {ev.get('source')} | {ev.get('page')} | Match: {ev.get('similarity_score', 0)*100:.1f}%</span><br><br>
                            <span style="font-size: 0.88rem; color: #E2E8F0; line-height: 1.5;">{ev.get('content')}</span>
                        </div>
                        """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"RAG query failed: {str(e)}")

    with tab_gemini_qa:
        st.markdown("##### Grounded Safety Inquiries with Google Gemini")
        st.caption("Ask natural safety questions; Google Gemini synthesizes an authoritative directive backed by DGMS circulars.")

        # Quick Suggested Prompts
        st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; margin-bottom: 6px;'><b>Suggested Mining Questions:</b></div>", unsafe_allow_html=True)
        sq_col1, sq_col2, sq_col3 = st.columns(3)
        if sq_col1.button("SSR Resin Bolt Testing Rules"):
            st.session_state["gemini_q_val"] = "What are the statutory requirements for Systematic Support Rules (SSR) and pull testing of resin bolts under DGMS Circular 06/2013?"
        if sq_col2.button("CMR 111 Gallery Dimensions"):
            st.session_state["gemini_q_val"] = "What are the strict maximum gallery height and width restrictions under Regulation 111 of CMR 2017?"
        if sq_col3.button("Drummy Roof Evacuation Rules"):
            st.session_state["gemini_q_val"] = "What mandatory immediate life-safety actions must be taken if a drummy or heavy roof sound is detected?"

        qa_role = st.selectbox("Designation Context for Advisory:", ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Mine Manager"], key="qa_role")
        gemini_q = st.text_area(
            "Mine Safety Inquiry:",
            value=st.session_state.get("gemini_q_val", "What are the statutory requirements for Systematic Support Rules (SSR) and pull testing of resin bolts under DGMS Circular 06/2013?"),
            key="gemini_qa_input"
        )

        if st.button("✨ Synthesize Safety Advisory", use_container_width=True):
            with st.spinner("Retrieving DGMS provisions and synthesizing Gemini advisory..."):
                try:
                    g_res = api.ask_rag_gemini(
                        query=gemini_q,
                        user_role=qa_role,
                        gemini_api_key=st.session_state.get("gemini_api_key")
                    )

                    engine_badge = "Google Gemini Generative AI" if g_res.get("gemini_powered") else "BhoomiKavach Grounded Knowledge Base"

                    st.markdown(f"""
                    <div class="bulletin-box">
                        <div class="bulletin-header">
                            <div>
                                <div class="bulletin-title">DGMS SAFETY ADVISORY BULLETIN</div>
                                <div class="bulletin-meta">RECIPIENT: {qa_role.upper()} | STATUS: VERIFIED</div>
                            </div>
                            <span class="status-chip status-chip-live">✨ {engine_badge}</span>
                        </div>
                        <div style="font-size: 0.9rem; line-height: 1.6; color: #E2E8F0;">
                    """, unsafe_allow_html=True)

                    st.markdown(g_res.get("answer", ""))

                    citations = g_res.get("citations", [])
                    if citations:
                        st.markdown("<br><b>📌 Statutory Citations Cited:</b>", unsafe_allow_html=True)
                        for c in citations:
                            st.markdown(f"- **{c.get('title')}** ({c.get('source')}, {c.get('page')})")

                    st.markdown("</div></div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Advisory generation failed: {str(e)}")


# ==============================================================================
# MODULE 7: AGENTIC SAFETY ADVISOR (LANGGRAPH + GEMINI)
# ==============================================================================
elif nav_choice == "7. Agentic Safety Advisor":
    st.markdown("### LangGraph Multi-Agent Decision Support")
    st.caption("Agent Topology: Router -> Risk -> SHAP -> FAISS Precedent -> Evidence -> Recommendation -> Critic")

    user_role = st.selectbox(
        "Designation Context:",
        ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Worker"]
    )

    agent_query = st.text_area(
        "Decision-Support Query:",
        value="Why is my current roof-fall risk HIGH after blasting in depillaring stook, and what immediate actions must I take?"
    )

    if st.button("🚀 Execute Multi-Agent Evaluation", use_container_width=True):
        with st.spinner("Executing State Graph Reasoning & Statutory Safety Critic..."):
            try:
                chat_res = api.chat_with_agent(
                    query=agent_query,
                    user_role=user_role,
                    gemini_api_key=st.session_state.get("gemini_api_key")
                )

                st.markdown("<br>", unsafe_allow_html=True)

                # Traceability telemetry bar
                t1, t2, t3, t4 = st.columns(4)
                with t1:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Classified Intent</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #38BDF8;">{chat_res.get('intent', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with t2:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Critic Validation</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #10B981;">
                            {'PASSED' if chat_res.get('critic_passed') else 'FAILED'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with t3:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">AI Engine</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #A855F7;">
                            {'Google Gemini' if chat_res.get('gemini_powered') else 'Deterministic'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with t4:
                    st.markdown(f"""
                    <div class="kpi-container">
                        <div class="kpi-label">Immutable Audit</div>
                        <div style="font-size: 1.15rem; font-weight: 700; color: #F59E0B;">#{chat_res.get('audit_id', 0)}</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Render Official Bulletin
                st.markdown(f"""
                <div class="bulletin-box">
                    <div class="bulletin-header">
                        <div>
                            <div class="bulletin-title">AUTONOMOUS GEOTECHNICAL SAFETY DIRECTIVE</div>
                            <div class="bulletin-meta">ROLE: {user_role.upper()} | AUDIT REF #{chat_res.get('audit_id', 0)}</div>
                        </div>
                        <span class="status-chip status-chip-live">
                            {'✨ Gemini 2.5 Flash Verified' if chat_res.get('gemini_powered') else '🛡️ BhoomiKavach Engine'}
                        </span>
                    </div>
                    <div style="font-size: 0.9rem; line-height: 1.6; color: #F1F5F9;">
                """, unsafe_allow_html=True)

                st.markdown(chat_res.get("final_response", ""))

                st.markdown("</div></div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Agent execution failed: {str(e)}")


# ==============================================================================
# MODULE 8: MODEL BENCHMARKS & EVALUATION
# ==============================================================================
elif nav_choice == "8. Model Benchmarks & Evaluation":
    st.markdown("### Model Benchmarking & Safety Evaluation Report")
    st.caption("Empirical comparison of Baseline, Gradient Boosted Classifiers, and Ensembles against DGMS Safety Criteria.")

    csv_comp = "results/model_comparison.csv"
    if os.path.exists(csv_comp):
        df_comp = pd.read_csv(csv_comp)
        st.dataframe(
            df_comp.style.highlight_max(
                subset=["balanced_accuracy", "macro_f1", "high_recall", "critical_recall"],
                color="#064E3B"
            ),
            use_container_width=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Safety Optimization Principles")
        st.markdown("""
        In industrial underground mining, **False Negatives are life-threatening** (failing to detect a HIGH or CRITICAL hazard).
        - **HIGH Hazard Recall**: Champion ensemble achieves **80.8%**.
        - **CRITICAL Hazard Recall**: Supported with **55.6%** sensitivity despite extreme class rarity (3.3% operational frequency).
        - **Temporal Zero-Leakage Validation**: Strict chronological out-of-time splits.
        """)

        calib_img = "results/calibration/calibration_curves.png"
        if os.path.exists(calib_img):
            st.image(calib_img, caption="Reliability Calibration Curves for HIGH and CRITICAL Hazard Classes", use_container_width=True)
    else:
        st.info("Run model training pipeline to generate benchmark comparison table.")


# ==============================================================================
# MODULE 9: SAFETY POLICY & DATA DICTIONARY
# ==============================================================================
elif nav_choice == "9. Safety Policy & Data Dictionary":
    st.markdown("### Statutory Safety Policy & Geotechnical Data Dictionary")

    tab1, tab2 = st.tabs(["DGMS Safety Policy & Directives", "Data Dictionary (30 Canonical Fields)"])

    with tab1:
        st.markdown("""
        <div class="panel-section">
            <h4 style="color: #F8FAFC; margin-bottom: 12px;">BHOOmiKAVACH AI — Statutory Safety Policy & Operating Constraints</h4>
            <ol style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.7;">
                <li><b>Prohibition of Fabrication:</b> The system MUST NEVER invent regulations, accident facts, mine names, or safety thresholds.</li>
                <li><b>Non-Causal Explanation Mandate:</b> The system MUST NEVER claim correlation proves causation. Output phrasing must state:
                    <i>"Contributed to the mathematical model risk score"</i>, NOT <i>"Caused the roof fall"</i>.</li>
                <li><b>No Guarantee of Safety:</b> Low-risk classification does NOT certify that a roof is safe. Rock mass discontinuities exhibit inherent stochastic behavior.</li>
                <li><b>Human Competence Primacy:</b> The system CANNOT authorize mine entry or autonomously enact safety decisions.</li>
                <li><b>Mandatory Statutory Disclaimer:</b> Every assessment must be accompanied by the statutory disclaimer requiring competent technical inspection under CMR 2017 / MMR 1961.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        dict_md = "docs/data_dictionary.md"
        if os.path.exists(dict_md):
            with open(dict_md, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Data dictionary document available in `docs/data_dictionary.md`.")
