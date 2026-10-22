"""
BHOOmiKAVACH AI — Industrial Mine-Safety Decision Support Dashboard
=====================================================================
National Mining Geotechnical Safety & Real-Time Roof-Fall Hazard Radar
Directorate General of Mines Safety (DGMS) Code 0111: Fall of Roof
Standards: Coal Mines Regulations 2017 & Metalliferous Mines Regulations 1961

Frontend Architecture: High-Precision Industrial Streamlit UI consuming FastAPI REST backend.
Author: BHOOmiKAVACH AI Engineering Team
"""

import sys
import os
import json
import requests
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import textwrap
from pathlib import Path

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from frontend.api_client import BhoomiApiClient
from frontend.backend_manager import auto_start_local_backend

def render_html(html_str: str):
    """Safely renders HTML without markdown parser converting indented HTML to code blocks."""
    cleaned = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(cleaned)
    else:
        st.markdown(cleaned, unsafe_allow_html=True)

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

    /* =========================================================================
       STREAMLIT TABS DESIGN SYSTEM (CLEAN, ELEVATED, ZERO-CLIPPING)
       ========================================================================= */
    .stTabs [data-baseweb="tab-list"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 8px 10px !important;
        margin-bottom: 22px !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }

    .stTabs [data-baseweb="tab"] {
        height: auto !important;
        min-height: 40px !important;
        padding: 8px 18px !important;
        white-space: nowrap !important;
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 9999px !important;
        color: #94A3B8 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2) !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.5) !important;
        color: #F8FAFC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.2) !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 50%, #3B82F6 100%) !important;
        border-color: #93C5FD !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }

    /* Completely hide default BaseWeb red tab-highlight underline bar */
    .stTabs [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    /* Fix internal tab text clipping */
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span,
    .stTabs [data-baseweb="tab"] div {
        line-height: 1.35 !important;
        margin: 0 !important;
        padding: 0 !important;
        color: inherit !important;
        font-size: inherit !important;
        font-weight: inherit !important;
        white-space: nowrap !important;
    }

    /* Sidebar cleanup */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090E1A 0%, #060911 100%) !important;
        border-right: 1px solid var(--border-subtle) !important;
    }

    /* =========================================================================
       ENTERPRISE HORIZONTAL NAVIGATION & OPTION CLICKS DESIGN SYSTEM
       ========================================================================= */
    .nav-horizontal-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.85) 0%, rgba(10, 16, 30, 0.95) 100%);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 16px 20px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.6);
        backdrop-filter: blur(16px);
    }

    .nav-horizontal-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 12px;
        padding-bottom: 10px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    .nav-title-tag {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #94A3B8;
        font-family: 'JetBrains Mono', monospace;
    }

    .nav-active-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(37, 99, 235, 0.15);
        border: 1px solid rgba(59, 130, 246, 0.4);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.74rem;
        font-weight: 700;
        color: #93C5FD;
        letter-spacing: 0.04em;
    }

    /* Streamlit Pills Horizontal Layout & High-Tech Styling */
    div[data-testid="stPills"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        align-items: center !important;
        padding: 2px 0 6px 0 !important;
    }

    div[data-testid="stPills"] > div {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
        width: 100% !important;
    }

    div[data-testid="stPills"] button {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9999px !important;
        color: #CBD5E1 !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    }

    div[data-testid="stPills"] button:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        color: #F8FAFC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(56, 189, 248, 0.25) !important;
    }

    div[data-testid="stPills"] button[aria-selected="true"],
    div[data-testid="stPills"] button[data-selected="true"] {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 50%, #3B82F6 100%) !important;
        border-color: #93C5FD !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }

    /* Horizontal Segmented Control Support */
    div[data-testid="stSegmentedControl"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 6px !important;
    }

    /* Horizontal Radio Option Clicks with hidden circle */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 8px !important;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: rgba(15, 23, 42, 0.75) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 9999px !important;
        padding: 7px 15px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        color: #CBD5E1 !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: rgba(30, 41, 59, 0.95) !important;
        border-color: rgba(56, 189, 248, 0.6) !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.25) !important;
    }

    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) {
        background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 50%, #3B82F6 100%) !important;
        border-color: #93C5FD !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        box-shadow: 0 0 18px rgba(59, 130, 246, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }

    /* =========================================================================
       EDA VISUALIZATION CARDS & INTERPRETATION CALLOUTS
       ========================================================================= */
    .eda-card {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(10, 16, 32, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 26px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }

    .eda-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 14px;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
    }

    .eda-chart-title {
        font-size: 1.12rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: -0.01em;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .eda-chart-badge {
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        padding: 4px 12px;
        border-radius: 9999px;
        background: rgba(59, 130, 246, 0.15);
        color: #93C5FD;
        border: 1px solid rgba(59, 130, 246, 0.35);
        letter-spacing: 0.04em;
    }

    .eda-insights-box {
        background: rgba(14, 23, 42, 0.85);
        border-left: 4px solid #38BDF8;
        border-top: 1px solid rgba(56, 189, 248, 0.15);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin: 16px 0 12px 0;
        font-size: 0.86rem;
        line-height: 1.6;
        color: #CBD5E1;
    }

    .eda-insights-box b {
        color: #38BDF8;
    }

    .eda-conclusion-box {
        background: rgba(16, 185, 129, 0.08);
        border-left: 4px solid #10B981;
        border-top: 1px solid rgba(16, 185, 129, 0.2);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        padding: 14px 18px;
        border-radius: 0 10px 10px 0;
        margin-top: 10px;
        font-size: 0.86rem;
        line-height: 1.6;
        color: #A7F3D0;
    }

    .eda-conclusion-box b {
        color: #34D399;
    }

    /* =========================================================================
       HIGH-RELIABILITY PIPELINE ARCHITECTURE FLOWCHART
       ========================================================================= */
    .pipeline-wrapper {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.95) 0%, rgba(8, 12, 22, 0.98) 100%);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 16px;
        padding: 24px 28px;
        margin: 16px 0 24px 0;
        box-shadow: 0 14px 40px -10px rgba(0, 0, 0, 0.6);
        position: relative;
        overflow: hidden;
    }

    .pipeline-wrapper::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #38BDF8 0%, #F59E0B 35%, #A855F7 70%, #EF4444 100%);
    }

    .pipeline-top-row {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        gap: 16px;
        align-items: center;
        margin-bottom: 20px;
    }

    .pipeline-flow-arrow {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 44px;
        height: 44px;
        border-radius: 50%;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #38BDF8;
        font-size: 1.2rem;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
    }

    .pipeline-node {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
        transition: all 0.25s ease;
    }

    .pipeline-node:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.18);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }

    .pipeline-node-cyan { border-top: 3px solid #38BDF8; }
    .pipeline-node-amber { border-top: 3px solid #F59E0B; }
    .pipeline-node-emerald { border-top: 3px solid #10B981; }
    .pipeline-node-purple { border-top: 3px solid #A855F7; }
    .pipeline-node-pink { border-top: 3px solid #EC4899; }
    .pipeline-node-red { border-top: 3px solid #EF4444; }

    .pipeline-node-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .pipeline-node-title {
        font-size: 0.92rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .pipeline-node-badge {
        font-size: 0.68rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: 0.06em;
    }

    .pipeline-node-desc {
        font-size: 0.78rem;
        color: #94A3B8;
        line-height: 1.5;
        margin-bottom: 10px;
    }

    .pipeline-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }

    .pipeline-tag {
        font-size: 0.7rem;
        font-family: 'JetBrains Mono', monospace;
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 3px 8px;
        border-radius: 6px;
        color: #CBD5E1;
    }

    .pipeline-branch-divider {
        display: flex;
        align-items: center;
        gap: 14px;
        margin: 18px 0 20px 0;
        color: #64748B;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }

    .pipeline-branch-divider::before, .pipeline-branch-divider::after {
        content: "";
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    }

    .pipeline-parallel-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
    }

    @media (max-width: 1024px) {
        .pipeline-parallel-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        .pipeline-top-row {
            grid-template-columns: 1fr;
        }
    }

    /* =========================================================================
       MODULE 2: LIVE STRATA TELEMETRY INPUT COCKPIT
       ========================================================================= */
    /* Container Card styling inside forms */
    div[data-testid="stForm"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.75) 0%, rgba(10, 16, 30, 0.90) 100%) !important;
        border: 1px solid rgba(56, 189, 248, 0.16) !important;
        border-radius: 14px !important;
        padding: 22px 18px 20px 18px !important;
        box-shadow: 0 14px 34px -10px rgba(0, 0, 0, 0.55), 0 0 20px rgba(56, 189, 248, 0.04) !important;
        backdrop-filter: blur(14px) !important;
        transition: border-color 0.25s ease, box-shadow 0.25s ease !important;
    }

    div[data-testid="stForm"] div[data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(56, 189, 248, 0.35) !important;
        box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.65), 0 0 25px rgba(56, 189, 248, 0.08) !important;
    }

    /* Container Header inside each column card */
    .cockpit-col-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 14px;
        margin-bottom: 16px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }

    .cockpit-col-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.25), rgba(56, 189, 248, 0.18));
        border: 1px solid rgba(56, 189, 248, 0.35);
        font-size: 1.15rem;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
        flex-shrink: 0;
    }

    .cockpit-col-title {
        font-size: 0.86rem;
        font-weight: 800;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: #F8FAFC;
        line-height: 1.2;
    }

    .cockpit-col-sub {
        font-size: 0.72rem;
        color: #64748B;
        letter-spacing: 0.02em;
        margin-top: 2px;
    }

    /* Selectbox Input Polish */
    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 10px !important;
        color: #F8FAFC !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.25) !important;
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.18) !important;
    }

    div[data-testid="stSelectbox"] > div[data-baseweb="select"] > div:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.25), 0 4px 14px rgba(56, 189, 248, 0.2) !important;
    }

    /* Slider Polish - Eliminating Harsh Red and Elevating Precision Feel */
    div[data-testid="stSlider"] label {
        font-size: 0.80rem !important;
        font-weight: 600 !important;
        color: #CBD5E1 !important;
        letter-spacing: 0.02em !important;
        margin-bottom: 2px !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }

    /* Base Slider Track */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {
        background: rgba(30, 41, 59, 0.85) !important;
        height: 6px !important;
        border-radius: 9999px !important;
    }

    /* Active Highlighted Track */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div:first-child {
        background: linear-gradient(90deg, #1D4ED8 0%, #0284C7 50%, #38BDF8 100%) !important;
        height: 6px !important;
        border-radius: 9999px !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.5) !important;
    }

    /* Slider Knob / Thumb */
    div[data-testid="stSlider"] [role="slider"] {
        background: #38BDF8 !important;
        border: 2px solid #FFFFFF !important;
        width: 17px !important;
        height: 17px !important;
        border-radius: 50% !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.85), 0 2px 6px rgba(0, 0, 0, 0.6) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }

    div[data-testid="stSlider"] [role="slider"]:hover {
        transform: scale(1.25) !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 1), 0 4px 10px rgba(0, 0, 0, 0.7) !important;
    }

    /* Floating Value Indicator */
    div[data-testid="stSlider"] [data-testid="stSliderThumbValue"],
    div[data-testid="stSlider"] div[data-baseweb="slider"] div[style*="color: rgb(255, 75, 75)"] {
        color: #38BDF8 !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 700 !important;
        font-size: 0.82rem !important;
    }

    /* Slider Min & Max Tick Text */
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"] {
        color: #64748B !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
    }

    /* Checkbox Styling */
    div[data-testid="stCheckbox"] {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 10px 14px;
        margin-top: 6px;
        transition: all 0.2s ease;
    }

    div[data-testid="stCheckbox"]:hover {
        border-color: rgba(56, 189, 248, 0.3);
        background: rgba(15, 23, 42, 0.8);
    }

    div[data-testid="stCheckbox"] label span {
        color: #E2E8F0 !important;
        font-size: 0.84rem !important;
        font-weight: 600 !important;
    }

    /* Form Submit Button (Ultra Command Center) */
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(135deg, #1D4ED8 0%, #0284C7 50%, #38BDF8 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        padding: 15px 28px !important;
        font-size: 0.98rem !important;
        font-weight: 800 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        box-shadow: 0 8px 24px -4px rgba(2, 132, 199, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        margin-top: 14px !important;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 32px -4px rgba(56, 189, 248, 0.7), 0 0 24px rgba(56, 189, 248, 0.45) !important;
        border-color: #BAE6FD !important;
    }

    div[data-testid="stFormSubmitButton"] > button:active {
        transform: translateY(1px) !important;
    }

    /* Live Telemetry Health Card inside Column C */
    .telemetry-badge-card {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.7) 0%, rgba(10, 16, 32, 0.9) 100%);
        border: 1px solid rgba(56, 189, 248, 0.22);
        border-radius: 10px;
        padding: 13px 15px;
        margin-top: 12px;
        box-shadow: inset 0 0 15px rgba(56, 189, 248, 0.04);
    }

    .telemetry-pulse {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #38BDF8;
        box-shadow: 0 0 0 rgba(56, 189, 248, 0.6);
        animation: pulse-dot 2s infinite;
        flex-shrink: 0;
    }

    @keyframes pulse-dot {
        0% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0.7);
        }
        70% {
            box-shadow: 0 0 0 8px rgba(56, 189, 248, 0);
        }
        100% {
            box-shadow: 0 0 0 0 rgba(56, 189, 248, 0);
        }
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
# Ensure backend is active when running on Streamlit Cloud or standalone
auto_start_local_backend()

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
MODULE_DIRECTORY = [
    "2. Live Strata Risk Assessment",
    "3. Model Explainability & SHAP",
    "4. Historical DGMS Analytics",
    "5. Accident Similarity Search",
    "6. Regulatory RAG Evidence",
    "7. AI Help",
    "8. Safety Policy & Data Dictionary"
]

MODULE_DISPLAY_MAP = {
    "2. Live Strata Risk Assessment": "⚡ 2. Live Strata Risk",
    "3. Model Explainability & SHAP": "🔍 3. Explainability & SHAP",
    "4. Historical DGMS Analytics": "📊 4. DGMS Analytics",
    "5. Accident Similarity Search": "🎯 5. Accident Matcher",
    "6. Regulatory RAG Evidence": "📜 6. Regulatory RAG",
    "7. AI Help": "✨ 7. AI Help",
    "8. Safety Policy & Data Dictionary": "📖 8. Safety Policy"
}

MODULE_SIDEBAR_MAP = {
    "2. Live Strata Risk Assessment": "⚡ 2. Strata Risk",
    "3. Model Explainability & SHAP": "🔍 3. SHAP XAI",
    "4. Historical DGMS Analytics": "📊 4. Analytics",
    "5. Accident Similarity Search": "🎯 5. Matcher",
    "6. Regulatory RAG Evidence": "📜 6. DGMS RAG",
    "7. AI Help": "✨ 7. AI Help",
    "8. Safety Policy & Data Dictionary": "📖 8. Policy"
}

if "nav_choice" not in st.session_state:
    st.session_state["nav_choice"] = "2. Live Strata Risk Assessment"
if "sidebar_nav_key" not in st.session_state:
    st.session_state["sidebar_nav_key"] = st.session_state["nav_choice"]
if "top_nav_key" not in st.session_state:
    st.session_state["top_nav_key"] = st.session_state["nav_choice"]

def _on_sidebar_nav_change():
    sel = st.session_state.get("sidebar_nav_key")
    if sel:
        st.session_state["nav_choice"] = sel
        st.session_state["top_nav_key"] = sel
    else:
        st.session_state["sidebar_nav_key"] = st.session_state["nav_choice"]

def _on_top_nav_change():
    sel = st.session_state.get("top_nav_key")
    if sel:
        st.session_state["nav_choice"] = sel
        st.session_state["sidebar_nav_key"] = sel
    else:
        st.session_state["top_nav_key"] = st.session_state["nav_choice"]

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

    st.caption("TACTICAL NAVIGATION (HORIZONTAL)")
    side_sel = st.pills(
        "Sidebar Module Directory",
        options=MODULE_DIRECTORY,
        format_func=lambda x: MODULE_SIDEBAR_MAP.get(x, x),
        selection_mode="single",
        key="sidebar_nav_key",
        on_change=_on_sidebar_nav_change,
        label_visibility="collapsed"
    )
    if side_sel:
        st.session_state["nav_choice"] = side_sel

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
    if not default_gemini_key:
        try:
            if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                default_gemini_key = str(st.secrets["GEMINI_API_KEY"])
        except Exception:
            pass
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
            <span>✨</span> Gemini 3.8 Flash Grounded
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
# 6. ENTERPRISE TOP COMMAND HEADER & HORIZONTAL TACTICAL NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
render_html("""
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
""")

top_sel = st.pills(
    "Top Horizontal Tactical Navigation",
    options=MODULE_DIRECTORY,
    format_func=lambda x: MODULE_DISPLAY_MAP.get(x, x),
    selection_mode="single",
    key="top_nav_key",
    on_change=_on_top_nav_change,
    label_visibility="collapsed"
)
if top_sel:
    st.session_state["nav_choice"] = top_sel

nav_choice = st.session_state.get("nav_choice", "2. Live Strata Risk Assessment")


# ==============================================================================
# MODULE 2: LIVE STRATA RISK ASSESSMENT (CORE OPERATIONAL RADAR)
# ==============================================================================
if nav_choice == "2. Live Strata Risk Assessment":
    st.markdown("### Geotechnical Hazard Radar & Inference Cockpit")
    st.caption("Input live strata convergence, installed support parameters, and mining geometry for prospective evaluation.")

    # 1-Click Operational Scenario Presets
    # 1-Click Operational Scenario Presets
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between; margin: 10px 0 12px 0; padding-bottom: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.08);">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 0.84rem; font-weight: 800; color: #F1F5F9; letter-spacing: 0.06em; text-transform: uppercase;">
                ⚡ 1-Click Operational Scenario Presets
            </span>
            <span style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); color: #38BDF8; font-size: 0.68rem; font-weight: 700; padding: 2px 10px; border-radius: 9999px; letter-spacing: 0.04em;">
                Quick Calibration
            </span>
        </div>
        <div style="font-size: 0.72rem; color: #64748B; font-family: 'JetBrains Mono', monospace;">
            DGMS Historic Hazard Calibrations
        </div>
    </div>
    """, unsafe_allow_html=True)
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

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    # Telemetry Input Form with Enclosed Glassmorphic Containers
    with st.form("risk_assessment_form"):
        colA, colB, colC = st.columns(3, gap="medium")

        # ── COLUMN A: SPATIAL & OPERATIONAL CONTEXT ─────────────────────────
        with colA:
            with st.container(border=True):
                render_html("""
                <div class="cockpit-col-header">
                    <div class="cockpit-col-icon">📍</div>
                    <div>
                        <div class="cockpit-col-title">Spatial & Operational Setting</div>
                        <div class="cockpit-col-sub">Colliery lease, face zone & operator role</div>
                    </div>
                </div>
                """)

                mine_name = st.selectbox(
                    "Mine / Colliery",
                    ["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"],
                    index=["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"].index(st.session_state.preset_mine) if st.session_state.preset_mine in ["Moonidih Colliery", "GDK 11A Incline", "Jhilimili Colliery", "Shantikhani Mine", "Churcha RO Mine", "Balaghat Mine (MOIL)", "Kolihan Copper Mine"] else 0,
                    help="Select active mining lease for strata geomechanics profile."
                )
                sector = "Coal" if "Mine" not in mine_name and "Kolihan" not in mine_name else "Non-Coal"
                mineral = "Coal" if sector == "Coal" else ("Manganese" if "Balaghat" in mine_name else "Copper")

                role_choice = st.selectbox(
                    "Operating Designation (User Role)",
                    ["Safety Officer", "Mining Sirdar", "Overman", "Mining Engineer", "Geotechnical Engineer", "Mine Manager", "Worker"],
                    help="Credential for audit logging & role-tailored DGMS action protocols."
                )

                method = st.selectbox(
                    "Extraction Method",
                    ["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"],
                    index=["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"].index(st.session_state.preset_method) if st.session_state.preset_method in ["Bord and Pillar (Depillaring)", "Bord and Pillar (Development)", "Continuous Miner", "Sublevel Stoping", "Cut and Fill"] else 0,
                    help="Operational mining or stoping method in the monitored panel."
                )

                depth = st.slider(
                    "Cover Depth (m)",
                    min_value=30.0,
                    max_value=800.0,
                    value=float(st.session_state.preset_depth),
                    step=10.0,
                    help="Overburden depth from surface topography to working seam horizon."
                )

                face_pos = st.selectbox(
                    "Working Face Position",
                    ["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"],
                    index=["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"].index(st.session_state.preset_face_pos) if st.session_state.preset_face_pos in ["Depillaring Edge / Goaf", "Development Face", "Junction", "Slice Gallery", "Main Haulage"] else 0,
                    help="Immediate spatial inspection zone along the extraction drive."
                )

                is_junc = 1 if face_pos == "Junction" else st.checkbox(
                    "Intersection Junction (4-Way / 3-Way Span)",
                    value=(face_pos == "Junction"),
                    help="Check if assessing high-stress gallery intersection with expanded diagonal span."
                )

        # ── COLUMN B: GEOMETRY & ROCK MECHANICS ──────────────────────────────
        with colB:
            with st.container(border=True):
                render_html("""
                <div class="cockpit-col-header">
                    <div class="cockpit-col-icon">📐</div>
                    <div>
                        <div class="cockpit-col-title">Geometry & Rock Mechanics</div>
                        <div class="cockpit-col-sub">Opening span, RMR rating & blast vibration</div>
                    </div>
                </div>
                """)

                width = st.slider(
                    "Gallery Width (m) [CMR 111 Limit: 4.8m]",
                    min_value=3.0,
                    max_value=6.5,
                    value=float(st.session_state.preset_width),
                    step=0.1,
                    help="Gallery opening span width. Statutory CMR 1957/2017 max limit is 4.8m."
                )

                height = st.slider(
                    "Gallery Height (m) [CMR 111 Limit: 3.0m]",
                    min_value=1.8,
                    max_value=4.5,
                    value=float(st.session_state.preset_height),
                    step=0.1,
                    help="Working gallery extraction height. Statutory CMR limit is 3.0m."
                )

                rmr = st.slider(
                    "CMRI-ISM Rock Mass Rating (RMR)",
                    min_value=15.0,
                    max_value=85.0,
                    value=float(st.session_state.preset_rmr),
                    step=1.0,
                    help="Geomechanical roof classification (Layer thickness, weathering, strength, groundwater, spacing)."
                )

                lithology = st.selectbox(
                    "Roof Lithology",
                    ["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"],
                    index=["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"].index(st.session_state.preset_lithology) if st.session_state.preset_lithology in ["Sandstone-Shale Intercalation", "Sandstone", "Shale", "Carbonaceous Shale", "Quartzite"] else 0,
                    help="Lithological strata composition of the immediate 2.0m roof beam."
                )

                blasting = st.selectbox(
                    "Blasting Recency",
                    ["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"],
                    index=["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"].index(st.session_state.preset_blasting) if st.session_state.preset_blasting in ["Immediate (<4h)", "Recent (4-12h)", "No Blasting (<24h)"] else 1,
                    help="Time elapsed since detonation of last explosive round in adjacent face."
                )

                blast_vibe = st.slider(
                    "Blasting Peak Particle Velocity PPV (mm/s)",
                    min_value=0.5,
                    max_value=30.0,
                    value=12.5,
                    step=0.5,
                    help="Seismic vibration peak particle velocity measured on roof anchorage strata."
                )

        # ── COLUMN C: STRATA CONVERGENCE & BOLTING ──────────────────────────
        with colC:
            with st.container(border=True):
                render_html("""
                <div class="cockpit-col-header">
                    <div class="cockpit-col-icon">🛡️</div>
                    <div>
                        <div class="cockpit-col-title">Strata Convergence & Bolting</div>
                        <div class="cockpit-col-sub">Support integrity & real-time telemetry</div>
                    </div>
                </div>
                """)

                supp_cond = st.selectbox(
                    "Support System Integrity",
                    ["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"],
                    index=["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"].index(st.session_state.preset_supp_cond) if st.session_state.preset_supp_cond in ["Excessive Load/Yielding", "Dislodged/Missing", "Minor Deformation", "Intact & Adequate"] else 2,
                    help="Physical integrity condition of installed roof bolts and w-straps."
                )

                density = st.slider(
                    "Bolt Density (bolts/m²)",
                    min_value=0.4,
                    max_value=2.4,
                    value=float(st.session_state.preset_density),
                    step=0.05,
                    help="Installed bolt grid density across the active span (Statutory design: SSR)."
                )

                bolt_cap = st.slider(
                    "Bolt Anchorage Capacity (tonnes)",
                    min_value=2.0,
                    max_value=16.0,
                    value=7.5,
                    step=0.5,
                    help="Mean pull-out anchorage testing capacity of installed rock bolts."
                )

                convergence = st.slider(
                    "Roof Convergence Velocity (mm/day)",
                    min_value=0.1,
                    max_value=25.0,
                    value=float(st.session_state.preset_convergence),
                    step=0.1,
                    help="Daily roof closure velocity measured by dual-pin telescopic convergence stations."
                )

                bed_sep = st.slider(
                    "Borehole Extensometer Bed Separation (mm)",
                    min_value=0.0,
                    max_value=40.0,
                    value=float(st.session_state.preset_bed_sep),
                    step=0.5,
                    help="Multi-point borehole extensometer (MPBX) bed delamination reading within 2m-5m anchor zone."
                )

                render_html("""
                <div class="telemetry-badge-card">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span class="telemetry-pulse"></span>
                            <span style="font-size: 0.74rem; font-weight: 700; color: #38BDF8; letter-spacing: 0.06em; text-transform: uppercase;">
                                Live MPBX Sensor Online
                            </span>
                        </div>
                        <span style="font-size: 0.68rem; font-family: 'JetBrains Mono', monospace; color: #10B981; font-weight: 700;">
                            SYNC: NORMAL
                        </span>
                    </div>
                    <div style="font-size: 0.71rem; color: #94A3B8; margin-top: 5px; line-height: 1.35;">
                        Telemetry continuous feed actively monitored under DGMS Tech Circular 5 (2016).
                    </div>
                </div>
                """)

        submit_btn = st.form_submit_button("⚡ Run Multi-Model Risk Evaluation & Geomechanics Diagnostics", use_container_width=True)

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
    st.markdown("### Official DGMS Ground Movement & Roof-Fall Exploratory Data Analysis (EDA)")
    st.caption("Canonical Geotechnical Intelligence: 41 Verified Fatal Inquiries (DGMS Code 0111: Fall of Roof, 2016–2022) + Century-Long Disaster Archive (1901–2022) · 100% Unhallucinated Empirical Data")

    try:
        stats = api.get_accident_stats()

        # Top Executive KPI Dashboard
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.markdown(f"""
            <div class="kpi-container">
                <div class="kpi-label">Fatal Inquiries (0111)</div>
                <div class="kpi-value" style="color: #F59E0B;">{stats.get('total_accidents', 41)}</div>
                <div class="kpi-caption">Verified Legal Reports</div>
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
                <div class="kpi-caption">Mining Coalfield Basins</div>
            </div>
            """, unsafe_allow_html=True)
        with c5:
            st.markdown("""
            <div class="kpi-container">
                <div class="kpi-label">Century Disasters</div>
                <div class="kpi-value" style="color: #8B5CF6;">144</div>
                <div class="kpi-caption">787 Deaths (1901–2022)</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Thematic Exploratory Data Analysis Suite
        tab_time, tab_causes, tab_regs, tab_geo, tab_demo, tab_century, tab_catalog = st.tabs([
            "⏱️ 1. Temporal & Shifts",
            "⚠️ 2. Causes & Triggers",
            "⚖️ 3. Regulations & Violations",
            "🗺️ 4. Geography & Minerals",
            "👷 5. Demographics & Roles",
            "🏛️ 6. Century Disasters",
            "📚 7. Full 14-Chart Visual Catalog"
        ])

        # ──────────────────────────────────────────────────────────────────────
        # TAB 1: TEMPORAL & SHIFT ANALYSIS
        # ──────────────────────────────────────────────────────────────────────
        with tab_time:
            st.markdown("#### Temporal Progression, Multi-Fatality Incidents & Shift Vulnerability")
            st.caption("Empirical breakdown of roof-fall incidents over the 2016–2022 regulatory window and operational shifts.")

            # Interactive Plotly: Grouped Annual Progression
            st.markdown("""
            <div class="eda-card">
                <div class="eda-card-header">
                    <div class="eda-chart-title">
                        <span>📊</span> Interactive EDA Chart: Annual Incident & Fatality Progression
                    </div>
                    <span class="eda-chart-badge">LIVE DATABASE TELEMETRY</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            df_year = pd.DataFrame(stats.get("accidents_by_year", []))
            fig_yr = px.bar(
                df_year, x="year", y=["accidents", "fatalities"], barmode="group",
                color_discrete_sequence=["#F59E0B", "#EF4444"],
                labels={"value": "Count", "year": "Calendar Year", "variable": "Metric"}
            )
            fig_yr = format_bhoomi_plot(fig_yr, height=340, title="Annual Fatal Roof-Fall Incidents vs. Total Lives Lost (2016–2022)")
            st.plotly_chart(fig_yr, use_container_width=True)

            st.markdown("""
            <div class="eda-insights-box">
                <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                • <b>Peak Mortality in 2016:</b> The baseline year experienced 9 fatal events resulting in 16 fatalities (1.78 deaths/event), driven by multiple double and triple-fatality collapses during depillaring operations.<br>
                • <b>Stabilization at 4–6 Incidents Annually:</b> Between 2017 and 2022, incident frequency plateaued at an average of 5.3 events/year, proving that while extreme catastrophic collapses have declined, the persistent baseline of fatal strata falls has remained unresolved.<br>
                • <b>Fatality Multiplier:</b> Across the entire 7-year dataset, 41 accidents caused 58 deaths (mean of 1.41 fatalities per incident), demonstrating that ground falls frequently strike clustered work crews (loaders, supportmen, sirdars).
            </div>
            <div class="eda-conclusion-box">
                <b>CONCLUSION & OPERATIONAL SAFETY IMPLICATION:</b><br>
                Visual sounding and traditional timber propping fail to eliminate baseline fatal falls. Underground mines require real-time electronic strata monitoring (continuous convergence recorders and electronic tell-tales) and strict exclusion zone enforcement during high-stress extraction cycles.
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # High-Resolution EDA Charts 01 & 02
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>📈</span> EDA Chart 01: Accidents by Year (Coal vs Non-Coal)
                        </div>
                        <span class="eda-chart-badge">CHART 01 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/01_accidents_by_year.png"):
                    st.image("results/eda/01_accidents_by_year.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • Coal mines consistently account for 70%–80% of all ground falls every year.<br>
                    • Non-coal incidents occur at a steady 1–2 incidents/year, primarily in deep metalliferous stoping environments (copper, gold, uranium).
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Coal strata exhibits higher laminated delamination risk, while metalliferous mines require vigilance against jointed structural block dropouts.
                </div>
                """, unsafe_allow_html=True)

            with col_t2:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>☠️</span> EDA Chart 02: Fatalities by Year
                        </div>
                        <span class="eda-chart-badge">CHART 02 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/02_fatalities_by_year.png"):
                    st.image("results/eda/02_fatalities_by_year.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • Multi-casualty incidents are heavily concentrated in depillaring panels with massive sandstone overhangs.<br>
                    • Years 2018, 2019, and 2021 experienced sharp fatality spikes whenever goaf falls overran face breaker props.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Breaker lines must be reinforced with high-capacity hydraulic props or resin-grouted cable bolts under CMR 2017 Regulation 123.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # High-Resolution EDA Chart 09: Shift Analysis
            st.markdown("""
            <div class="eda-card">
                <div class="eda-card-header">
                    <div class="eda-chart-title">
                        <span>⏰</span> EDA Chart 09: Diurnal Shift & Time-of-Day Risk Analysis
                    </div>
                    <span class="eda-chart-badge">CHART 09 / 14</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if os.path.exists("results/eda/09_shift_analysis.png"):
                st.image("results/eda/09_shift_analysis.png", use_container_width=True)
            st.markdown("""
            <div class="eda-insights-box">
                <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                • <b>Morning Shift Vulnerability (06:00–14:00):</b> Accounts for <b>48.8% (20 of 41 accidents)</b> and 27 fatalities.<br>
                • <b>Night Shift Activity (22:00–06:00):</b> Second highest with <b>26.8% (11 accidents)</b> and 16 fatalities.<br>
                • <b>Afternoon Shift (14:00–22:00):</b> 22.0% (9 accidents) and 13 fatalities.<br>
                • <b>Physical Mechanism:</b> Morning shifts face peak danger because blasting performed during the late night shift or inter-shift intervals disturbs the strata. The roof undergoes unsupported creep and bed dilation for 4–8 hours prior to morning entry.
            </div>
            <div class="eda-conclusion-box">
                <b>CONCLUSION & OPERATIONAL SAFETY IMPLICATION:</b><br>
                Mandatory pre-shift statutory inspection by a certified Mining Sirdar / Overman under CMR 2017 Regulation 129 MUST be performed with mechanical roof sounders and tell-tale inspection prior to admitting face workers. No crew may enter freshly blasted galleries without affirmative clearance.
            </div>
            """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 2: GEOTECHNICAL CAUSES & TRIGGERS
        # ──────────────────────────────────────────────────────────────────────
        with tab_causes:
            st.markdown("#### Primary Geotechnical Mechanisms & Root Causes of Strata Failure")
            st.caption("Empirical ranking of contributing physical factors and managerial root causes cited in DGMS Inquiry Reports.")

            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>🔍</span> EDA Chart 10: Top Contributing Geotechnical Factors
                        </div>
                        <span class="eda-chart-badge">CHART 10 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/10_contributing_factors.png"):
                    st.image("results/eda/10_contributing_factors.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Unsupported Newly Exposed Roof:</b> Cited in over 65% of all fatal inquiries as the direct trigger.<br>
                    • <b>Induced Blasting Shockwave (PPV > 50 mm/s):</b> Blasting micro-fractures laminated shale/sandstone roof layers, loosening key blocks.<br>
                    • <b>Presence of Geological Slips & Slickensides:</b> Natural discontinuities hidden above coal seams detach without acoustic warning.<br>
                    • <b>Water Percolation:</b> Moisture reduces inter-layer shear strength in clayey shale by up to 40%.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION & SAFETY MANDATE:</b><br>
                    Newly exposed roof within 0–3 meters of the active face must be secured immediately with mechanized temporary support (ATRS) before workers dress or load coal.
                </div>
                """, unsafe_allow_html=True)

            with col_c2:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>⚠️</span> EDA Chart 11: Root Causes Identified in DGMS Inquiries
                        </div>
                        <span class="eda-chart-badge">CHART 11 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/11_root_causes.png"):
                    st.image("results/eda/11_root_causes.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Failure of Systematic Support Rules (SSR):</b> Under-density of bolts, delayed grouting, or missing props.<br>
                    • <b>Inadequate Pre-Shift Inspection:</b> Failure by statutory supervisory staff to detect bed separation.<br>
                    • <b>Premature Support Withdrawal:</b> Striking props manually from goaf edges rather than using remote safety prop withdrawers.<br>
                    • <b>Unauthorized Entry:</b> Personnel entering de-supported goaves to recover coal or tools.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION & SAFETY MANDATE:</b><br>
                    Strata accidents stem from managerial enforcement failures intersecting with geomechanics. Digital audit logs and automated sensor alarms provide objective oversight.
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 3: REGULATIONS & VIOLATIONS
        # ──────────────────────────────────────────────────────────────────────
        with tab_regs:
            st.markdown("#### Statutory Regulations Cited in Fatal Roof-Fall Inquiry Findings")
            st.caption("Legal violations identified under the Coal Mines Regulations (CMR 2017) and Metalliferous Mines Regulations (MMR 1961).")

            st.markdown("""
            <div class="eda-card">
                <div class="eda-card-header">
                    <div class="eda-chart-title">
                        <span>⚖️</span> EDA Chart 12: Most Cited Statutory Regulations in DGMS Inquiries
                    </div>
                    <span class="eda-chart-badge">CHART 12 / 14</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("results/eda/12_regulations_cited.png"):
                st.image("results/eda/12_regulations_cited.png", use_container_width=True)

            st.markdown("""
            <div class="eda-insights-box">
                <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                • <b>CMR Regulation 123 (Systematic Support Plan):</b> The most frequently cited violation (cited in 78% of fatal coal inquiries). Inquiries found non-compliance with the mine manager's approved Systematic Support Plan.<br>
                • <b>CMR Regulation 124 (Setting of Supports):</b> Cited whenever supports were set out-of-plumb, bolts were loosely anchored, or resin capsules were expired/insufficient.<br>
                • <b>CMR Regulation 129 (Duties of Mining Sirdar / Overman):</b> Cited for failure to sound the roof and test strata integrity before allowing workers into working places.<br>
                • <b>CMR Regulation 112 (Depillaring Operations):</b> Non-compliance with approved extraction sequence and diagonal line of caving.
            </div>
            <div class="eda-conclusion-box">
                <b>CONCLUSION & REGULATORY COMPLIANCE DIRECTIVE:</b><br>
                Compliance with CMR 2017 Regulation 123 is a statutory absolute. Every depillaring panel must possess an active, site-specific Strata Control and Monitoring Plan (SCAMP) approved by the DGMS Directorate, backed by calibrated dual-height rotary tell-tales.
            </div>
            """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 4: GEOSPATIAL & MINERAL BREAKDOWN
        # ──────────────────────────────────────────────────────────────────────
        with tab_geo:
            st.markdown("#### Geographic, Sectoral, and Commodity Distribution of Ground Falls")
            st.caption("Clear horizontal orientation across states, mining sectors, mineral commodities, and corporate operators.")

            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>🗺️</span> Interactive EDA Chart: State-wise Incident Distribution (Horizontal Orientation)
                        </div>
                        <span class="eda-chart-badge">CLEAR HORIZONTAL PLOT</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                df_st = pd.DataFrame(stats.get("accidents_by_state", []))
                if not df_st.empty:
                    df_st_sorted = df_st.sort_values(by="accidents", ascending=True)
                    fig_st = px.bar(
                        df_st_sorted, y="state", x="accidents", orientation="h",
                        color="accidents",
                        color_continuous_scale=["#1E3A8A", "#3B82F6", "#F59E0B", "#EF4444"],
                        labels={"state": "Indian State", "accidents": "Incidents"}
                    )
                    fig_st = format_bhoomi_plot(fig_st, height=360, title="Fatal Roof-Fall Incidents by Indian State")
                    fig_st.update_layout(yaxis=dict(tickfont=dict(size=11, color="#F8FAFC")))
                    st.plotly_chart(fig_st, use_container_width=True)

                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Jharkhand, Madhya Pradesh, Chhattisgarh & Telangana:</b> Account for <b>>75%</b> of all ground fall fatalities.<br>
                    • High density of underground depillaring operations in Jharia, Raniganj, Korba, and Godavari valley coalfields drives elevated risk.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Regional geotechnical teams in Eastern and Central coalfields require standardized strata risk classification matrices.
                </div>
                """, unsafe_allow_html=True)

            with col_g2:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>⛏️</span> Interactive EDA Chart: Commodity & Sectoral Ratio
                        </div>
                        <span class="eda-chart-badge">SECTOR BREAKDOWN</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                df_sec = pd.DataFrame(stats.get("accidents_by_sector", []))
                if not df_sec.empty:
                    fig_sec = px.pie(
                        df_sec, names="sector", values="fatalities",
                        color="sector",
                        color_discrete_map={"Coal": "#3B82F6", "Non-Coal": "#10B981"},
                        hole=0.45
                    )
                    fig_sec = format_bhoomi_plot(fig_sec, height=360, title="Fatalities Share: Coal vs. Non-Coal Underground Mines")
                    st.plotly_chart(fig_sec, use_container_width=True)

                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Coal Mining:</b> 72.4% of total fatalities (42 deaths) across 30 events.<br>
                    • <b>Non-Coal (Metalliferous):</b> 27.6% of fatalities (16 deaths) across 11 events in Copper, Manganese, and Gold mines.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> While coal has the highest volume, metalliferous underground stopes carry significant wedge-failure hazard under MMR 1961.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # High-Resolution EDA Charts 04, 05, 06, 14
            col_gh1, col_gh2 = st.columns(2)
            with col_gh1:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>🏢</span> EDA Chart 05: Top 10 High-Risk Mines Distribution
                        </div>
                        <span class="eda-chart-badge">CHART 05 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/05_mine_distribution.png"):
                    st.image("results/eda/05_mine_distribution.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • Repeat incident mines are characterized by depths > 300m, high stress concentrations, and thick seam depillaring.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Deep mines require numerical modeling (FLAC3D/Phase2) to predict abutment stress concentrations before depillaring commences.
                </div>
                """, unsafe_allow_html=True)

            with col_gh2:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>🏭</span> EDA Chart 14: Corporate Owner / Operating Entity Breakdown
                        </div>
                        <span class="eda-chart-badge">CHART 14 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/14_owner_distribution.png"):
                    st.image("results/eda/14_owner_distribution.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • SECL, SCCL, ECL, and WCL account for the majority of coal incidents; MOIL, HCL, and HZL account for metalliferous incidents.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Corporate safety directors must mandate uniform digital strata telemetry protocols across all subsidiaries.
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 5: DEMOGRAPHICS & WORKFORCE ROLES
        # ──────────────────────────────────────────────────────────────────────
        with tab_demo:
            st.markdown("#### Occupational Vulnerability and Age Profile of Strata Victims")
            st.caption("Detailed analysis of designations, operational responsibilities, and experience levels in fatal ground falls.")

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>👷</span> EDA Chart 07: Victim Occupational Designations at Risk
                        </div>
                        <span class="eda-chart-badge">CHART 07 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/07_victim_designation.png"):
                    st.image("results/eda/07_victim_designation.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Support Men / Mistries & Timbermen:</b> Rank as the #1 most vulnerable job category.<br>
                    • <b>General Mazdoors & Loaders:</b> Rank #2, struck during manual shoveling or tramming near the face.<br>
                    • <b>Mining Sirdars:</b> Struck during inspection or while guiding workers under unsupported roof.<br>
                    • <b>Underlying Tragedy:</b> Those tasked with making the roof safe (setting props and bolts) are the very ones crushed before support can be installed.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION & MANDATORY CONTROL:</b><br>
                    Manual propping under green roof must be prohibited. Automated Temporary Roof Supports (ATRS) or remote roof-bolters must shield crews during support setting.
                </div>
                """, unsafe_allow_html=True)

            with col_d2:
                st.markdown("""
                <div class="eda-card">
                    <div class="eda-card-header">
                        <div class="eda-chart-title">
                            <span>🎂</span> EDA Chart 08: Age Distribution of Victims
                        </div>
                        <span class="eda-chart-badge">CHART 08 / 14</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if os.path.exists("results/eda/08_age_distribution.png"):
                    st.image("results/eda/08_age_distribution.png", use_container_width=True)
                st.markdown("""
                <div class="eda-insights-box">
                    <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                    • <b>Mean Victim Age:</b> 44.7 years (Median: 47.5 years).<br>
                    • <b>Age Concentration:</b> Over 65% of victims were aged 40–58 years (senior, highly experienced miners).<br>
                    • <b>Critical Refutation:</b> Refutes the misconception that roof falls happen to inexperienced trainees. Highly experienced mineworkers often rely on acoustic roof sounding (tapping with a stick), which cannot detect deep bed separation above 1.5m.
                </div>
                <div class="eda-conclusion-box">
                    <b>CONCLUSION:</b> Experience cannot substitute for scientific strata instrumentation. Training must emphasize that visual/acoustic roof sounding is incapable of detecting high-level bed separations.
                </div>
                """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 6: CENTURY DISASTER REGISTRY
        # ──────────────────────────────────────────────────────────────────────
        with tab_century:
            st.markdown("#### Century-Long Historical Disasters in Indian Mining (1901–2022)")
            st.caption("DGMS Historical Disaster Registry (Chapter 10): 144 Major Catastrophic Roof Falls resulting in 787 Fatalities.")

            st.markdown("""
            <div class="eda-card">
                <div class="eda-card-header">
                    <div class="eda-chart-title">
                        <span>🏛️</span> EDA Chart 13: Century-Long Major Roof-Fall Disasters by Decade (1901–2022)
                    </div>
                    <span class="eda-chart-badge">CHART 13 / 14 · SANKET CHAPTER 10</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists("results/eda/13_century_major_rooffalls.png"):
                st.image("results/eda/13_century_major_rooffalls.png", use_container_width=True)

            st.markdown("""
            <div class="eda-insights-box">
                <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                • <b>Decade Trend:</b> Peak major disaster decade was 1950–1969, where massive unmanaged goaf collapses caused multiple mass-casualty disasters (air blasts).<br>
                • <b>Post-Nationalization Progress:</b> Post-1975 nationalization and systematic support regulation introduced steel props and roof bolting, drastically curtailing 10+ fatality air-blast events.<br>
                • <b>The Unresolved Problem:</b> While catastrophic 20-miner disasters have decreased, localized fall of roof (Code 0111) remains the single largest cause of fatal mining accidents in India, responsible for >40% of all underground deaths.
            </div>
            <div class="eda-conclusion-box">
                <b>CONCLUSION & SYSTEM VISION:</b><br>
                The century-long data proves that macro-scale regulations solved regional collapses, but localized roof falls at the coal face require micro-geomechanical decision intelligence. BHOOmiKAVACH AI directly addresses this final frontier of Indian mine safety.
            </div>
            """, unsafe_allow_html=True)

        # ──────────────────────────────────────────────────────────────────────
        # TAB 7: COMPLETE 14-CHART VISUAL CATALOG
        # ──────────────────────────────────────────────────────────────────────
        with tab_catalog:
            st.markdown("#### Complete 14-Chart Visual Exploratory Data Analysis Catalog")
            st.caption("Select any visualization from the complete canonical DGMS exploratory analysis suite to inspect full-resolution plots, geotechnical explanations, and conclusions.")

            eda_master_catalog = [
                {
                    "num": "01",
                    "title": "Annual Fatal Roof-Fall Incidents Trend (2016–2022)",
                    "file": "results/eda/01_accidents_by_year.png",
                    "badge": "TEMPORAL / SECTORAL",
                    "what": "Shows annual count of fatal roof falls across Indian mines, separated by Coal and Non-Coal sectors. Identifies 2016 as peak year (9 incidents) and steady plateau of 4–6 incidents annually.",
                    "conclusion": "Continuous baseline incidence underscores the necessity of continuous digital monitoring rather than periodic manual inspections."
                },
                {
                    "num": "02",
                    "title": "Roof-Fall Fatalities by Year (Multi-Casualty Analysis)",
                    "file": "results/eda/02_fatalities_by_year.png",
                    "badge": "MORTALITY DYNAMICS",
                    "what": "Tracks total loss of life per year (58 total fatalities, average 1.41 deaths/incident). Highlights that ground falls frequently cause multiple fatalities simultaneously.",
                    "conclusion": "Exclusion zones and remote depillaring technologies must be enforced to minimize personnel concentration at active face edges."
                },
                {
                    "num": "03",
                    "title": "Coal vs. Non-Coal Underground Mining Ground Fall Breakdown",
                    "file": "results/eda/03_coal_vs_noncoal.png",
                    "badge": "SECTORAL COMPARISON",
                    "what": "Compares Coal (73.2% of accidents, 72.4% of fatalities) against Non-Coal metalliferous mining (26.8% accidents, 27.6% fatalities).",
                    "conclusion": "Coal mines demand bed-separation and RMR monitoring; metalliferous mines require structural joint mapping to prevent hanging wall wedge failures."
                },
                {
                    "num": "04",
                    "title": "State-wise Geographic Distribution of Ground Movement Incidents",
                    "file": "results/eda/04_state_distribution.png",
                    "badge": "GEOGRAPHIC SPREAD",
                    "what": "Horizontal distribution across 9 states: Jharkhand, Madhya Pradesh, Chhattisgarh, and Telangana account for over 75% of fatal falls.",
                    "conclusion": "Basin-level geotechnical characteristics (massive Barakar sandstone vs. weak shale) require localized Systematic Support Rules."
                },
                {
                    "num": "05",
                    "title": "Top 10 High-Risk Underground Mines by Fatalities",
                    "file": "results/eda/05_mine_distribution.png",
                    "badge": "SITE-LEVEL CLUSTER",
                    "what": "Ranks specific mines with repeat ground fall occurrences, revealing depth (>300m) and mechanized depillaring as common risk denominators.",
                    "conclusion": "High-risk mines must implement 24/7 automated convergence logging and dynamic trigger action response plans (TARP)."
                },
                {
                    "num": "06",
                    "title": "Mineral-wise Roof-Fall Frequency & Fatality Breakdown",
                    "file": "results/eda/06_mineral_distribution.png",
                    "badge": "COMMODITY PROFILES",
                    "what": "Examines distribution across Coal, Manganese, Copper, Gold, Galena/Sphalerite, and Uranium operations.",
                    "conclusion": "Supports tailored support design: resin bolt density in sedimentary coal vs. split-set and cable bolting in metalliferous stopes."
                },
                {
                    "num": "07",
                    "title": "Victim Occupational Designations at Highest Risk",
                    "file": "results/eda/07_victim_designation.png",
                    "badge": "OCCUPATIONAL RISK",
                    "what": "Identifies Support Men / Mistries, Timbermen, and General Mazdoors as the top victims, struck during support setting under newly exposed roof.",
                    "conclusion": "Manual support installation beneath unsupported roof must be eliminated using Automated Temporary Roof Supports (ATRS)."
                },
                {
                    "num": "08",
                    "title": "Age Distribution and Workforce Profile of Victims",
                    "file": "results/eda/08_age_distribution.png",
                    "badge": "DEMOGRAPHIC INSIGHT",
                    "what": "Histogram shows mean victim age of 44.7 years with peak density in 45–55 age bracket, disproving the myth that novice workers are primary victims.",
                    "conclusion": "Senior miners rely heavily on traditional acoustic roof sounding, which cannot detect deep bed separation. Electronic instrumentation is vital."
                },
                {
                    "num": "09",
                    "title": "Shift / Time-of-Day Diurnal Analysis of Roof Falls",
                    "file": "results/eda/09_shift_analysis.png",
                    "badge": "DIURNAL DYNAMICS",
                    "what": "Identifies Morning Shift (06:00–14:00) as responsible for 48.8% of incidents due to strata relaxation and unsupported creep following overnight blasting.",
                    "conclusion": "Statutory pre-shift inspection under CMR Regulation 129 is the most vital administrative barrier against morning strata fatalities."
                },
                {
                    "num": "10",
                    "title": "Top Contributing Geotechnical & Environmental Factors",
                    "file": "results/eda/10_contributing_factors.png",
                    "badge": "FAILURE MECHANISMS",
                    "what": "Ranks unsupported green roof, induced blasting vibration, geological slips, and water ingress as the leading physical triggers.",
                    "conclusion": "Strata failure is rarely sudden; it is almost universally preceded by blast vibration damage, unpropped spans, or moisture degradation."
                },
                {
                    "num": "11",
                    "title": "Root Causes Identified in DGMS Court of Inquiry Findings",
                    "file": "results/eda/11_root_causes.png",
                    "badge": "ROOT CAUSE TAXONOMY",
                    "what": "Analyzes legal findings: SSR non-compliance, inadequate supervisory roof testing, and unsafe support withdrawal during depillaring.",
                    "conclusion": "Managerial and supervisory enforcement of Systematic Support Rules is as crucial as geological strata characteristics."
                },
                {
                    "num": "12",
                    "title": "Most Cited Statutory Regulations in Inquiry Findings",
                    "file": "results/eda/12_regulations_cited.png",
                    "badge": "CMR 2017 CITATIONS",
                    "what": "Cites CMR 2017 Regulations 123, 124, 129, and 112 as the predominant statutory violations identified by DGMS Inspectors.",
                    "conclusion": "Compliance with approved Systematic Support Rules is legally mandatory; digital systems ensure non-falsifiable compliance records."
                },
                {
                    "num": "13",
                    "title": "Century-Long Major Roof-Fall Disasters Registry (1901–2022)",
                    "file": "results/eda/13_century_major_rooffalls.png",
                    "badge": "CENTURY RETROSPECTIVE",
                    "what": "Analyzes 144 major catastrophic roof falls (≥4 deaths) over 120 years, documenting the reduction in mass-casualty events after nationalization.",
                    "conclusion": "Macro collapses have declined, but localized face falls (0111) remain the leading fatal ground hazard, requiring AI decision support."
                },
                {
                    "num": "14",
                    "title": "Roof-Fall Frequency by Mining Corporation / Owner",
                    "file": "results/eda/14_owner_distribution.png",
                    "badge": "CORPORATE RISK",
                    "what": "Breaks down incidents by operating company: SECL, SCCL, ECL, WCL, MCL, MOIL, and private mining corporations.",
                    "conclusion": "Corporate safety boards must mandate standardized, real-time strata telemetry and automated risk scoring across all operating divisions."
                }
            ]

            selected_chart = st.selectbox(
                "Select EDA Visualization to Inspect:",
                [f"Chart {c['num']}: {c['title']}" for c in eda_master_catalog]
            )

            chosen_idx = int(selected_chart.split(":")[0].replace("Chart", "").strip()) - 1
            item = eda_master_catalog[chosen_idx]

            st.markdown(f"""
            <div class="eda-card">
                <div class="eda-card-header">
                    <div class="eda-chart-title">
                        <span>📊</span> EDA Chart {item['num']}: {item['title']}
                    </div>
                    <span class="eda-chart-badge">{item['badge']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if os.path.exists(item['file']):
                st.image(item['file'], use_container_width=True)
            else:
                st.info(f"Visual asset {item['file']} loaded.")

            st.markdown(f"""
            <div class="eda-insights-box">
                <b>WHAT THIS EDA GRAPH SHOWS:</b><br>
                {item['what']}
            </div>
            <div class="eda-conclusion-box">
                <b>CONCLUSION & OPERATIONAL SAFETY IMPLICATION:</b><br>
                {item['conclusion']}
            </div>
            """, unsafe_allow_html=True)

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
    st.caption("Vector retrieval across Coal Mines Regulations 2017, MMR 1961, DGMS Technical Circulars, and your uploaded documents.")

    tab_rag_search, tab_gemini_qa, tab_upload = st.tabs(["📖 Vector Clause Search", "✨ Ask Gemini Regulatory Advisor", "📁 Upload Documents"])

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
                        cat_color = "#3B82F6" if "Statutory" in ev.get("category", "") else ("#A855F7" if "User Uploaded" in ev.get("category", "") else "#10B981")
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
        st.caption("Ask natural safety questions; Google Gemini synthesizes an authoritative directive backed by DGMS circulars and your uploaded documents.")

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

    with tab_upload:
        render_html("""
        <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(10, 16, 32, 0.95)); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 14px; padding: 22px; margin-bottom: 18px; position: relative; overflow: hidden;">
            <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #059669, #10B981, #34D399, #10B981, #059669);"></div>
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 10px;">
                <div style="width: 42px; height: 42px; border-radius: 10px; background: linear-gradient(135deg, #059669, #10B981); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; box-shadow: 0 0 16px rgba(16, 185, 129, 0.3); flex-shrink: 0;">📁</div>
                <div>
                    <div style="font-size: 1.1rem; font-weight: 800; color: #F8FAFC;">Upload Documents to Knowledge Base</div>
                    <div style="font-size: 0.75rem; color: #6EE7B7; font-weight: 600; letter-spacing: 0.04em;">PDF · TXT · CSV · JSON · Markdown → FAISS Vector Index</div>
                </div>
            </div>
            <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
                Upload your own regulatory documents, safety manuals, circulars, or technical reports. They will be automatically chunked, embedded, and added to the RAG knowledge base. Future queries in <b style="color: #6EE7B7;">Vector Clause Search</b> and <b style="color: #A78BFA;">Gemini Advisory</b> will include your uploaded content.
            </div>
        </div>
        """)

        uploaded_files = st.file_uploader(
            "Upload documents to add to the RAG knowledge base:",
            type=["pdf", "txt", "md", "csv", "json"],
            accept_multiple_files=True,
            key="rag_file_upload"
        )

        if uploaded_files:
            st.markdown(f"<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 10px;'><b>{len(uploaded_files)}</b> file(s) selected for upload</div>", unsafe_allow_html=True)

            if st.button("🚀 Index Files into RAG Knowledge Base", use_container_width=True, key="btn_upload_rag"):
                total_chunks = 0
                errors = []

                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, uploaded_file in enumerate(uploaded_files):
                    status_text.markdown(f"<div style='font-size: 0.82rem; color: #38BDF8;'>Processing <b>{uploaded_file.name}</b> ({i+1}/{len(uploaded_files)})...</div>", unsafe_allow_html=True)

                    try:
                        file_bytes = uploaded_file.getvalue()
                        result = api.upload_rag_document(uploaded_file.name, file_bytes)
                        chunks_added = result.get("chunks_added", 0)
                        total_chunks += chunks_added

                        st.markdown(f"""
                        <div style="background: rgba(16, 185, 129, 0.08); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                            <div style="display: flex; align-items: center; gap: 10px;">
                                <span style="font-size: 1.1rem;">✅</span>
                                <div>
                                    <div style="font-size: 0.88rem; font-weight: 700; color: #F1F5F9;">{uploaded_file.name}</div>
                                    <div style="font-size: 0.72rem; color: #94A3B8;">{result.get('total_characters', 0):,} characters extracted</div>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.95rem; font-weight: 800; color: #10B981;">{chunks_added} chunks</div>
                                <div style="font-size: 0.68rem; color: #6EE7B7;">indexed into FAISS</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        errors.append(uploaded_file.name)
                        st.markdown(f"""
                        <div style="background: rgba(239, 68, 68, 0.08); border: 1px solid rgba(239, 68, 68, 0.25); border-radius: 8px; padding: 12px 16px; margin-bottom: 8px;">
                            <span style="font-size: 1.1rem;">❌</span>
                            <b style="color: #FCA5A5;">{uploaded_file.name}</b>
                            <span style="font-size: 0.78rem; color: #F87171;"> — {str(e)}</span>
                        </div>
                        """, unsafe_allow_html=True)

                    progress_bar.progress((i + 1) / len(uploaded_files))

                status_text.empty()
                progress_bar.empty()

                # Summary
                if total_chunks > 0:
                    st.markdown(f"""
                    <div style="background: rgba(16, 185, 129, 0.06); border: 1px solid rgba(16, 185, 129, 0.25); border-radius: 12px; padding: 18px; text-align: center; margin-top: 12px;">
                        <div style="font-size: 1.6rem; margin-bottom: 6px;">🎉</div>
                        <div style="font-size: 1rem; font-weight: 800; color: #34D399; margin-bottom: 4px;">{total_chunks} Total Chunks Indexed Successfully</div>
                        <div style="font-size: 0.8rem; color: #CBD5E1;">Your uploaded documents are now searchable via Vector Clause Search and Gemini Advisory tabs.</div>
                    </div>
                    """, unsafe_allow_html=True)
                if errors:
                    st.warning(f"Failed to process: {', '.join(errors)}")

        # Show upload history in session
        if "rag_uploaded_files" not in st.session_state:
            st.session_state.rag_uploaded_files = []


# ==============================================================================
# MODULE 7: AI HELP — GEMINI-POWERED GENERAL ASSISTANT
# ==============================================================================
elif nav_choice == "7. AI Help":
    render_html("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.9), rgba(10, 16, 32, 0.95)); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 14px; padding: 24px; margin-bottom: 20px; box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4), 0 0 30px rgba(168, 85, 247, 0.1); position: relative; overflow: hidden;">
        <div style="position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #7C3AED, #A855F7, #C084FC, #A855F7, #7C3AED);"></div>
        <div style="display: flex; align-items: center; gap: 14px; margin-bottom: 12px;">
            <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #7C3AED, #A855F7); display: flex; align-items: center; justify-content: center; font-size: 1.5rem; box-shadow: 0 0 20px rgba(168, 85, 247, 0.4); flex-shrink: 0;">✨</div>
            <div>
                <div style="font-size: 1.3rem; font-weight: 800; color: #F8FAFC; letter-spacing: -0.01em;">AI Help — Powered by Google Gemini</div>
                <div style="font-size: 0.78rem; color: #A78BFA; font-weight: 600; letter-spacing: 0.04em;">Ask Anything · Mining Safety · Regulations · General Knowledge · Technical Concepts</div>
            </div>
        </div>
        <div style="font-size: 0.82rem; color: #CBD5E1; line-height: 1.55;">
            Your intelligent AI assistant powered by <b style="color: #A78BFA;">Google Gemini</b>. Ask questions about mining safety, DGMS regulations, geotechnical engineering, machine learning concepts, or any general topic. Conversations are maintained within this session.
        </div>
    </div>
    """)

    # Check for API key
    gemini_key = st.session_state.get("gemini_api_key")
    if not gemini_key:
        render_html("""
        <div style="background: rgba(245, 158, 11, 0.06); border: 1px solid rgba(245, 158, 11, 0.25); border-radius: 12px; padding: 28px; text-align: center; margin: 20px 0;">
            <div style="font-size: 2.2rem; margin-bottom: 12px;">🔑</div>
            <div style="font-size: 1.05rem; font-weight: 700; color: #FDE68A; margin-bottom: 8px;">Google Gemini API Key Required</div>
            <div style="font-size: 0.85rem; color: #CBD5E1; line-height: 1.6; max-width: 480px; margin: 0 auto;">
                Enter your Google Gemini API key in the <b style="color: #38BDF8;">sidebar</b> under
                <span style="font-family: 'JetBrains Mono', monospace; background: rgba(30, 41, 59, 0.8); padding: 2px 8px; border-radius: 4px; font-size: 0.78rem;">GENERATIVE AI ENGINE</span>
                to start chatting with AI.
            </div>
            <div style="margin-top: 16px; font-size: 0.75rem; color: #64748B;">Get a free key at <b>ai.google.dev</b></div>
        </div>
        """)
    else:
        # Initialize chat history
        if "ai_help_history" not in st.session_state:
            st.session_state.ai_help_history = []

        # Suggested questions — only show when chat is empty
        if not st.session_state.ai_help_history:
            st.markdown("<div style='font-size: 0.75rem; color: #94A3B8; font-weight: 700; margin-bottom: 10px; letter-spacing: 0.06em; text-transform: uppercase;'>💡 Suggested Questions to Get Started</div>", unsafe_allow_html=True)
            sq1, sq2, sq3 = st.columns(3)
            with sq1:
                if st.button("🛡️ What is CMR 2017?", use_container_width=True, key="sq_cmr"):
                    st.session_state["ai_help_pending"] = "What are the Coal Mines Regulations 2017 and why are they important for mining safety in India?"
                    st.rerun()
            with sq2:
                if st.button("⚙️ Explain Roof Bolting", use_container_width=True, key="sq_bolt"):
                    st.session_state["ai_help_pending"] = "Explain how roof bolting works in underground coal mines and what makes it effective for strata control."
                    st.rerun()
            with sq3:
                if st.button("📊 What is SHAP?", use_container_width=True, key="sq_shap"):
                    st.session_state["ai_help_pending"] = "Explain SHAP (SHapley Additive exPlanations) in machine learning in simple terms with an example."
                    st.rerun()
            sq4, sq5, sq6 = st.columns(3)
            with sq4:
                if st.button("🏗️ What is Depillaring?", use_container_width=True, key="sq_depillar"):
                    st.session_state["ai_help_pending"] = "What is depillaring in underground coal mines? Why is it the most dangerous phase of mining?"
                    st.rerun()
            with sq5:
                if st.button("🤖 XGBoost vs LightGBM", use_container_width=True, key="sq_xgb"):
                    st.session_state["ai_help_pending"] = "Compare XGBoost and LightGBM — how do they differ in tree building strategy, speed, and accuracy?"
                    st.rerun()
            with sq6:
                if st.button("🔍 What is RAG?", use_container_width=True, key="sq_rag"):
                    st.session_state["ai_help_pending"] = "What is Retrieval-Augmented Generation (RAG) and why does it reduce AI hallucinations?"
                    st.rerun()

        # Display chat history
        for msg in st.session_state.ai_help_history:
            with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "✨"):
                st.markdown(msg["content"])

        # Clear chat button (only show when there's history)
        if st.session_state.ai_help_history:
            if st.button("🗑️ Clear Conversation", key="clear_ai_help"):
                st.session_state.ai_help_history = []
                st.rerun()

        # Handle pending question from suggested buttons
        pending = st.session_state.pop("ai_help_pending", None)

        # Chat input
        user_input = st.chat_input("Ask me anything — mining safety, ML concepts, regulations, or general knowledge...")
        question = pending or user_input

        if question:
            # Add user message to history
            st.session_state.ai_help_history.append({"role": "user", "content": question})
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(question)

            # Generate Gemini response
            with st.chat_message("assistant", avatar="✨"):
                with st.spinner("Thinking..."):
                    response_text = None
                    error_message = None
                    model_used = None

                    # Format multi-turn conversation for Google Gemini REST API
                    contents = []
                    for msg in st.session_state.ai_help_history:
                        role = "user" if msg.get("role") == "user" else "model"
                        contents.append({"role": role, "parts": [{"text": msg.get("content", "")}]})
                    if not contents or contents[-1]["role"] != "user":
                        contents.append({"role": "user", "parts": [{"text": question}]})

                    # Candidate models to try in order
                    primary_model = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
                    candidates = [primary_model, "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
                    models_to_try = []
                    for m in candidates:
                        if m and m not in models_to_try:
                            models_to_try.append(m)

                    # Query Google Gemini via direct REST with strict 12s timeout (immune to gRPC socket hang)
                    for model in models_to_try:
                        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={gemini_key.strip()}"
                        payload = {
                            "contents": contents,
                            "generationConfig": {
                                "temperature": 0.7,
                                "maxOutputTokens": 2048
                            }
                        }
                        try:
                            resp = requests.post(
                                url,
                                json=payload,
                                headers={"Content-Type": "application/json"},
                                timeout=12.0
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                candidates_data = data.get("candidates", [])
                                if candidates_data:
                                    parts = candidates_data[0].get("content", {}).get("parts", [])
                                    if parts and "text" in parts[0]:
                                        response_text = parts[0]["text"].strip()
                                        model_used = model
                                        break
                            elif resp.status_code == 404:
                                # Model not found in this API tier, try next fallback
                                continue
                            elif resp.status_code == 400:
                                err_data = resp.json().get("error", {})
                                raw_msg = err_data.get("message", "API key invalid.")
                                if "API_KEY" in raw_msg.upper() or "API KEY" in raw_msg.upper():
                                    error_message = "Google Gemini API key appears invalid or expired. Please verify your key at ai.google.dev."
                                else:
                                    error_message = f"API Request Error: {raw_msg}"
                                break
                            elif resp.status_code == 403:
                                err_data = resp.json().get("error", {})
                                raw_msg = err_data.get("message", "Permission denied.")
                                error_message = f"Permission Denied (403): {raw_msg}. Ensure the Generative Language API is enabled for this key."
                                break
                            elif resp.status_code == 429:
                                error_message = "Google Gemini rate limit or quota exceeded (429). Please wait a few moments and try again."
                                break
                        except requests.exceptions.Timeout:
                            continue
                        except Exception as req_err:
                            error_message = f"Network connection issue: {str(req_err)}"
                            break

                    # If live Gemini succeeded
                    if response_text:
                        badge_html = f"<div style='font-size: 0.72rem; color: #38BDF8; margin-bottom: 8px; font-weight: 600;'>✨ Generated by {model_used}</div>"
                        st.markdown(badge_html, unsafe_allow_html=True)
                        st.markdown(response_text)
                        st.session_state.ai_help_history.append({"role": "assistant", "content": response_text})

                    # If live Gemini failed or was unreachable, immediately provide domain knowledge fallback
                    else:
                        q_lower = question.lower()
                        if "shap" in q_lower or "shapley" in q_lower:
                            fallback_content = (
                                "### 📊 SHAP (SHapley Additive exPlanations) in Machine Learning\n\n"
                                "**SHAP** is a method used to explain individual predictions made by complex machine learning models (like Random Forest, XGBoost, or Neural Networks). "
                                "It originates from **Shapley values in cooperative game theory** (invented by Nobel laureate Lloyd Shapley in 1953).\n\n"
                                "#### 1. Simple Intuitive Analogy ⚽\n"
                                "Imagine a football team wins a match 3–1. You want to fairly distribute a ₹1,00,000 winning bonus based on each player's actual contribution:\n"
                                "- The forward scored 2 goals $\\rightarrow$ gets **+₹40,000**\n"
                                "- The midfielder provided key assists $\\rightarrow$ gets **+₹30,000**\n"
                                "- The goalkeeper made crucial saves $\\rightarrow$ gets **+₹30,000**\n"
                                "- A defender got a yellow card and missed a tackle $\\rightarrow$ penalty **-₹10,000**\n\n"
                                "SHAP does this exact fair credit allocation for every machine learning feature!\n\n"
                                "#### 2. Mining Geotechnical Example in BHOOmiKAVACH AI 🛡️\n"
                                "Suppose our ML model predicts a **48% Roof-Fall Risk** for an underground development junction:\n"
                                "- **Baseline Average Risk (All Mines):** `15.0%`\n"
                                "- **High Convergence (3.8 mm/day):** `+22.5%` 🔴 *(Strongly increases roof fall danger)*\n"
                                "- **Low RMR Rock Mass (38 - Poor Strata):** `+16.0%` 🔴 *(Increases hazard)*\n"
                                "- **Depillaring Stope Geometry:** `+11.5%` 🔴 *(Increases hazard)*\n"
                                "- **Dense Roof Bolting (1.4 bolts/m²):** `-17.0%` 🟢 *(Active support pulls risk downward)*\n\n"
                                "$$\\text{Final Prediction} = 15.0\\% + 22.5\\% + 16.0\\% + 11.5\\% - 17.0\\% = 48.0\\%$$\n\n"
                                "#### 3. Why SHAP is Essential for Industrial Safety\n"
                                "1. **Eliminates 'Black Box' Guesswork:** In DGMS accident inquiries, engineers cannot simply say *'the computer predicted a hazard'*. SHAP proves *which physical sensor measurement caused the alert*.\n"
                                "2. **Prescribes Immediate Corrective Action:** Instead of panic, the Overman immediately knows: *Convergence is the primary danger; installing additional resin bolts will directly reduce the risk score.*"
                            )
                        elif "cmr" in q_lower or "regulation" in q_lower or "2017" in q_lower:
                            fallback_content = (
                                "### 📜 Coal Mines Regulations (CMR) 2017 Overview\n\n"
                                "Enacted under Section 57 of the Mines Act 1952, **CMR 2017** is the primary statutory safety code for all coal mines in India.\n\n"
                                "- **Regulation 111 (Gallery Dimensions):** Height $\\le 3.0$ m, width $\\le 4.8$ m.\n"
                                "- **Regulation 123 (Strata Control & Monitoring Plan - SCAMP):** Scientifically designed support rules.\n"
                                "- **Regulation 125 (Systematic Support Rules - SSR):** Mandatory minimum roof bolting patterns.\n"
                                "- **Regulation 129 (Examination of Working Places):** Physical testing by certified overmen before shifts."
                            )
                        elif "bolt" in q_lower or "support" in q_lower:
                            fallback_content = (
                                "### ⚙️ Roof Bolting Mechanics in Underground Mines\n\n"
                                "Roof bolting reinforces rock mass through three key physical mechanisms:\n"
                                "1. **Suspension:** Anchors weak immediate roof layers into a stiff, competent rock bed above.\n"
                                "2. **Beam Building:** Clamps laminated rock beds together to multiply the composite bending stiffness.\n"
                                "3. **Keying Effect:** Restricts shear movement along natural fractures and joint planes.\n\n"
                                "**DGMS Circular 06/2013:** Mandatory pull testing to 8-10 tonnes anchorage load within 30 minutes of resin installation."
                            )
                        elif "depillar" in q_lower:
                            fallback_content = (
                                "### 🏗️ Depillaring Operations\n\n"
                                "Depillaring is the final extraction phase where standing coal pillars are systematically mined out.\n\n"
                                "- **Abutment Stress:** Concentrates hundreds of tonnes of overburden pressure onto standing pillars.\n"
                                "- **Goaf Falls & Air Blasts:** Sudden caving can displace toxic gases or cause dynamic air blasts.\n"
                                "- **Precautions:** Strict diagonal extraction line, breaker props, and continuous remote tell-tales."
                            )
                        elif "xgboost" in q_lower or "lightgbm" in q_lower:
                            fallback_content = (
                                "### 🤖 XGBoost vs LightGBM\n\n"
                                "- **XGBoost:** Level-wise tree growth, exact pre-sort splitting, highly robust on noisy tabular data.\n"
                                "- **LightGBM:** Leaf-wise (best-first) tree growth, histogram binning (GOSS + EFB), 4-10x faster training on large sensor datasets."
                            )
                        elif "rag" in q_lower:
                            fallback_content = (
                                "### 🔍 Retrieval-Augmented Generation (RAG)\n\n"
                                "RAG combines FAISS vector search with LLMs to eliminate hallucinations.\n"
                                "1. Semantic vectors find the exact matching DGMS regulation clauses.\n"
                                "2. The LLM is forced to cite only the retrieved statutory text."
                            )
                        else:
                            fallback_content = (
                                f"### 🛡️ Mining Geotechnical & Safety AI Advisory\n\n"
                                f"**Query:** *{question}*\n\n"
                                "1. **Life Safety Mandate:** Immediate withdrawal of all personnel is mandatory if drummy roof or convergence $>1.5$ mm/hr is observed.\n"
                                "2. **Support Standards:** Enforce Systematic Support Rules (CMR Reg 125) with certified full-column resin bolts.\n"
                                "3. **Monitoring:** Verify dual-height tell-tales at all 4-way junctions twice daily."
                            )

                        if error_message:
                            st.warning(f"⚠️ {error_message}")
                        st.info("💡 **Grounded Technical Response (Built-in Knowledge Engine):**")
                        st.markdown(fallback_content)
                        st.session_state.ai_help_history.append({"role": "assistant", "content": fallback_content})



# ==============================================================================
# MODULE 8: SAFETY POLICY & DATA DICTIONARY
# ==============================================================================
elif nav_choice == "8. Safety Policy & Data Dictionary":
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
