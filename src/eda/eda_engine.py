"""
BHOOmiKAVACH AI — Phase 2: Exploratory Data Analysis (EDA) Engine
==================================================================
Performs comprehensive historical analysis of DGMS roof-fall accidents.

Generates:
  1.  Roof-fall accidents by year (bar + trend line)
  2.  Fatalities by year
  3.  Coal vs Non-Coal breakdown (pie + grouped bar)
  4.  State distribution
  5.  Mine distribution (Top 10)
  6.  Mineral distribution
  7.  Activity at time of accident
  8.  Victim designation
  9.  Age distribution
  10. Shift / time-of-day analysis
  11. Cause & contributing factors
  12. Preventive regulatory findings (most cited regulations)
  13. Major accidents since 1901 — century trend
  14. Fatality magnitude distribution

Outputs: PNG charts → results/eda/
         JSON analytics summary → results/eda/eda_summary.json

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from collections import Counter

warnings.filterwarnings("ignore")

# ── Output Directories ────────────────────────────────────────────────────────
OUT_DIR = "results/eda"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour Palette (BHOOmiKAVACH Brand) ──────────────────────────────────────
CLR_COAL      = "#E65100"   # Deep orange — danger
CLR_NONCOAL   = "#1565C0"   # Deep blue — metalliferous
CLR_ACCENT    = "#D32F2F"   # Red for fatalities
CLR_BG        = "#0D1117"   # Dark background
CLR_FG        = "#E8E8E8"   # Light foreground
CLR_GRID      = "#21262D"   # Grid lines
CLR_HIGHLIGHT = "#F9A825"   # Gold highlight
CLR_GREEN     = "#2E7D32"   # Good / safe
PALETTE_YEAR  = ["#E65100","#D32F2F","#B71C1C","#880E4F","#4A148C","#1A237E","#01579B"]
PALETTE_STATES = ["#E65100","#F57C00","#FBC02D","#388E3C","#0288D1","#7B1FA2","#C2185B","#455A64","#0277BD"]

def apply_dark_style(fig, ax_list):
    """Apply BHOOmiKAVACH dark industrial styling to a figure."""
    fig.patch.set_facecolor(CLR_BG)
    for ax in (ax_list if isinstance(ax_list, list) else [ax_list]):
        ax.set_facecolor(CLR_BG)
        ax.tick_params(colors=CLR_FG, labelsize=9)
        ax.xaxis.label.set_color(CLR_FG)
        ax.yaxis.label.set_color(CLR_FG)
        ax.title.set_color(CLR_FG)
        for spine in ax.spines.values():
            spine.set_edgecolor(CLR_GRID)
        ax.grid(color=CLR_GRID, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def save_chart(fig, name: str):
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=CLR_BG)
    plt.close(fig)
    print(f"  [OK] Saved: {path}")
    return path


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
def load_data():
    df = pd.read_csv("data/processed/roof_fall_accidents.csv")
    df_major = pd.read_csv("data/processed/major_roof_falls_1901_2022.csv")
    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_major["date"] = pd.to_datetime(df_major["date"], errors="coerce")
    df_major["year"] = df_major["date"].dt.year
    # Add shift column from time
    def classify_shift(t):
        if t == "NULL" or pd.isna(t):
            return "Unknown"
        try:
            h = int(str(t).split(":")[0])
            if 6 <= h < 14:
                return "Morning (06-14)"
            elif 14 <= h < 22:
                return "Evening (14-22)"
            else:
                return "Night (22-06)"
        except:
            return "Unknown"
    df["shift"] = df["time"].astype(str).apply(classify_shift)
    return df, df_major


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Roof-Fall Accidents by Year
# ─────────────────────────────────────────────────────────────────────────────
def chart_accidents_by_year(df):
    by_year = df.groupby("year").size().reset_index(name="accidents")
    by_year_coal = df[df["coal_noncoal"]=="Coal"].groupby("year").size()
    by_year_nc   = df[df["coal_noncoal"]=="Non-Coal"].groupby("year").size()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = by_year["year"].values
    w = 0.35
    bars_c  = ax.bar(x - w/2, [by_year_coal.get(y, 0) for y in x], w, label="Coal", color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG)
    bars_nc = ax.bar(x + w/2, [by_year_nc.get(y, 0) for y in x], w, label="Non-Coal", color=CLR_NONCOAL, alpha=0.9, edgecolor=CLR_BG)
    ax.plot(x, by_year["accidents"].values, "o--", color=CLR_HIGHLIGHT, linewidth=1.5, markersize=5, label="Total", zorder=5)

    for bar in list(bars_c) + list(bars_nc):
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.08, str(int(h)), ha="center", va="bottom", fontsize=8, color=CLR_FG)

    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Fatal Roof-Fall Accidents (0111)", fontsize=11)
    ax.set_title("BHOOmiKAVACH AI  |  DGMS Fatal Roof-Fall Accidents (Code 0111) by Year  [2016–2022]", fontsize=12, fontweight="bold", color=CLR_HIGHLIGHT, pad=12)
    ax.set_xticks(x)
    ax.legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=9)
    apply_dark_style(fig, ax)
    return save_chart(fig, "01_accidents_by_year")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Fatalities by Year
# ─────────────────────────────────────────────────────────────────────────────
def chart_fatalities_by_year(df):
    by_year = df.groupby("year")["fatalities"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(by_year["year"], by_year["fatalities"], color=CLR_ACCENT, alpha=0.9, edgecolor=CLR_BG, linewidth=0.5)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.1, str(int(h)), ha="center", va="bottom", fontsize=9, color=CLR_FG, fontweight="bold")
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("Fatalities", fontsize=11)
    ax.set_title("BHOOmiKAVACH AI  |  Roof-Fall Fatalities by Year  [2016–2022]\n(Code 0111 — All Fatalities Represent Loss of Life in DGMS Enquiry Records)", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    ax.set_xticks(by_year["year"])
    ax.set_ylim(0, by_year["fatalities"].max() * 1.2)
    apply_dark_style(fig, ax)
    return save_chart(fig, "02_fatalities_by_year")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Coal vs Non-Coal
# ─────────────────────────────────────────────────────────────────────────────
def chart_coal_vs_noncoal(df):
    fig = plt.figure(figsize=(14, 5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    ax_pie = fig.add_subplot(gs[0, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    fig.patch.set_facecolor(CLR_BG)

    # Pie
    counts = df["coal_noncoal"].value_counts()
    wedges, texts, autotexts = ax_pie.pie(
        counts.values, labels=counts.index, autopct="%1.1f%%",
        colors=[CLR_COAL, CLR_NONCOAL], startangle=90,
        wedgeprops=dict(edgecolor=CLR_BG, linewidth=2),
        textprops=dict(color=CLR_FG, fontsize=10)
    )
    for at in autotexts:
        at.set_fontsize(10); at.set_fontweight("bold"); at.set_color(CLR_FG)
    ax_pie.set_title("Accident Distribution\n(by Mine Category)", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    ax_pie.set_facecolor(CLR_BG)

    # Grouped bar by year (fatalities)
    years = sorted(df["year"].unique())
    coal_fat = [df[(df["year"]==y) & (df["coal_noncoal"]=="Coal")]["fatalities"].sum() for y in years]
    nc_fat   = [df[(df["year"]==y) & (df["coal_noncoal"]=="Non-Coal")]["fatalities"].sum() for y in years]
    x = np.arange(len(years)); w = 0.35
    ax_bar.bar(x - w/2, coal_fat, w, label="Coal Fatalities", color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG)
    ax_bar.bar(x + w/2, nc_fat, w, label="Non-Coal Fatalities", color=CLR_NONCOAL, alpha=0.9, edgecolor=CLR_BG)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels(years, rotation=45)
    ax_bar.set_ylabel("Fatalities", fontsize=10)
    ax_bar.set_title("Fatalities by Year\n(Coal vs Non-Coal)", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    ax_bar.legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=8)
    apply_dark_style(fig, [ax_bar])
    fig.suptitle("BHOOmiKAVACH AI  |  Coal vs Non-Coal Mine Roof-Fall Analysis", fontsize=12, fontweight="bold", color=CLR_FG, y=1.02)
    return save_chart(fig, "03_coal_vs_noncoal")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: State Distribution
# ─────────────────────────────────────────────────────────────────────────────
def chart_state_distribution(df):
    state_acc  = df.groupby("state").size().sort_values(ascending=True)
    state_fat  = df.groupby("state")["fatalities"].sum()
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor(CLR_BG)

    colors_acc = [PALETTE_STATES[i % len(PALETTE_STATES)] for i in range(len(state_acc))]
    state_acc.plot(kind="barh", ax=axes[0], color=colors_acc, edgecolor=CLR_BG, alpha=0.9)
    axes[0].set_xlabel("Number of Accidents", fontsize=10)
    axes[0].set_title("Roof-Fall Accidents by State", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT)
    for i, v in enumerate(state_acc.values):
        axes[0].text(v + 0.05, i, str(v), va="center", fontsize=9, color=CLR_FG)

    state_fat_sorted = state_fat.sort_values(ascending=True)
    colors_fat = [PALETTE_STATES[i % len(PALETTE_STATES)] for i in range(len(state_fat_sorted))]
    state_fat_sorted.plot(kind="barh", ax=axes[1], color=colors_fat, edgecolor=CLR_BG, alpha=0.9)
    axes[1].set_xlabel("Total Fatalities", fontsize=10)
    axes[1].set_title("Fatalities by State", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT)
    for i, v in enumerate(state_fat_sorted.values):
        axes[1].text(v + 0.05, i, str(v), va="center", fontsize=9, color=CLR_FG)

    apply_dark_style(fig, list(axes))
    fig.suptitle("BHOOmiKAVACH AI  |  State-wise Roof-Fall Distribution  [2016–2022]", fontsize=12, fontweight="bold", color=CLR_FG, y=1.02)
    return save_chart(fig, "04_state_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 5: Mine Distribution (Top 10 by accidents)
# ─────────────────────────────────────────────────────────────────────────────
def chart_mine_distribution(df):
    mine_acc = df.groupby("mine").agg(accidents=("mine","count"), fatalities=("fatalities","sum")).sort_values("accidents", ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(mine_acc))
    ax.bar(x - 0.2, mine_acc["accidents"], 0.35, label="Accidents", color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG)
    ax.bar(x + 0.2, mine_acc["fatalities"], 0.35, label="Fatalities", color=CLR_ACCENT, alpha=0.9, edgecolor=CLR_BG)
    ax.set_xticks(x)
    ax.set_xticklabels([m[:22] for m in mine_acc.index], rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Top 10 Mines by Roof-Fall Accidents & Fatalities  [2016–2022]", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    ax.legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=9)
    apply_dark_style(fig, ax)
    return save_chart(fig, "05_mine_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 6: Mineral Distribution
# ─────────────────────────────────────────────────────────────────────────────
def chart_mineral_distribution(df):
    mineral_data = df.groupby("mineral").agg(accidents=("mineral","count"), fatalities=("fatalities","sum")).sort_values("accidents", ascending=False)
    colors = [CLR_COAL, CLR_NONCOAL, "#7B1FA2", "#F9A825", "#2E7D32", "#0288D1", "#C62828"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(CLR_BG)

    axes[0].pie(mineral_data["accidents"], labels=mineral_data.index, autopct="%1.0f%%",
                colors=colors[:len(mineral_data)], startangle=140,
                wedgeprops=dict(edgecolor=CLR_BG, linewidth=2),
                textprops=dict(color=CLR_FG, fontsize=9))
    axes[0].set_title("Accidents by Mineral", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    axes[0].set_facecolor(CLR_BG)

    x = np.arange(len(mineral_data))
    axes[1].bar(x, mineral_data["fatalities"].values, color=colors[:len(mineral_data)], alpha=0.9, edgecolor=CLR_BG)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(mineral_data.index, rotation=35, ha="right", fontsize=9)
    axes[1].set_ylabel("Fatalities", fontsize=10)
    axes[1].set_title("Fatalities by Mineral", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    for i, v in enumerate(mineral_data["fatalities"].values):
        axes[1].text(i, v + 0.1, str(v), ha="center", fontsize=9, color=CLR_FG, fontweight="bold")
    apply_dark_style(fig, list(axes))
    fig.suptitle("BHOOmiKAVACH AI  |  Mineral-wise Roof-Fall Analysis  [2016–2022]", fontsize=12, fontweight="bold", color=CLR_FG, y=1.02)
    return save_chart(fig, "06_mineral_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 7: Victim Designation
# ─────────────────────────────────────────────────────────────────────────────
def chart_designation(df):
    all_desig = []
    for d in df["designation"].dropna():
        if d != "NULL":
            for part in str(d).split(";"):
                part = part.strip()
                if part and part != "NULL":
                    # Normalize common variations
                    part = re.sub(r"(?i)support\s*man|support\s*mistry|supportman", "Support Man/Mistry", part)
                    part = re.sub(r"(?i)timber\s*man|timberman", "Timber Man", part)
                    part = re.sub(r"(?i)gen\.?\s*mazdoor|general\s*mazdoor", "General Mazdoor", part)
                    part = re.sub(r"(?i)bg\s*worker|ug\s*worker|belowground\s*worker", "BG/UG Worker", part)
                    part = re.sub(r"(?i)cat[\s\-]?v|cat\s*v", "Category-V Worker", part)
                    all_desig.append(part.strip())

    counter = Counter(all_desig).most_common(12)
    labels, counts = zip(*counter)

    fig, ax = plt.subplots(figsize=(13, 5))
    colors = [CLR_ACCENT if i < 3 else CLR_COAL if i < 6 else CLR_NONCOAL for i in range(len(labels))]
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=colors, alpha=0.9, edgecolor=CLR_BG)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Fatalities", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Victim Designations in Roof-Fall Fatalities  [2016–2022]\n(Top roles at highest risk)", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, str(cnt), va="center", fontsize=9, color=CLR_FG)
    apply_dark_style(fig, ax)
    return save_chart(fig, "07_victim_designation")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 8: Age Distribution of Victims
# ─────────────────────────────────────────────────────────────────────────────
def chart_age_distribution(df):
    all_ages = []
    for a in df["age"].dropna():
        if str(a) not in ("NULL", "nan", ""):
            for part in str(a).split(";"):
                try:
                    all_ages.append(int(float(part.strip())))
                except:
                    pass
    ages = np.array(all_ages)
    bins = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
    fig, ax = plt.subplots(figsize=(11, 5))
    n, b, patches = ax.hist(ages, bins=bins, color=CLR_COAL, alpha=0.85, edgecolor=CLR_BG, linewidth=0.8)
    for patch, count in zip(patches, n):
        if count > 0:
            ax.text(patch.get_x() + patch.get_width()/2, count + 0.1, str(int(count)), ha="center", va="bottom", fontsize=9, color=CLR_FG)
    ax.axvline(np.mean(ages), color=CLR_HIGHLIGHT, linestyle="--", linewidth=1.5, label=f"Mean Age: {np.mean(ages):.1f} yrs")
    ax.axvline(np.median(ages), color=CLR_ACCENT, linestyle="--", linewidth=1.5, label=f"Median Age: {np.median(ages):.1f} yrs")
    ax.set_xlabel("Age of Victim (Years)", fontsize=10)
    ax.set_ylabel("Number of Fatalities", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Age Distribution of Roof-Fall Victims  [2016–2022]", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    ax.legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=9)
    apply_dark_style(fig, ax)
    return save_chart(fig, "08_age_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 9: Shift / Time-of-Day Analysis
# ─────────────────────────────────────────────────────────────────────────────
def chart_shift_analysis(df):
    shift_data = df.groupby("shift").agg(accidents=("shift","count"), fatalities=("fatalities","sum")).reindex(
        ["Morning (06-14)", "Evening (14-22)", "Night (22-06)", "Unknown"], fill_value=0)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor(CLR_BG)
    shift_colors = [CLR_HIGHLIGHT, CLR_COAL, CLR_NONCOAL, "#555555"]

    axes[0].pie(shift_data["accidents"], labels=shift_data.index, autopct="%1.0f%%",
                colors=shift_colors, startangle=90,
                wedgeprops=dict(edgecolor=CLR_BG, linewidth=2),
                textprops=dict(color=CLR_FG, fontsize=9))
    axes[0].set_title("Accidents by Shift", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    axes[0].set_facecolor(CLR_BG)

    x = np.arange(len(shift_data)); w = 0.35
    axes[1].bar(x - w/2, shift_data["accidents"].values, w, label="Accidents", color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG)
    axes[1].bar(x + w/2, shift_data["fatalities"].values, w, label="Fatalities", color=CLR_ACCENT, alpha=0.9, edgecolor=CLR_BG)
    axes[1].set_xticks(x); axes[1].set_xticklabels([s.split(" ")[0] for s in shift_data.index], fontsize=9)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title("Accidents & Fatalities by Shift", color=CLR_HIGHLIGHT, fontsize=11, fontweight="bold")
    axes[1].legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=9)
    apply_dark_style(fig, list(axes))
    fig.suptitle("BHOOmiKAVACH AI  |  Shift / Time-of-Day Analysis of Roof-Fall Accidents", fontsize=12, fontweight="bold", color=CLR_FG, y=1.02)
    return save_chart(fig, "09_shift_analysis")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 10: Key Contributing Factors
# ─────────────────────────────────────────────────────────────────────────────
def chart_contributing_factors(df):
    all_factors = []
    for row in df["contributing_factors"].dropna():
        if str(row) not in ("NULL", "nan"):
            for f in str(row).split(";"):
                f = f.strip()
                if f:
                    all_factors.append(f)
    counter = Counter(all_factors).most_common(10)
    labels = [l[:55] for l, _ in counter]
    counts = [c for _, c in counter]

    fig, ax = plt.subplots(figsize=(13, 6))
    y_pos = np.arange(len(labels))
    gradient_colors = [f"#{max(0xE6 - i*0x18, 0x44):02X}{0x50:02X}{0x10:02X}" for i in range(len(labels))]
    bars = ax.barh(y_pos, counts, color=gradient_colors, alpha=0.9, edgecolor=CLR_BG)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Frequency in 41 Accident Records", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Top Contributing Factors in Roof-Fall Accidents\n(Based on DGMS Enquiry Findings — NOT Statistical Causation Claims)", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, str(cnt), va="center", fontsize=9, color=CLR_FG)
    apply_dark_style(fig, ax)
    return save_chart(fig, "10_contributing_factors")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 11: Root Causes
# ─────────────────────────────────────────────────────────────────────────────
def chart_root_causes(df):
    all_rc = []
    for row in df["root_cause"].dropna():
        if str(row) not in ("NULL","nan"):
            for rc in str(row).split(";"):
                rc = rc.strip()
                if rc:
                    all_rc.append(rc[:60])
    counter = Counter(all_rc).most_common(8)
    if not counter:
        return None
    labels, counts = zip(*counter)
    fig, ax = plt.subplots(figsize=(13, 5))
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, counts, color=CLR_ACCENT, alpha=0.85, edgecolor=CLR_BG)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Frequency", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Root Causes Identified in DGMS Enquiry Findings", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2, str(cnt), va="center", fontsize=9, color=CLR_FG)
    apply_dark_style(fig, ax)
    return save_chart(fig, "11_root_causes")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 12: Most Cited Regulations
# ─────────────────────────────────────────────────────────────────────────────
def chart_regulations(df):
    all_regs = []
    for row in df["regulation_reference"].dropna():
        if str(row) not in ("NULL","nan"):
            for reg in str(row).split(";"):
                reg = reg.strip()
                # Shorten
                m = re.search(r"Regulation\s+([0-9A-Za-z\(\)]+)", reg, re.I)
                if m:
                    all_regs.append(f"Reg. {m.group(1)}")
    counter = Counter(all_regs).most_common(10)
    if not counter:
        return None
    labels, counts = zip(*counter)
    fig, ax = plt.subplots(figsize=(12, 5))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color=CLR_NONCOAL, alpha=0.85, edgecolor=CLR_BG)
    ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Times Cited in Enquiry Findings", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Most Cited Statutory Regulations in DGMS Roof-Fall Enquiries", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    for i, (cnt, label) in enumerate(zip(counts, labels)):
        ax.text(cnt + 0.05, i, str(cnt), va="center", fontsize=9, color=CLR_FG)
    apply_dark_style(fig, ax)
    return save_chart(fig, "12_regulations_cited")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 13: Century-Long Major Roof Falls (1901-2022)
# ─────────────────────────────────────────────────────────────────────────────
def chart_century_trend(df_major):
    df_valid = df_major.dropna(subset=["year"]).copy()
    df_valid["decade"] = (df_valid["year"] // 10 * 10).astype(int)
    decade_data = df_valid.groupby("decade").agg(events=("decade","count"), fatalities=("fatalities","sum")).reset_index()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    fig.patch.set_facecolor(CLR_BG)

    # Decade events
    axes[0].bar(decade_data["decade"].astype(str), decade_data["events"], color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG, width=0.6)
    axes[0].set_ylabel("Major Roof-Fall Events (≥4 deaths)", fontsize=10)
    axes[0].set_title("Century-Long Major Coal Mine Roof-Fall Events by Decade  [1901–2022]", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT)

    # Decade fatalities
    axes[1].bar(decade_data["decade"].astype(str), decade_data["fatalities"], color=CLR_ACCENT, alpha=0.9, edgecolor=CLR_BG, width=0.6)
    axes[1].set_ylabel("Total Fatalities", fontsize=10)
    axes[1].set_xlabel("Decade", fontsize=10)
    axes[1].set_title("Century-Long Fatalities in Major Coal Mine Roof-Falls  [1901–2022]", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT)

    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
        apply_dark_style(fig, ax)

    fig.suptitle("BHOOmiKAVACH AI  |  DGMS Chapter 10: Major Roof-Fall Disasters Since 1901\n(Source: SANKET Coal Mine Accidents, Chapter 10)", fontsize=12, fontweight="bold", color=CLR_FG, y=1.02)
    fig.tight_layout()
    return save_chart(fig, "13_century_major_rooffalls")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 14: Owner / Company Distribution
# ─────────────────────────────────────────────────────────────────────────────
def chart_owner_distribution(df):
    def shorten_owner(o):
        if "South Eastern Coalfields" in o or "SECL" in o: return "SECL"
        if "Singareni" in o or "SCCL" in o: return "SCCL"
        if "Eastern Coalfields" in o or "ECL" in o: return "ECL"
        if "Western Coalfields" in o or "WCL" in o: return "WCL"
        if "Mahanadi Coalfields" in o or "MCL" in o: return "MCL"
        if "Manganese Ore" in o or "MOIL" in o: return "MOIL"
        if "Hindustan Zinc" in o or "HZL" in o: return "Hindustan Zinc Ltd."
        if "HUTTI GOLD" in o.upper(): return "Hutti Gold Mines"
        if "HINDUSTAN COPPER" in o.upper(): return "Hindustan Copper Ltd."
        if "URANIUM" in o.upper() or "UCIL" in o.upper(): return "UCIL"
        if "J. K. MINERALS" in o.upper(): return "J.K. Minerals"
        if "RCCPL" in o.upper(): return "RCCPL Pvt. Ltd."
        return o[:20]

    df["owner_short"] = df["owner"].apply(shorten_owner)
    owner_data = df.groupby("owner_short").agg(accidents=("owner_short","count"), fatalities=("fatalities","sum")).sort_values("accidents", ascending=False)
    colors = [CLR_COAL if i % 2 == 0 else CLR_NONCOAL for i in range(len(owner_data))]

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(owner_data)); w = 0.35
    ax.bar(x - w/2, owner_data["accidents"].values, w, label="Accidents", color=CLR_COAL, alpha=0.9, edgecolor=CLR_BG)
    ax.bar(x + w/2, owner_data["fatalities"].values, w, label="Fatalities", color=CLR_ACCENT, alpha=0.9, edgecolor=CLR_BG)
    ax.set_xticks(x)
    ax.set_xticklabels(owner_data.index, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("BHOOmiKAVACH AI  |  Roof-Fall Accidents & Fatalities by Mining Company  [2016–2022]", fontsize=11, fontweight="bold", color=CLR_HIGHLIGHT, pad=10)
    ax.legend(facecolor=CLR_BG, edgecolor=CLR_GRID, labelcolor=CLR_FG, fontsize=9)
    apply_dark_style(fig, ax)
    return save_chart(fig, "14_owner_distribution")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EDA RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_eda():
    print("\n" + "="*70)
    print("  BHOOmiKAVACH AI — PHASE 2: EXPLORATORY DATA ANALYSIS (EDA)")
    print("="*70)

    df, df_major = load_data()
    print(f"\nLoaded {len(df)} primary accident records and {len(df_major)} major accident records.\n")

    chart_paths = {}
    print("Generating charts...")

    chart_paths["01_accidents_by_year"]   = chart_accidents_by_year(df)
    chart_paths["02_fatalities_by_year"]  = chart_fatalities_by_year(df)
    chart_paths["03_coal_vs_noncoal"]     = chart_coal_vs_noncoal(df)
    chart_paths["04_state_distribution"]  = chart_state_distribution(df)
    chart_paths["05_mine_distribution"]   = chart_mine_distribution(df)
    chart_paths["06_mineral_distribution"]= chart_mineral_distribution(df)
    chart_paths["07_victim_designation"]  = chart_designation(df)
    chart_paths["08_age_distribution"]    = chart_age_distribution(df)
    chart_paths["09_shift_analysis"]      = chart_shift_analysis(df)
    chart_paths["10_contributing_factors"]= chart_contributing_factors(df)
    chart_paths["11_root_causes"]         = chart_root_causes(df)
    chart_paths["12_regulations_cited"]   = chart_regulations(df)
    chart_paths["13_century_major"]       = chart_century_trend(df_major)
    chart_paths["14_owner_distribution"]  = chart_owner_distribution(df)

    # ── Analytics Summary ─────────────────────────────────────────────────────
    all_ages = []
    for a in df["age"].dropna():
        for part in str(a).split(";"):
            try: all_ages.append(int(float(part.strip())))
            except: pass

    summary = {
        "total_accidents":              int(len(df)),
        "coal_accidents":               int((df["coal_noncoal"]=="Coal").sum()),
        "noncoal_accidents":            int((df["coal_noncoal"]=="Non-Coal").sum()),
        "total_fatalities":             int(df["fatalities"].sum()),
        "total_serious_injuries":       int(df["serious_injuries"].sum()),
        "years_covered":                sorted(df["year"].unique().tolist()),
        "unique_states":                sorted(df["state"].unique().tolist()),
        "unique_districts":             int(df["district"].nunique()),
        "unique_mines":                 int(df["mine"].nunique()),
        "unique_minerals":              sorted(df["mineral"].unique().tolist()),
        "unique_owners":                int(df["owner"].nunique()),
        "mean_fatalities_per_accident": round(float(df["fatalities"].mean()), 2),
        "max_fatalities_single_event":  int(df["fatalities"].max()),
        "mean_victim_age":              round(float(np.mean(all_ages)), 1) if all_ages else None,
        "median_victim_age":            float(np.median(all_ages)) if all_ages else None,
        "night_shift_accidents":        int((df["shift"]=="Night (22-06)").sum()),
        "morning_shift_accidents":      int((df["shift"]=="Morning (06-14)").sum()),
        "evening_shift_accidents":      int((df["shift"]=="Evening (14-22)").sum()),
        "peak_year_accidents":          int(df.groupby("year").size().idxmax()),
        "peak_year_fatalities":         int(df.groupby("year")["fatalities"].sum().idxmax()),
        "major_accidents_1901_2022":    int(len(df_major)),
        "major_fatalities_1901_2022":   int(df_major["fatalities"].sum()),
        "charts_generated":             len([p for p in chart_paths.values() if p])
    }

    summary_path = os.path.join(OUT_DIR, "eda_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n  [OK] Analytics summary saved: {summary_path}")
    print("\n" + "="*70)
    print("  EDA COMPLETE -- Key Findings")
    print("="*70)
    print(f"  Total Accidents (2016-2022) : {summary['total_accidents']}")
    print(f"  Coal / Non-Coal             : {summary['coal_accidents']} / {summary['noncoal_accidents']}")
    print(f"  Total Fatalities            : {summary['total_fatalities']}")
    print(f"  Mean Victim Age             : {summary['mean_victim_age']} yrs")
    print(f"  Peak Accident Year          : {summary['peak_year_accidents']}")
    print(f"  Peak Fatality Year          : {summary['peak_year_fatalities']}")
    print(f"  Night Shift Accidents       : {summary['night_shift_accidents']}")
    print(f"  Charts Generated            : {summary['charts_generated']}")
    print("="*70 + "\n")
    return summary


if __name__ == "__main__":
    run_eda()
