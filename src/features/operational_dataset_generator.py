"""
BHOOmiKAVACH AI — Operational Prospective Risk Dataset Generator (Dataset B)
=============================================================================
CRITICAL DOMAIN NOTICE:
  "SYNTHETIC DATA — NOT REAL MINE OBSERVATIONS"
  This synthetic operational dataset is generated for machine learning model training
  and prospective decision-support validation. It is strictly segregated from
  Dataset A (Historical DGMS Fatal Accident Intelligence).

Domain Grounding & Geotechnical Mechanics Assumptions:
  1. Rock Mass Rating (RMR) follows CMRI-ISM empirical distributions for Indian coal & metal mines.
  2. Mining geometry conforms to CMR 2017 Regulation 111 (max gallery width 4.8m, height 3.0m).
  3. Increased depth (>300m) and depillaring/pillar extraction induce concentrated abutment stresses.
  4. Discontinuities (slip planes, low joint spacing, water seepage) reduce rock mass shear strength.
  5. Delayed or inadequate support installation (density < 1.0 bolt/m2, capacity < 8 tonnes)
     drastically escalates roof delamination and fall probability.
  6. Roof convergence rate (>5 mm/day) and bed separation (>15 mm) serve as acute dynamic precursor warnings.

Target Variable:
  `roof_fall_risk`: ["LOW", "MODERATE", "HIGH", "CRITICAL"]

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any

OUT_PATH = "data/operational/operational_risk_dataset.csv"
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

MINE_PROFILES = [
    {"name": "Moonidih Colliery", "owner": "BCCL", "state": "Jharkhand", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 450, "rmr_mean": 48, "method": "Bord and Pillar (Depillaring)"},
    {"name": "GDK 11A Incline", "owner": "SCCL", "state": "Telangana", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 280, "rmr_mean": 54, "method": "Continuous Miner"},
    {"name": "Shantikhani Mine", "owner": "SCCL", "state": "Telangana", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 310, "rmr_mean": 52, "method": "Longwall Retracing"},
    {"name": "Jhilimili Colliery", "owner": "SECL", "state": "Chhattisgarh", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 180, "rmr_mean": 42, "method": "Bord and Pillar (Development)"},
    {"name": "Churcha RO Mine", "owner": "SECL", "state": "Chhattisgarh", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 230, "rmr_mean": 38, "method": "Bord and Pillar (Depillaring)"},
    {"name": "Duman Hill Colliery", "owner": "SECL", "state": "Chhattisgarh", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 160, "rmr_mean": 45, "method": "Bord and Pillar (Development)"},
    {"name": "Gondegaon Underground", "owner": "WCL", "state": "Maharashtra", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 210, "rmr_mean": 44, "method": "Bord and Pillar (Development)"},
    {"name": "Sonepur Bazari UG", "owner": "ECL", "state": "West Bengal", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 190, "rmr_mean": 40, "method": "Bord and Pillar (Depillaring)"},
    {"name": "Nandira Colliery", "owner": "MCL", "state": "Orissa", "coal_noncoal": "Coal", "mineral": "Coal", "depth_mean": 150, "rmr_mean": 46, "method": "Bord and Pillar (Development)"},
    {"name": "Kolihan Copper Mine", "owner": "HCL", "state": "Rajasthan", "coal_noncoal": "Non-Coal", "mineral": "Copper", "depth_mean": 380, "rmr_mean": 62, "method": "Sublevel Open Stoping"},
    {"name": "Balaghat Mine", "owner": "MOIL", "state": "Madhya Pradesh", "coal_noncoal": "Non-Coal", "mineral": "Manganese", "depth_mean": 320, "rmr_mean": 55, "method": "Cut and Fill Stoping"},
    {"name": "Sindesar Khurd Mine", "owner": "HZL", "state": "Rajasthan", "coal_noncoal": "Non-Coal", "mineral": "Galena & Sphalerite", "depth_mean": 420, "rmr_mean": 65, "method": "Blast Hole Stoping"},
    {"name": "Hutti Gold Mine", "owner": "HGML", "state": "Karnataka", "coal_noncoal": "Non-Coal", "mineral": "Gold", "depth_mean": 580, "rmr_mean": 58, "method": "Sublevel Stoping"},
    {"name": "Narwapahar Mine", "owner": "UCIL", "state": "Jharkhand", "coal_noncoal": "Non-Coal", "mineral": "Uranium", "depth_mean": 290, "rmr_mean": 50, "method": "Cut and Fill Stoping"},
]

ROOF_LITHOLOGIES = ["Sandstone", "Shale", "Carbonaceous Shale", "Sandstone-Shale Intercalation", "Quartzite", "Granite-Gneiss"]
FLOOR_LITHOLOGIES = ["Shale", "Sandstone", "Carbonaceous Clay", "Massive Sandstone", "Mica Schist"]
JOINT_CONDITIONS = ["Tight/Rough", "Slightly Weathered", "Open/Slickensided", "Gouged/Clay Filled"]
WATER_CONDITIONS = ["Completely Dry", "Damp", "Wet/Dripping", "Flowing Inflow"]
SUPPORT_CONDITIONS = ["Intact & Adequate", "Minor Deformation", "Excessive Load/Yielding", "Dislodged/Missing"]


def generate_operational_records(n_samples: int = 1500, random_seed: int = 42) -> pd.DataFrame:
    np.random.seed(random_seed)
    records = []

    # Temporal sequencing across 2023-2024 to support clean time-based splitting
    base_date = datetime.date(2023, 1, 1)

    for i in range(n_samples):
        # 1-2 days advance per batch
        current_date = base_date + datetime.timedelta(days=int(i // 3))
        profile = np.random.choice(MINE_PROFILES)
        
        # Operational context
        shift = np.random.choice(["Shift 1 (06:00-14:00)", "Shift 2 (14:00-22:00)", "Shift 3 (22:00-06:00)"], p=[0.42, 0.38, 0.20])
        panel = f"Panel-{np.random.choice(['A1', 'B2', 'C3', 'D4', 'E5', 'S1', 'N2', 'W3'])}"
        mining_method = profile["method"]
        
        # Geotechnical & Strata
        depth = float(np.clip(np.random.normal(profile["depth_mean"], 50), 60, 750))
        working_height = float(np.clip(np.random.normal(2.6, 0.35), 1.8, 4.5))
        gallery_width = float(np.clip(np.random.normal(4.2, 0.45), 3.0, 5.8))
        gallery_height = working_height
        is_junction = int(np.random.choice([0, 1], p=[0.65, 0.35]))
        face_position = np.random.choice(["Development Face", "Junction", "Depillaring Edge / Goaf", "Slice Gallery", "Main Haulage"])
        if is_junction and face_position != "Depillaring Edge / Goaf":
            face_position = "Junction"

        roof_lithology = np.random.choice(ROOF_LITHOLOGIES)
        floor_lithology = np.random.choice(FLOOR_LITHOLOGIES)
        
        # Rock mass ratings
        rmr = float(np.clip(np.random.normal(profile["rmr_mean"], 10), 18, 85))
        # Q-system and GSI correlated with RMR
        q_value = float(np.clip(10 ** ((rmr - 44) / 9.0), 0.05, 45.0))
        gsi = float(np.clip(rmr - 5 + np.random.normal(0, 3), 15, 80))
        rqd = float(np.clip(rmr * 1.15 + np.random.normal(0, 5), 15, 98))
        rock_strength_mpa = float(np.clip((rmr * 0.9) + np.random.normal(10, 8), 12, 110))
        
        # Discontinuities
        joint_spacing_cm = float(np.clip(np.random.exponential(35), 5, 130))
        joint_orientation = np.random.choice(["Favorable (Perpendicular to Drive)", "Fair", "Unfavorable (Parallel to Drive)"], p=[0.35, 0.45, 0.20])
        joint_condition = np.random.choice(JOINT_CONDITIONS, p=[0.40, 0.35, 0.18, 0.07])
        water_condition = np.random.choice(WATER_CONDITIONS, p=[0.50, 0.30, 0.15, 0.05])
        water_seepage_lpm = float(np.clip(np.random.exponential(3.5 if water_condition != "Completely Dry" else 0.1), 0.0, 40.0))
        
        # Support System
        support_density = float(np.clip(np.random.normal(1.2, 0.35), 0.4, 2.4)) # bolts/m2
        bolt_spacing_m = float(np.clip(1.0 / np.sqrt(max(0.4, support_density)), 0.8, 1.8))
        bolt_length_m = float(np.clip(np.random.choice([1.5, 1.8, 2.2, 2.4]), 1.2, 2.5))
        bolt_capacity_tonnes = float(np.clip(np.random.normal(8.5, 2.2), 2.5, 15.0))
        cable_bolts_installed = int(np.random.choice([1, 0], p=[0.25, 0.75]))
        props_installed = int(np.random.choice([1, 0], p=[0.55, 0.45]))
        support_condition = np.random.choice(SUPPORT_CONDITIONS, p=[0.60, 0.25, 0.10, 0.05])

        # Blasting & Operations
        recent_blasting = np.random.choice(["No Blasting (<24h)", "Recent (4-12h)", "Immediate (<4h)"], p=[0.45, 0.35, 0.20])
        blast_distance_m = float(np.clip(np.random.uniform(5, 80), 5, 100))
        blast_vibration_ppv = float(np.clip(
            np.random.exponential(6.0) * (2.2 if recent_blasting == "Immediate (<4h)" else 0.8),
            0.5, 35.0
        ))
        production_tpd = float(np.clip(np.random.normal(650, 180), 100, 1400))
        machine_activity = np.random.choice(["Continuous Miner Cutting", "LHD / Shuttle Car Loading", "Drilling & Blasting", "Roof Bolting Rig", "Idle / Maintenance"])
        persons_exposed = int(np.random.choice([2, 4, 6, 8, 12, 15], p=[0.25, 0.35, 0.20, 0.12, 0.05, 0.03]))
        temperature_c = float(np.clip(26.0 + (depth / 100.0) * 1.5 + np.random.normal(0, 1.5), 22.0, 39.0))
        humidity_pct = float(np.clip(np.random.normal(82, 8), 55, 99))
        
        previous_roof_fall = int(np.random.choice([0, 1], p=[0.82, 0.18]))
        recent_geological_disturbance = int(np.random.choice([0, 1], p=[0.78, 0.22]))
        
        # Dynamic Strata Convergence Precursors
        depillaring_mult = 1.9 if "Depillaring" in mining_method or face_position == "Depillaring Edge / Goaf" else 1.0
        roof_convergence_mm_day = float(np.clip(
            (depth / 280.0) * (55.0 / rmr) * depillaring_mult * (1.6 if support_density < 1.0 else 0.9) * np.random.lognormal(0.0, 0.35),
            0.1, 30.0
        ))
        roof_to_floor_convergence_mm_day = float(roof_convergence_mm_day * 1.45 + np.random.normal(0, 0.2))
        bed_separation_mm = float(np.clip(
            (roof_convergence_mm_day * 1.3) + (water_seepage_lpm * 0.18) + (6.0 if recent_geological_disturbance else 0.0),
            0.0, 48.0
        ))

        # ── DOMAIN RISK SCORING (Mechanics-grounded) ──────────────────────────
        score = 0.0
        # 1. RMR Factor (CMRI-ISM rating: lower = worse)
        score += max(0, (62.0 - rmr) * 0.55)
        # 2. Depth overburden stress
        score += (depth / 100.0) * 1.4
        # 3. Geometry span penalty (CMR 111 statutory max 4.8m)
        if gallery_width > 4.8:
            score += (gallery_width - 4.8) * 8.5
        if is_junction:
            score += 3.8
        # 4. Secondary extraction abutment
        if "Depillaring" in mining_method or face_position == "Depillaring Edge / Goaf":
            score += 5.0
        # 5. Geotechnical defects
        if joint_condition in ["Open/Slickensided", "Gouged/Clay Filled"]:
            score += 5.5
        if joint_spacing_cm < 20.0:
            score += 3.5
        if recent_geological_disturbance:
            score += 4.5
        score += min(water_seepage_lpm * 0.32, 6.0)
        # 6. Blasting dynamic shock
        if recent_blasting == "Immediate (<4h)" and blast_vibration_ppv > 12.0:
            score += 4.5
        # 7. Support inadequacy
        if support_density < 1.0:
            score += (1.0 - support_density) * 9.0
        if bolt_capacity_tonnes < 7.0:
            score += (7.0 - bolt_capacity_tonnes) * 1.8
        if support_condition in ["Excessive Load/Yielding", "Dislodged/Missing"]:
            score += 6.0
        # 8. Dynamic Convergence
        if roof_convergence_mm_day > 4.5:
            score += min((roof_convergence_mm_day - 4.5) * 0.7, 7.5)
        if bed_separation_mm > 12.0:
            score += min((bed_separation_mm - 12.0) * 0.55, 7.5)
        if previous_roof_fall:
            score += 3.0

        # Target Risk Categorization: [LOW, MODERATE, HIGH, CRITICAL]
        if score < 16.0:
            risk_level = "LOW"
            risk_code = 0
        elif score < 26.5:
            risk_level = "MODERATE"
            risk_code = 1
        elif score < 39.0:
            risk_level = "HIGH"
            risk_code = 2
        else:
            risk_level = "CRITICAL"
            risk_code = 3

        records.append({
            "record_id": f"OP-INSP-{i+1:05d}",
            "date": current_date.isoformat(),
            "shift": shift,
            "mine": profile["name"],
            "owner": profile["owner"],
            "state": profile["state"],
            "coal_noncoal": profile["coal_noncoal"],
            "mineral": profile["mineral"],
            "panel": panel,
            "mining_method": mining_method,
            "depth_m": round(depth, 1),
            "working_height_m": round(working_height, 2),
            "gallery_width_m": round(gallery_width, 2),
            "gallery_height_m": round(gallery_height, 2),
            "is_junction": is_junction,
            "face_position": face_position,
            "roof_lithology": roof_lithology,
            "floor_lithology": floor_lithology,
            "rmr_rating": round(rmr, 1),
            "q_value": round(q_value, 2),
            "gsi": round(gsi, 1),
            "rqd": round(rqd, 1),
            "rock_strength_mpa": round(rock_strength_mpa, 1),
            "joint_spacing_cm": round(joint_spacing_cm, 1),
            "joint_orientation": joint_orientation,
            "joint_condition": joint_condition,
            "water_condition": water_condition,
            "water_seepage_lpm": round(water_seepage_lpm, 2),
            "roof_convergence_mm_day": round(roof_convergence_mm_day, 2),
            "roof_to_floor_convergence_mm_day": round(roof_to_floor_convergence_mm_day, 2),
            "bed_separation_mm": round(bed_separation_mm, 2),
            "support_density_bolts_per_m2": round(support_density, 2),
            "bolt_spacing_m": round(bolt_spacing_m, 2),
            "bolt_length_m": round(bolt_length_m, 2),
            "bolt_capacity_tonnes": round(bolt_capacity_tonnes, 2),
            "cable_bolts_installed": cable_bolts_installed,
            "props_installed": props_installed,
            "support_condition": support_condition,
            "recent_blasting": recent_blasting,
            "blast_distance_m": round(blast_distance_m, 1),
            "blast_vibration_ppv": round(blast_vibration_ppv, 2),
            "production_tpd": round(production_tpd, 1),
            "machine_activity": machine_activity,
            "persons_exposed": persons_exposed,
            "temperature_c": round(temperature_c, 1),
            "humidity_pct": round(humidity_pct, 1),
            "previous_roof_fall": previous_roof_fall,
            "recent_geological_disturbance": recent_geological_disturbance,
            "raw_hazard_score": round(score, 2),
            "roof_fall_risk": risk_level,
            "risk_code": risk_code,
            "data_label": "SYNTHETIC DATA — NOT REAL MINE OBSERVATIONS"
        })

    df = pd.DataFrame(records)
    df.to_csv(OUT_PATH, index=False)
    print(f"Generated {len(df)} operational inspection records -> {OUT_PATH}")
    print("\nRisk Level Distribution:")
    print(df["roof_fall_risk"].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))
    return df


if __name__ == "__main__":
    generate_operational_records()
