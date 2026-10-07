"""
BHOOmiKAVACH AI — Causal Factor Taxonomy & Regulatory Reference System
=======================================================================
Defines standard taxonomic categories, subcategories, codes, descriptions,
and corresponding DGMS statutory regulation links for roof-fall hazards.

Taxonomy Pillars:
  1. GEOLOGY & STRATA (GEO)
  2. SUPPORT SYSTEM & INTEGRITY (SUP)
  3. MINING METHOD & STRESS CONCENTRATION (METH)
  4. HUMAN FACTOR, SUPERVISION & COMPLIANCE (HUM)
  5. ENVIRONMENTAL & OPERATIONAL CONDITIONS (ENV)

Author: BHOOmiKAVACH AI Engineering Team
"""

from typing import Dict, List, Any

CAUSAL_TAXONOMY: Dict[str, Dict[str, Any]] = {
    # ── 1. GEOLOGY & STRATA (GEO) ──────────────────────────────────────────────
    "GEO_SLIP_PLANE": {
        "pillar": "Geology & Strata",
        "name": "Geological Slip Plane / Fault / Cleat Weakness",
        "description": "Presence of geological discontinuities, slickensides, slip planes, slips along bedding planes, or prominent cleats impairing tensile strength.",
        "keywords": ["slip", "slip plane", "slip-plane", "fault", "discontinuity", "cleat", "joint", "slickenside", "bedding plane"],
        "regulations": ["CMR 2017 Reg 123", "MMR 1961 Reg 112"],
        "severity_weight": 0.90,
    },
    "GEO_SANDSTONE_SEPARATION": {
        "pillar": "Geology & Strata",
        "name": "Sandstone / Shale Bed Separation & Weathering",
        "description": "Laminated shale/sandstone roof prone to parting, bed separation, deterioration under moisture/air slaking or low RMR.",
        "keywords": ["sandstone", "shale", "bed separation", "laminated", "delamination", "parting", "weathering", "slaking"],
        "regulations": ["CMR 2017 Reg 123", "DGMS Tech Circular 3 of 2010"],
        "severity_weight": 0.85,
    },
    "GEO_HIDDEN_CRACK": {
        "pillar": "Geology & Strata",
        "name": "Concealed Cracks / False Roof / Kettle Bottoms",
        "description": "Hidden slips, kettle bottoms, pots, fossil tree trunks, or micro-fissures undetectable by visual inspection alone.",
        "keywords": ["hidden crack", "concealed crack", "false roof", "pot", "kettle bottom", "fissure", "sounding"],
        "regulations": ["CMR 2017 Reg 123(2)", "MMR 1961 Reg 112"],
        "severity_weight": 0.88,
    },

    # ── 2. SUPPORT SYSTEM & INTEGRITY (SUP) ────────────────────────────────────
    "SUP_UNSUPPORTED_ROOF": {
        "pillar": "Support System & Integrity",
        "name": "Deployment Under Unsupported Roof / Inadequate Density",
        "description": "Work persons deployed or moving into freshly exposed or unsupported spans without systematic support rules (SSR) compliance.",
        "keywords": ["unsupported", "freshly exposed", "support density", "lack of support", "inadequate support", "distance from face"],
        "regulations": ["CMR 2017 Reg 123", "CMR 2017 Reg 125", "MMR 1961 Reg 112"],
        "severity_weight": 0.95,
    },
    "SUP_DISLODGED_DAMAGED": {
        "pillar": "Support System & Integrity",
        "name": "Dislodged, Defective, or Blasted-out Supports",
        "description": "Props, roof bolts, or cogs dislodged by blasting flying rock, coal shuttle movement, or substandard resin anchorage.",
        "keywords": ["dislodged", "blasted-out", "displaced support", "bolt failure", "anchorage", "resin grout", "broken prop"],
        "regulations": ["CMR 2017 Reg 125", "DGMS Circular 6 of 2013 (Bolt Testing)"],
        "severity_weight": 0.86,
    },
    "SUP_DRESSING_FAILURE": {
        "pillar": "Support System & Integrity",
        "name": "Failure to Test and Dress Loose Roof / Sides",
        "description": "Supervisors or mining sirdars failed to test, sound, and systematically dress down loose rock before engaging workers.",
        "keywords": ["dressing", "dress loose", "sound the roof", "test and dress", "loose roof", "crowbar", "sounding rod"],
        "regulations": ["CMR 2017 Reg 123(1)", "CMR 2017 Reg 129", "MMR 1961 Reg 112"],
        "severity_weight": 0.92,
    },

    # ── 3. MINING METHOD & STRESS CONCENTRATION (METH) ─────────────────────────
    "METH_DEPILLARING_ABUTMENT": {
        "pillar": "Mining Method & Stress Concentration",
        "name": "Depillaring / Goaf Abutment Front Stress",
        "description": "Secondary extraction, slice/stook extraction causing high front/side abutment stress concentration near goaf edges.",
        "keywords": ["depillaring", "goaf", "abutment", "stook", "slice", "rib", "caving", "front abutment", "goaf edge prop"],
        "regulations": ["CMR 2017 Reg 112", "CMR 2017 Reg 123", "CMR 2017 Reg 126"],
        "severity_weight": 0.94,
    },
    "METH_JUNCTION_SPAN": {
        "pillar": "Mining Method & Stress Concentration",
        "name": "Excessive Junction Diagonal Span / Over-width Gallery",
        "description": "Galleries exceeding statutory dimensions (CMR Reg 111: max width 4.8m, height 3.0m) or 4-way junctions creating large diagonal roof exposure.",
        "keywords": ["junction", "diagonal span", "excessive width", "over-width", "gallery width", "4-way junction", "intersection"],
        "regulations": ["CMR 2017 Reg 111", "CMR 2017 Reg 123"],
        "severity_weight": 0.87,
    },
    "METH_BLASTING_SHOCK": {
        "pillar": "Mining Method & Stress Concentration",
        "name": "Blasting Shock Waves & Strata Vibration",
        "description": "Solid blasting, over-charging, or delay detonators inducing tensile fractures and dynamic vibration in overlying immediate roof.",
        "keywords": ["blasting", "solid blasting", "shock wave", "vibration", "explosives", "detonator", "blasting induced"],
        "regulations": ["CMR 2017 Reg 193", "CMR 2017 Reg 202", "MMR 1961 Reg 160"],
        "severity_weight": 0.82,
    },

    # ── 4. HUMAN FACTOR, SUPERVISION & COMPLIANCE (HUM) ───────────────────────
    "HUM_SUPERVISION_LAPSE": {
        "pillar": "Human Factor, Supervision & Compliance",
        "name": "Inadequate Inspection & Statutory Supervision Lapse",
        "description": "Mining Sirdar, Overman, or Shift In-charge failed to inspect the working place prior to shift start or during change of shifts.",
        "keywords": ["supervision", "sirdar", "overman", "inspection", "statutory check", "supervisory lapse", "unsupervised"],
        "regulations": ["CMR 2017 Reg 129", "CMR 2017 Reg 47", "CMR 2017 Reg 48", "MMR 1961 Reg 43"],
        "severity_weight": 0.91,
    },
    "HUM_UNAUTHORIZED_ENTRY": {
        "pillar": "Human Factor, Supervision & Compliance",
        "name": "Entry into Goaf / Fenced-off / De-energized Danger Zones",
        "description": "Workers entering unventilated, depillared, or fenced-off goaves for coal picking, defecation, or resting.",
        "keywords": ["unauthorized", "danger zone", "fenced off", "goaf entry", "trespass", "unauthorized entry"],
        "regulations": ["CMR 2017 Reg 124", "CMR 2017 Reg 139", "MMR 1961 Reg 115"],
        "severity_weight": 0.89,
    },

    # ── 5. ENVIRONMENTAL & OPERATIONAL CONDITIONS (ENV) ───────────────────────
    "ENV_WATER_SEEPAGE": {
        "pillar": "Environmental & Operational Conditions",
        "name": "Water Seepage, Slaking & Pore Pressure Rise",
        "description": "Inflow of underground water, dripping roof, high humidity accelerating shale swelling, clay decomposition, or grout wash-out.",
        "keywords": ["water seepage", "dripping", "percolation", "slaking", "pore pressure", "water bearing strata", "swelling"],
        "regulations": ["CMR 2017 Reg 149", "DGMS Tech Circular 1999"],
        "severity_weight": 0.80,
    },
    "ENV_SHIFT_FATIGUE": {
        "pillar": "Environmental & Operational Conditions",
        "name": "Night Shift / Extended Hours & Sensory Fatigue",
        "description": "Accidents occurring in late night / early morning hours (22:00 - 06:00) influenced by reduced vigilance, poor illumination, or fatigue.",
        "keywords": ["night shift", "third shift", "fatigue", "illumination", "poor visibility", "lux level"],
        "regulations": ["Mines Rules 1955 Rule 40 (Lighting)", "Mines Act 1952 Sec 30 (Hours of Work)"],
        "severity_weight": 0.74,
    }
}


def classify_accident_text(text: str) -> List[str]:
    """
    Tags accident text with relevant taxonomy codes based on domain keyword matching.
    """
    if not text or text == "NULL":
        return []
    
    matched_codes = []
    text_lower = text.lower()
    
    for code, info in CAUSAL_TAXONOMY.items():
        for kw in info["keywords"]:
            if kw.lower() in text_lower:
                matched_codes.append(code)
                break
                
    return matched_codes


def get_taxonomy_summary() -> Dict[str, Any]:
    """Returns a structural summary of the taxonomy grouped by pillar."""
    pillars: Dict[str, List[Dict[str, Any]]] = {}
    for code, data in CAUSAL_TAXONOMY.items():
        p = data["pillar"]
        if p not in pillars:
            pillars[p] = []
        pillars[p].append({
            "code": code,
            "name": data["name"],
            "regulations": data["regulations"],
            "weight": data["severity_weight"]
        })
    return pillars
