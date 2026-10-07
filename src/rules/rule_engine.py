"""
BHOOmiKAVACH AI — Independent Geotechnical Safety Rule Engine (Phase 19)
========================================================================
Implements deterministic, auditable safety rules based on:
  - Coal Mines Regulations, 2017 (CMR 2017)
  - Metalliferous Mines Regulations, 1961 (MMR 1961)
  - DGMS Technical Circulars (Circular 3 of 2010, Circular 6 of 2013)
  - CMRI-ISM Strata Mechanics Criteria

Notice:
  Operates independently of machine learning model predictions.
  If critical telemetry or inspection parameters are missing, returns:
  "INSUFFICIENT DATA — HUMAN/COMPETENT TECHNICAL ASSESSMENT REQUIRED."

Author: BHOOmiKAVACH AI Engineering Team
"""

from typing import Dict, Any, List


class GeotechnicalSafetyRuleEngine:
    def __init__(self):
        self.critical_fields = ["depth_m", "gallery_width_m", "rmr_rating", "support_density_bolts_per_m2"]

    def evaluate_rules(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates deterministic statutory and geotechnical safety rules.
        """
        # 1. Critical Data Sufficiency Check
        missing_critical = [f for f in self.critical_fields if f not in observation or observation[f] is None]
        if missing_critical:
            return {
                "status": "INSUFFICIENT_DATA",
                "triggered_rules": [],
                "rule_severity": "INSUFFICIENT_DATA",
                "recommended_action": "INSUFFICIENT DATA — HUMAN/COMPETENT TECHNICAL ASSESSMENT REQUIRED.",
                "missing_fields": missing_critical
            }

        depth = float(observation.get("depth_m", 0))
        width = float(observation.get("gallery_width_m", 0))
        rmr = float(observation.get("rmr_rating", 50))
        density = float(observation.get("support_density_bolts_per_m2", 1.2))
        capacity = float(observation.get("bolt_capacity_tonnes", 8.5))
        convergence = float(observation.get("roof_convergence_mm_day", 0))
        bed_sep = float(observation.get("bed_separation_mm", 0))
        support_cond = str(observation.get("support_condition", "Intact & Adequate"))
        blasting = str(observation.get("recent_blasting", "No Blasting (<24h)"))
        vibration = float(observation.get("blast_vibration_ppv", 0))
        is_junction = int(observation.get("is_junction", 0))
        mining_method = str(observation.get("mining_method", ""))

        triggered_rules: List[Dict[str, Any]] = []

        # RULE 1: CMR 2017 Regulation 111 — Statutory Dimension Over-span
        if width > 4.8:
            triggered_rules.append({
                "rule_id": "RULE_CMR_111_SPAN",
                "severity": "HIGH",
                "regulation": "CMR 2017 Regulation 111",
                "title": "Statutory Gallery Width Exceeded",
                "finding": f"Gallery width ({width:.2f}m) exceeds statutory ceiling of 4.80m.",
                "action": "Immediate setting of auxiliary middle props/chocks; halt widening until DGMS permission/re-support."
            })

        # RULE 2: CMR 2017 Regulation 125 — Systematic Support Rules (SSR) Inadequacy
        if density < 1.0:
            triggered_rules.append({
                "rule_id": "RULE_CMR_125_DENSITY",
                "severity": "HIGH",
                "regulation": "CMR 2017 Regulation 125 (SSR)",
                "title": "Substandard Roof Bolt Density",
                "finding": f"Installed support density ({density:.2f} bolts/m2) is below standard minimum 1.0-1.2 bolts/m2.",
                "action": "Immediately suspend extraction; install supplemental roof bolts to satisfy approved SSR pattern."
            })

        # RULE 3: DGMS Technical Circular 6 of 2013 — Anchorage Load Bearing
        if capacity < 6.0:
            triggered_rules.append({
                "rule_id": "RULE_DGMS_BOLT_ANCHORAGE",
                "severity": "HIGH",
                "regulation": "DGMS Tech Circular 6 of 2013",
                "title": "Compromised Roof Bolt Anchorage",
                "finding": f"Bolt anchorage load ({capacity:.1f} tonnes) is below 8-10 tonne statutory requirement.",
                "action": "Conduct immediate pull-out testing; augment with resin capsule re-anchorage or cable bolts."
            })

        # RULE 4: Acute Strata Instability — Convergence & Support Distress
        if convergence > 5.0 and support_cond in ["Excessive Load/Yielding", "Dislodged/Missing"]:
            triggered_rules.append({
                "rule_id": "RULE_CRITICAL_CONVERGENCE_FAILURE",
                "severity": "CRITICAL",
                "regulation": "CMR 2017 Regulation 123",
                "title": "Acute Strata Dilation with Active Support Failure",
                "finding": f"High convergence rate ({convergence:.1f} mm/day) combined with damaged/dislodged support.",
                "action": "CRITICAL: Immediately withdraw all personnel to safety berm; bar access and alert Manager/Overman."
            })

        # RULE 5: Bed Separation Indicator
        if bed_sep > 15.0:
            triggered_rules.append({
                "rule_id": "RULE_STRATA_BED_SEPARATION",
                "severity": "CRITICAL" if bed_sep > 25.0 else "HIGH",
                "regulation": "CMR 2017 Regulation 123 / DGMS Tech Circular 3 of 2010",
                "title": "Severe Immediate Roof Bed Separation",
                "finding": f"Dual-height extensometer indicates {bed_sep:.1f} mm bed parting in immediate roof.",
                "action": "Install full-column grouted cable bolts penetrating beyond upper anchor horizon."
            })

        # RULE 6: Post-Blasting Vibration Shock
        if blasting == "Immediate (<4h)" and vibration > 12.0:
            triggered_rules.append({
                "rule_id": "RULE_BLASTING_INDUCED_SHOCK",
                "severity": "MEDIUM",
                "regulation": "CMR 2017 Regulation 193 / MMR 1961 Reg 160",
                "title": "High Dynamic Blast Vibration Shock",
                "finding": f"Recent blasting induced PPV of {vibration:.1f} mm/s, risking dislodgement of loosened roof slabs.",
                "action": "Enforce mandatory sounding, barring down, and dressed inspection before allowing general mazdoors."
            })

        # RULE 7: 4-Way Junction in Depillaring Panel
        if is_junction and ("Depillaring" in mining_method or depth > 300):
            triggered_rules.append({
                "rule_id": "RULE_DEPILLARING_JUNCTION_HAZARD",
                "severity": "HIGH",
                "regulation": "CMR 2017 Regulation 112 & 123",
                "title": "High Abutment Stress at 4-Way Junction",
                "finding": "Diagonal intersection span subject to severe front abutment stress during pillar extraction.",
                "action": "Erect corner breaker props and central steel chock support as mandated by depillaring permission."
            })

        # Determine overall severity from rules
        if any(r["severity"] == "CRITICAL" for r in triggered_rules):
            overall_severity = "CRITICAL"
        elif any(r["severity"] == "HIGH" for r in triggered_rules):
            overall_severity = "HIGH"
        elif any(r["severity"] == "MEDIUM" for r in triggered_rules):
            overall_severity = "MEDIUM"
        else:
            overall_severity = "NORMAL"

        return {
            "status": "EVALUATED",
            "triggered_rules": triggered_rules,
            "rule_severity": overall_severity,
            "rule_count": len(triggered_rules),
            "safety_summary": f"{len(triggered_rules)} statutory/geotechnical rules triggered (Severity: {overall_severity})."
        }


if __name__ == "__main__":
    engine = GeotechnicalSafetyRuleEngine()
    test_obs = {
        "depth_m": 350,
        "gallery_width_m": 5.1,
        "rmr_rating": 42,
        "support_density_bolts_per_m2": 0.85,
        "bolt_capacity_tonnes": 5.2,
        "roof_convergence_mm_day": 7.2,
        "bed_separation_mm": 18.5,
        "support_condition": "Excessive Load/Yielding",
        "recent_blasting": "Immediate (<4h)",
        "blast_vibration_ppv": 14.5,
        "is_junction": 1,
        "mining_method": "Bord and Pillar (Depillaring)"
    }
    res = engine.evaluate_rules(test_obs)
    print("Rule Engine Test Result:")
    print(f"Overall Severity: {res['rule_severity']} | Count: {res['rule_count']}")
    for r in res["triggered_rules"]:
        print(f"  [{r['severity']}] {r['title']} ({r['regulation']}): {r['finding']}")
