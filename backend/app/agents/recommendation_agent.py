"""
BHOOmiKAVACH AI — Role-Based Recommendation Agent (Phase 22)
============================================================
Formulates actionable, stratified safety recommendations across 7 mining designations:
  1. Worker (General Mazdoor / Driller / Dresser)
  2. Supervisor (Mining Sirdar / Overman)
  3. Safety Officer
  4. Mining Engineer / Under-Manager
  5. Geotechnical Engineer / Strata Control Officer
  6. Mine Manager (Agent / Statutory First Class Manager)
  7. Executive / Director (Technical) / Board Level

Adheres strictly to DGMS mandates:
  - Immediate Life Safety Actions
  - Engineering & Support Control Actions
  - Monitoring & Strata Audit Actions
  - Relevant Statutory References
"""

from typing import Dict, Any, List
from backend.app.agents.state import AgentState

ROLE_PROTOCOLS = {
    "Worker": {
        "immediate": "Immediately retreat from under any unsupported or drummy strata to a supported bay. Do NOT venture past the last row of roof bolts.",
        "engineering": "Ensure full personal protective equipment (safety helmet, steel-toed boots, cap lamp). Report any sounds of cracking or loose rock flaking.",
        "monitoring": "Observe for trickling dust, slaking shale, or bent props. Await statutory clearance by the Mining Sirdar before resuming loading."
    },
    "Supervisor": {
        "immediate": "Conduct mandatory pre-shift sounding test with a 1.8m testing rod (CMR 129). Bar down all loose roof and side slabs using a pinch bar.",
        "engineering": "Verify that support density adheres strictly to Systematic Support Rules (CMR 125). Ensure temporary props/breaker cogs are set before face workers enter.",
        "monitoring": "Inspect tell-tales and bed separation gauges. Record observations in the statutory shift logbook under CMR Regulation 129."
    },
    "Safety Officer": {
        "immediate": "Issue immediate safety advisory or stop-work order on unapproved over-width galleries exceeding 4.8m (CMR 111).",
        "engineering": "Review compliance of current working panel against approved Systematic Support Rules (SSR) and DGMS permission letters.",
        "monitoring": "Audit routine 10% pull testing of resin-grouted roof bolts per DGMS Circular 6 of 2013. Ensure testing register is updated."
    },
    "Mining Engineer": {
        "immediate": "Re-evaluate face advance sequence and solid blasting charging patterns to mitigate excessive ground vibration (PPV < 10 mm/s).",
        "engineering": "Design supplemental support (e.g. twin-strand cable bolts) for high-stress 4-way junctions and depillaring slice extractions.",
        "monitoring": "Analyze face convergence velocity; ensure extraction rates do not exceed strata equilibrium limits."
    },
    "Geotechnical Engineer": {
        "immediate": "Assess dual-height borehole extensometer and multi-point tell-tale dilation. Compare convergence rates against TARP green/amber/red thresholds.",
        "engineering": "Recalculate RMR (CMRI-ISM rating) and Q-system values across changing geological lithologies and slip plane intersections.",
        "monitoring": "Implement acoustic emission / microseismic monitoring per DGMS Circular 03 of 2010. Issue numerical modeling stress updates."
    },
    "Mine Manager": {
        "immediate": "Exercise statutory authority under CMR 2017: withdraw workers from any panel exhibiting critical convergence or dislodged support.",
        "engineering": "Convene Strata Control Cell meeting. If over-span or severe geological disturbance occurs, submit formal revision to DGMS.",
        "monitoring": "Review mine-wide risk dashboard weekly. Ensure adequate inventory of approved high-grade resin capsules and high-tensile roof bolts."
    },
    "Executive": {
        "immediate": "Ensure uncompromised organizational safety commitment: zero tolerance for deployment under unsupported ground across all subsidiaries.",
        "engineering": "Authorize capital allocation for mechanised roof bolting rigs and automated digital strata monitoring systems.",
        "monitoring": "Review quarterly safety audit metrics and historical 0111 incident recurrence across regional areas."
    }
}


def recommendation_node(state: AgentState) -> Dict[str, Any]:
    user_role = state.get("user_role", "Supervisor")
    risk_data = state.get("risk_data") or {}
    assessed_risk = risk_data.get("assessed_risk", "HIGH")
    
    protocol = ROLE_PROTOCOLS.get(user_role, ROLE_PROTOCOLS["Supervisor"])

    recommendations = {
        "target_role": user_role,
        "assessed_risk": assessed_risk,
        "immediate_actions": protocol["immediate"],
        "engineering_actions": protocol["engineering"],
        "monitoring_actions": protocol["monitoring"],
        "statutory_mandate": "Coal Mines Regulations 2017 / Metalliferous Mines Regulations 1961",
        "competent_advice": "This automated recommendation must be validated by a statutory competent person on-site."
    }

    return {"role_recommendations": recommendations}
