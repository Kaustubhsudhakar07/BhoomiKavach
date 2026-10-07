"""
BHOOmiKAVACH AI — Causal Network & Directed Acyclic Graph (DAG) Engine
=======================================================================
Constructs a domain-grounded Directed Acyclic Graph (DAG) for roof-fall
hazards in Indian underground coal & non-coal mines.

Nodes represent:
  - Geological Preconditions (Depth, RMR, Jointing, Slip planes)
  - Mining Activities & Geometry (Depillaring, Gallery Width, Blasting)
  - Operational & Environmental States (Water Inflow, Abutment Stress, Night Shift)
  - Control & Supervisory Barriers (Roof Bolting, Sounding/Dressing, SSR Inspection)
  - Proximate Hazards (Strata Delamination, Roof Bed Dilation, Support Dislodgement)
  - Terminal Outcome: Roof Fall Incident (Code 0111)

Capabilities:
  - Directed Acyclic Graph construction with NetworkX
  - Structural Causal Path tracing & Markov blanket discovery
  - Upstream root cause tracing from any observation
  - High-res visual generation of the causal graph for web UI / docs

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import json
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple

OUT_DIR = "results/causal_graph"
os.makedirs(OUT_DIR, exist_ok=True)


def build_roof_fall_causal_graph() -> nx.DiGraph:
    """
    Builds the structural causal DAG representing geotechnical, operational,
    and human factors driving roof falls in Indian mines.
    """
    G = nx.DiGraph()

    # ── NODE DEFINITIONS WITH ROLES ──────────────────────────────────────────
    # Geological Antecedents
    G.add_node("Mining_Depth", type="Geology", desc="Depth of Cover (m)")
    G.add_node("RMR_Rating", type="Geology", desc="CMRI-ISM Rock Mass Rating (0-100)")
    G.add_node("Geological_Discontinuities", type="Geology", desc="Slip planes, joints, slips")
    G.add_node("Groundwater_Seepage", type="Environmental", desc="Water percolation, slaking")

    # Mining Geometry & Methods
    G.add_node("Gallery_Span", type="Mining_Geometry", desc="Gallery width / 4-way diagonal span (m)")
    G.add_node("Depillaring_Extraction", type="Mining_Geometry", desc="Active pillar extraction / goaf boundary")
    G.add_node("Blasting_Intensity", type="Operations", desc="Solid blasting vibration & dynamic shock")

    # Stress & Mechanics
    G.add_node("Abutment_Stress_Concentration", type="Mechanics", desc="High vertical induced stress near goaf/pillars")
    G.add_node("Bed_Separation_Dilation", type="Mechanics", desc="Delamination between sandstone/shale layers")

    # Protective Barriers & Human Supervisions
    G.add_node("Statutory_Supervision", type="Human_Barrier", desc="Sirdar/Overman pre-shift check (CMR 129)")
    G.add_node("Dressing_Testing_Standard", type="Human_Barrier", desc="Sounding & dressing down loose rock (CMR 123)")
    G.add_node("Support_Installation_Density", type="Engineering_Barrier", desc="Roof bolts / props installed to SSR (CMR 125)")
    G.add_node("Support_Anchorage_Integrity", type="Engineering_Barrier", desc="Resin capsule anchorage & bolt load bearing capacity")

    # Intermediate Failure States
    G.add_node("Unsupported_Exposure_Span", type="Failure_State", desc="Fresh face or wide span left unsupported")
    G.add_node("Support_Overload_Failure", type="Failure_State", desc="Yielding, snapping, or dislodging of supports")
    G.add_node("Personnel_In_Hazard_Zone", type="Hazard_Exposure", desc="Workers stationed under compromised strata")

    # Terminal Outcome
    G.add_node("ROOF_FALL_INCIDENT", type="Terminal_Outcome", desc="Fatal or Serious Fall of Roof (DGMS Code 0111)")

    # ── DIRECTED EDGES (CAUSAL MECHANISMS) ───────────────────────────────────
    # Geological & Depth effects on stress & separation
    G.add_edge("Mining_Depth", "Abutment_Stress_Concentration", weight=0.85, mechanism="Higher overburden induces deeper stress abutments")
    G.add_edge("Depillaring_Extraction", "Abutment_Stress_Concentration", weight=0.95, mechanism="Caving creates front & side abutment pressure zones")
    
    G.add_edge("RMR_Rating", "Bed_Separation_Dilation", weight=-0.90, mechanism="Lower RMR significantly increases layer delamination risk")
    G.add_edge("Geological_Discontinuities", "Bed_Separation_Dilation", weight=0.88, mechanism="Slip planes act as natural parting surfaces")
    G.add_edge("Groundwater_Seepage", "Bed_Separation_Dilation", weight=0.75, mechanism="Pore pressure reduces friction across bedding planes")
    G.add_edge("Gallery_Span", "Bed_Separation_Dilation", weight=0.82, mechanism="Wider span increases bending deflection and tensile failure")

    # Blasting effects
    G.add_edge("Blasting_Intensity", "Bed_Separation_Dilation", weight=0.80, mechanism="Shock waves propagate micro-cracks into immediate roof")
    G.add_edge("Blasting_Intensity", "Support_Overload_Failure", weight=0.70, mechanism="Flyrock or ground vibration dislodges props/bolts")

    # Protective barrier interactions
    G.add_edge("Statutory_Supervision", "Dressing_Testing_Standard", weight=0.90, mechanism="Strict inspection enforces systematic sounding")
    G.add_edge("Statutory_Supervision", "Support_Installation_Density", weight=0.85, mechanism="Enforces adherence to approved Systematic Support Rules")
    
    G.add_edge("Support_Installation_Density", "Unsupported_Exposure_Span", weight=-0.95, mechanism="Adequate bolt density eliminates unsupported spans")
    G.add_edge("Gallery_Span", "Unsupported_Exposure_Span", weight=0.60, mechanism="Excessive width strains standard support spacing")

    G.add_edge("Abutment_Stress_Concentration", "Support_Overload_Failure", weight=0.85, mechanism="Excessive rock load exceeds rated capacity of supports")
    G.add_edge("Support_Anchorage_Integrity", "Support_Overload_Failure", weight=-0.90, mechanism="High anchorage capacity prevents premature pullout")

    # Exposure pathway
    G.add_edge("Statutory_Supervision", "Personnel_In_Hazard_Zone", weight=-0.80, mechanism="Supervisors prevent worker entry into high-risk spots")
    G.add_edge("Dressing_Testing_Standard", "ROOF_FALL_INCIDENT", weight=-0.75, mechanism="Eliminating loose flakes directly removes small fall hazard")

    # Convergence to Terminal Incident
    G.add_edge("Bed_Separation_Dilation", "ROOF_FALL_INCIDENT", weight=0.90, mechanism="Separated slab detaches under self-weight")
    G.add_edge("Unsupported_Exposure_Span", "ROOF_FALL_INCIDENT", weight=0.95, mechanism="Direct fall of immediate stratum over exposed workers")
    G.add_edge("Support_Overload_Failure", "ROOF_FALL_INCIDENT", weight=0.92, mechanism="Catastrophic collapse once support barrier yields")
    G.add_edge("Personnel_In_Hazard_Zone", "ROOF_FALL_INCIDENT", weight=0.88, mechanism="Fatal outcome conditioned on physical presence")

    return G


def trace_causal_paths(G: nx.DiGraph, source: str, target: str = "ROOF_FALL_INCIDENT") -> List[List[str]]:
    """Finds all directed causal pathways from a given risk factor to terminal outcome."""
    if source not in G or target not in G:
        return []
    try:
        return list(nx.all_simple_paths(G, source, target))
    except Exception:
        return []


def get_markov_blanket(G: nx.DiGraph, node: str) -> Dict[str, List[str]]:
    """Computes parents, children, and spouses (coparents) for a node."""
    if node not in G:
        return {}
    parents = list(G.predecessors(node))
    children = list(G.successors(node))
    spouses = set()
    for child in children:
        for p in G.predecessors(child):
            if p != node:
                spouses.add(p)
    return {
        "node": node,
        "parents": parents,
        "children": children,
        "spouses": list(spouses)
    }


def render_causal_graph(G: nx.DiGraph) -> str:
    """Renders the causal DAG as an aesthetically designed dark industrial diagram."""
    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor("#0D1117")
    ax.set_facecolor("#0D1117")

    # Layered hierarchical layout
    pos = {
        "Mining_Depth": (-2.5, 3.0),
        "RMR_Rating": (-1.0, 3.2),
        "Geological_Discontinuities": (0.5, 3.2),
        "Groundwater_Seepage": (2.0, 3.0),
        "Gallery_Span": (3.5, 2.8),
        "Depillaring_Extraction": (-3.0, 1.8),
        "Blasting_Intensity": (-1.5, 1.8),

        "Statutory_Supervision": (1.0, 1.8),
        "Support_Anchorage_Integrity": (3.0, 1.8),

        "Abutment_Stress_Concentration": (-2.5, 0.4),
        "Bed_Separation_Dilation": (-0.8, 0.4),
        "Dressing_Testing_Standard": (0.8, 0.4),
        "Support_Installation_Density": (2.6, 0.4),

        "Unsupported_Exposure_Span": (1.8, -1.0),
        "Support_Overload_Failure": (-1.8, -1.0),
        "Personnel_In_Hazard_Zone": (0.0, -1.0),

        "ROOF_FALL_INCIDENT": (0.0, -2.4)
    }

    # Node coloring by category
    node_colors = []
    type_color_map = {
        "Geology": "#0288D1",              # Blue
        "Environmental": "#00897B",        # Teal
        "Mining_Geometry": "#FB8C00",      # Amber
        "Operations": "#E53935",           # Crimson
        "Mechanics": "#8E24AA",            # Purple
        "Human_Barrier": "#43A047",        # Emerald green
        "Engineering_Barrier": "#1E88E5",  # Dodger blue
        "Failure_State": "#D81B60",        # Hot pink/red
        "Hazard_Exposure": "#F4511E",      # Deep orange
        "Terminal_Outcome": "#B71C1C"      # Blood red
    }

    for node in G.nodes():
        ntype = G.nodes[node].get("type", "Geology")
        node_colors.append(type_color_map.get(ntype, "#757575"))

    # Draw edges with varying thickness
    for u, v, data in G.edges(data=True):
        weight = abs(data.get("weight", 0.5))
        is_negative = data.get("weight", 0.5) < 0
        edge_clr = "#26A69A" if is_negative else "#EF5350"
        style = "dashed" if is_negative else "solid"
        
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)],
            edge_color=edge_clr,
            width=1.0 + weight * 2.0,
            arrows=True,
            arrowsize=14,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            style=style,
            alpha=0.85,
            ax=ax
        )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=2800,
        edgecolors="#E0E0E0",
        linewidths=1.5,
        ax=ax
    )

    # Draw labels with clean formatting
    labels = {n: n.replace("_", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=7.5,
        font_color="#FFFFFF",
        font_weight="bold",
        font_family="sans-serif",
        ax=ax
    )

    # Custom legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=c, label=k.replace("_", " "))
        for k, c in type_color_map.items()
    ]
    legend_patches.append(mpatches.Patch(color="#EF5350", label="Exacerbating Link (+)"))
    legend_patches.append(mpatches.Patch(color="#26A69A", label="Mitigating Barrier Link (-)"))

    ax.legend(
        handles=legend_patches,
        loc="upper left",
        facecolor="#161B22",
        edgecolor="#30363D",
        labelcolor="#E6EDF3",
        fontsize=8.5,
        framealpha=0.9
    )

    ax.set_title(
        "BHOOmiKAVACH AI — Geotechnical & Regulatory Causal Graph for Mine Roof Fall (DGMS 0111)\n"
        "Structural Causal Model (SCM) mapping Geological Triggers, Operational Stresses & Safety Barrier Mitigations",
        fontsize=12, fontweight="bold", color="#F9A825", pad=15
    )
    ax.axis("off")

    out_path = os.path.join(OUT_DIR, "causal_dag.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#0D1117")
    plt.close(fig)
    print(f"  [OK] Causal DAG diagram saved: {out_path}")

    # Export graph structure JSON
    graph_dict = {
        "nodes": [{"id": n, "type": G.nodes[n].get("type"), "desc": G.nodes[n].get("desc")} for n in G.nodes()],
        "edges": [{"from": u, "to": v, "weight": d.get("weight"), "mechanism": d.get("mechanism")} for u, v, d in G.edges(data=True)]
    }
    json_path = os.path.join(OUT_DIR, "causal_graph.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f, indent=2)
    print(f"  [OK] Causal graph JSON saved: {json_path}")

    return out_path


if __name__ == "__main__":
    G = build_roof_fall_causal_graph()
    render_causal_graph(G)
    paths = trace_causal_paths(G, "Depillaring_Extraction")
    print(f"Traced {len(paths)} causal pathways from Depillaring to Roof Fall Incident.")
