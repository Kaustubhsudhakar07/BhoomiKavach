"""
BHOOmiKAVACH AI — Historical Accident Analytics Agent (Phase 22)
================================================================
Queries SQLite database `data/processed/bhoomikavach.db` to extract
rigorous, unhallucinated DGMS roof-fall facts, counts, and inquiry patterns.
"""

import sqlite3
import pandas as pd
from typing import Dict, Any
from backend.app.agents.state import AgentState

DB_PATH = "data/processed/bhoomikavach.db"


def historical_node(state: AgentState) -> Dict[str, Any]:
    query = state.get("query", "").lower()
    stats = {}

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 1. Total records and fatalities
        cursor.execute("SELECT count(*), sum(fatalities), count(DISTINCT mine) FROM accidents;")
        row = cursor.fetchone()
        stats["total_accidents_2016_2022"] = row[0]
        stats["total_fatalities_2016_2022"] = row[1]
        stats["total_mines"] = row[2]

        # 2. Coal vs Non-Coal
        cursor.execute("SELECT coal_noncoal, count(*), sum(fatalities) FROM accidents GROUP BY coal_noncoal;")
        stats["by_sector"] = {r[0]: {"accidents": r[1], "fatalities": r[2]} for r in cursor.fetchall()}

        # 3. Top States
        cursor.execute("SELECT state, count(*) as cnt FROM accidents GROUP BY state ORDER BY cnt DESC LIMIT 5;")
        stats["top_states"] = [{"state": r[0], "count": r[1]} for r in cursor.fetchall()]

        # 4. Top Minerals
        cursor.execute("SELECT mineral, count(*) as cnt FROM accidents GROUP BY mineral ORDER BY cnt DESC LIMIT 5;")
        stats["top_minerals"] = [{"mineral": r[0], "count": r[1]} for r in cursor.fetchall()]

        # 5. Top Victim Designations
        cursor.execute("SELECT designation, count(*) as cnt FROM accidents GROUP BY designation ORDER BY cnt DESC LIMIT 5;")
        stats["vulnerable_designations"] = [{"designation": r[0], "count": r[1]} for r in cursor.fetchall()]

        conn.close()
    except Exception as e:
        stats["error"] = str(e)

    return {"historical_stats": stats}
