"""
BHOOmiKAVACH AI — Audit Trail Logs API Route (Phase 27)
=======================================================
"""

import json
from fastapi import APIRouter
from typing import List, Dict, Any
from backend.app.database.session import get_db_connection

router = APIRouter()


@router.get("/logs")
def get_audit_logs(limit: int = 50, offset: int = 0):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    SELECT * FROM audit_logs ORDER BY audit_id DESC LIMIT ? OFFSET ?;
    """, (limit, offset))
    rows = [dict(r) for r in c.fetchall()]

    c.execute("SELECT count(*) FROM audit_logs;")
    total = c.fetchone()[0]
    conn.close()

    # Parse JSON snapshots for friendly display
    for r in rows:
        try:
            r["input_snapshot"] = json.loads(r["input_snapshot"])
        except Exception:
            pass
        try:
            r["agent_tool_calls"] = json.loads(r["agent_tool_calls"])
        except Exception:
            pass

    return {"total_audit_records": total, "logs": rows}
