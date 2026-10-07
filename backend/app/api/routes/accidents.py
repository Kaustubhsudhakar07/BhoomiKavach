"""
BHOOmiKAVACH AI — Historical Accidents API Routes (Phase 23)
============================================================
"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from backend.app.database.session import get_db_connection
from src.similarity.accident_matcher import AccidentSimilarityMatcher

router = APIRouter()
similarity_matcher = AccidentSimilarityMatcher()


class SimilarAccidentRequest(BaseModel):
    query: str
    top_k: int = 5
    filter_coal_noncoal: Optional[str] = None


@router.get("")
def list_historical_accidents(
    year: Optional[int] = None,
    coal_noncoal: Optional[str] = None,
    state: Optional[str] = None,
    mineral: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    conn = get_db_connection()
    c = conn.cursor()

    query = "SELECT * FROM accidents WHERE 1=1"
    params = []

    if year:
        query += " AND year = ?"
        params.append(year)
    if coal_noncoal:
        query += " AND coal_noncoal = ?"
        params.append(coal_noncoal)
    if state:
        query += " AND state = ?"
        params.append(state)
    if mineral:
        query += " AND mineral = ?"
        params.append(mineral)

    query += " ORDER BY date DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    c.execute(query, params)
    rows = [dict(r) for r in c.fetchall()]

    # Total count
    count_query = "SELECT count(*) FROM accidents WHERE 1=1"
    c.execute(count_query)
    total_count = c.fetchone()[0]

    conn.close()
    return {"total": total_count, "returned": len(rows), "records": rows}


@router.get("/stats")
def get_accident_statistics():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("SELECT count(*), sum(fatalities), count(DISTINCT mine), count(DISTINCT state) FROM accidents;")
    totals = c.fetchone()

    c.execute("SELECT year, count(*), sum(fatalities) FROM accidents GROUP BY year ORDER BY year ASC;")
    by_year = [{"year": r[0], "accidents": r[1], "fatalities": r[2]} for r in c.fetchall()]

    c.execute("SELECT coal_noncoal, count(*), sum(fatalities) FROM accidents GROUP BY coal_noncoal;")
    by_sector = [{"sector": r[0], "accidents": r[1], "fatalities": r[2]} for r in c.fetchall()]

    c.execute("SELECT state, count(*) as cnt, sum(fatalities) FROM accidents GROUP BY state ORDER BY cnt DESC;")
    by_state = [{"state": r[0], "accidents": r[1], "cnt": r[1], "fatalities": r[2]} for r in c.fetchall()]

    c.execute("SELECT mineral, count(*) as cnt, sum(fatalities) FROM accidents GROUP BY mineral ORDER BY cnt DESC;")
    by_mineral = [{"mineral": r[0], "accidents": r[1], "cnt": r[1], "fatalities": r[2]} for r in c.fetchall()]

    conn.close()
    return {
        "total_accidents": totals[0],
        "total_fatalities": totals[1],
        "total_mines_involved": totals[2],
        "total_states_involved": totals[3],
        "accidents_by_year": by_year,
        "accidents_by_sector": by_sector,
        "accidents_by_state": by_state,
        "accidents_by_mineral": by_mineral
    }


@router.get("/{accident_id}")
def get_accident_by_id(accident_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT * FROM accidents WHERE accident_id = ?;", (accident_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail=f"Accident record '{accident_id}' not found.")
    return dict(row)


@router.post("/similar")
def search_similar_accidents(req: SimilarAccidentRequest):
    try:
        matches = similarity_matcher.query_similar_cases(
            req.query,
            top_k=req.top_k,
            filter_coal_noncoal=req.filter_coal_noncoal
        )
        return {"query": req.query, "total_matches": len(matches), "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search failed: {str(e)}")
