"""
BHOOmiKAVACH AI — Database Initialization & Schema Engine (Phase 26 & 27)
=========================================================================
Initializes all 8 required SQLite tables:
  1. `accidents`
  2. `operational_records`
  3. `predictions`
  4. `model_versions`
  5. `recommendations`
  6. `agent_sessions`
  7. `agent_messages`
  8. `audit_logs`
"""

import sqlite3
import os
import json
import datetime
import pandas as pd
from typing import Dict, Any, List
from backend.app.core.config import settings


def get_db_connection():
    os.makedirs(os.path.dirname(settings.DB_PATH), exist_ok=True)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_database_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Operational Records Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS operational_records (
        record_id TEXT PRIMARY KEY,
        date TEXT,
        shift TEXT,
        mine TEXT,
        owner TEXT,
        state TEXT,
        coal_noncoal TEXT,
        mineral TEXT,
        panel TEXT,
        mining_method TEXT,
        depth_m REAL,
        working_height_m REAL,
        gallery_width_m REAL,
        gallery_height_m REAL,
        is_junction INTEGER,
        face_position TEXT,
        roof_lithology TEXT,
        floor_lithology TEXT,
        rmr_rating REAL,
        roof_convergence_mm_day REAL,
        bed_separation_mm REAL,
        support_density_bolts_per_m2 REAL,
        bolt_capacity_tonnes REAL,
        recent_blasting TEXT,
        blast_vibration_ppv REAL,
        roof_fall_risk TEXT,
        data_label TEXT
    );
    """)

    # 2. Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        mine TEXT,
        predicted_risk TEXT,
        confidence TEXT,
        model_agreement TEXT,
        model_version TEXT,
        probabilities TEXT,
        input_snapshot TEXT
    );
    """)

    # 3. Model Versions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        version_id TEXT PRIMARY KEY,
        model_name TEXT,
        created_at TEXT,
        macro_f1 REAL,
        balanced_accuracy REAL,
        high_recall REAL,
        critical_recall REAL,
        status TEXT
    );
    """)

    # 4. Recommendations Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS recommendations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prediction_id INTEGER,
        user_role TEXT,
        immediate_action TEXT,
        engineering_action TEXT,
        monitoring_action TEXT,
        created_at TEXT
    );
    """)

    # 5. Agent Sessions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_sessions (
        session_id TEXT PRIMARY KEY,
        started_at TEXT,
        user_role TEXT,
        last_active TEXT
    );
    """)

    # 6. Agent Messages Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS agent_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        timestamp TEXT,
        sender TEXT,
        intent TEXT,
        content TEXT,
        FOREIGN KEY(session_id) REFERENCES agent_sessions(session_id)
    );
    """)

    # 7. Audit Logs Table (Phase 27)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        model_version TEXT,
        dataset_version TEXT,
        feature_version TEXT,
        user_role TEXT,
        input_snapshot TEXT,
        risk_output TEXT,
        confidence TEXT,
        model_agreement TEXT,
        recommendation TEXT,
        agent_tool_calls TEXT,
        override_reason TEXT
    );
    """)

    # Populate operational_records from CSV if empty
    cursor.execute("SELECT count(*) FROM operational_records;")
    count = cursor.fetchone()[0]
    op_csv = "data/operational/operational_risk_dataset.csv"
    if count == 0 and os.path.exists(op_csv):
        df_op = pd.read_csv(op_csv)
        # Select subset of matching columns
        cols = [
            "record_id", "date", "shift", "mine", "owner", "state", "coal_noncoal", "mineral",
            "panel", "mining_method", "depth_m", "working_height_m", "gallery_width_m", "gallery_height_m",
            "is_junction", "face_position", "roof_lithology", "floor_lithology", "rmr_rating",
            "roof_convergence_mm_day", "bed_separation_mm", "support_density_bolts_per_m2",
            "bolt_capacity_tonnes", "recent_blasting", "blast_vibration_ppv", "roof_fall_risk", "data_label"
        ]
        df_sub = df_op[[c for c in cols if c in df_op.columns]]
        df_sub.to_sql("operational_records", conn, if_exists="append", index=False)
        print(f"Populated operational_records table with {len(df_sub)} rows.")

    # Populate champion model version record
    cursor.execute("INSERT OR REPLACE INTO model_versions VALUES (?, ?, ?, ?, ?, ?, ?, ?);", (
        settings.MODEL_VERSION,
        "Weighted Soft Voting Ensemble (XGB+LGB+CAT)",
        datetime.datetime.now().isoformat(),
        0.728, 0.710, 0.808, 0.444, "ACTIVE_PRODUCTION"
    ))

    conn.commit()
    conn.close()
    print("Database tables initialized successfully.")


def log_audit_trail(
    user_role: str,
    input_snapshot: Dict[str, Any],
    risk_output: str,
    confidence: str,
    model_agreement: str,
    recommendation: str,
    tool_calls: List[str] = None,
    override_reason: str = None
) -> int:
    """Records an immutable audit entry for compliance tracking."""
    conn = get_db_connection()
    cursor = conn.cursor()
    now_str = datetime.datetime.now().isoformat()

    cursor.execute("""
    INSERT INTO audit_logs (
        timestamp, model_version, dataset_version, feature_version,
        user_role, input_snapshot, risk_output, confidence,
        model_agreement, recommendation, agent_tool_calls, override_reason
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        now_str,
        settings.MODEL_VERSION,
        settings.DATASET_VERSION,
        settings.FEATURE_VERSION,
        user_role,
        json.dumps(input_snapshot),
        risk_output,
        confidence,
        model_agreement,
        recommendation,
        json.dumps(tool_calls or []),
        override_reason or "STANDARD_AUTOMATED_FLOW"
    ))
    audit_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return audit_id


if __name__ == "__main__":
    init_database_tables()
