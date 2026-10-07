"""
BHOOmiKAVACH AI - Phase 1: SQLite Database Builder
==================================================
Builds the relational SQLite database `bhoomikavach.db` and loads
the extracted historical DGMS roof-fall datasets into indexed tables.

Tables:
  - `accidents`: Detailed historical fatal roof fall records (2016-2022) with full narratives,
    causes, contributing factors, and preventive findings.
  - `major_accidents`: Century-long historical major roof fall events (1901-2022) from Chapter 10.

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import sqlite3
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DB_PATH = "data/processed/bhoomikavach.db"
ACCIDENTS_CSV = "data/processed/roof_fall_accidents.csv"
MAJOR_ACCIDENTS_CSV = "data/processed/major_roof_falls_1901_2022.csv"


def create_schema(conn: sqlite3.Connection):
    """Create normalized relational tables and indices."""
    cursor = conn.cursor()
    
    # 1. Primary Detailed Roof-Fall Accidents Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS accidents (
        accident_id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        time TEXT,
        year INTEGER NOT NULL,
        mine TEXT NOT NULL,
        owner TEXT,
        mineral TEXT NOT NULL,
        mine_type TEXT NOT NULL,
        coal_noncoal TEXT NOT NULL,
        state TEXT NOT NULL,
        district TEXT,
        location TEXT,
        underground_opencast TEXT NOT NULL,
        accident_code TEXT NOT NULL,
        cause TEXT NOT NULL,
        subcause TEXT NOT NULL,
        activity TEXT,
        designation TEXT,
        age TEXT,
        experience TEXT,
        fatalities INTEGER NOT NULL,
        serious_injuries INTEGER DEFAULT 0,
        accident_description TEXT NOT NULL,
        immediate_cause TEXT,
        contributing_factors TEXT,
        root_cause TEXT,
        preventive_action TEXT,
        regulation_reference TEXT,
        source_document TEXT NOT NULL,
        source_page TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # 2. Historical Major Roof Falls Since 1901 Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS major_accidents (
        accident_id TEXT PRIMARY KEY,
        date TEXT NOT NULL,
        year INTEGER NOT NULL,
        mine TEXT NOT NULL,
        coal_noncoal TEXT NOT NULL,
        mineral TEXT NOT NULL,
        fatalities INTEGER NOT NULL,
        serious_injuries INTEGER DEFAULT 0,
        cause TEXT NOT NULL,
        subcause TEXT NOT NULL,
        raw_cause TEXT,
        source_document TEXT NOT NULL,
        source_page TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Indices for high-performance agentic queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_year ON accidents(year);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_state ON accidents(state);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_mineral ON accidents(mineral);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_code ON accidents(accident_code);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_accidents_coal_noncoal ON accidents(coal_noncoal);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_major_year ON major_accidents(year);")
    
    conn.commit()
    logger.info("Database schema and indices created successfully.")


def populate_database(conn: sqlite3.Connection):
    """Load CSV records into SQLite tables."""
    if not os.path.exists(ACCIDENTS_CSV):
        raise FileNotFoundError(f"Missing {ACCIDENTS_CSV}. Run extractor.py first.")
    
    df_accidents = pd.read_csv(ACCIDENTS_CSV)
    df_accidents["accident_code"] = df_accidents["accident_code"].astype(str).str.zfill(4)
    # Fill NAs with appropriate defaults
    df_accidents.fillna("NULL", inplace=True)
    
    cursor = conn.cursor()
    cursor.execute("DELETE FROM accidents;")
    
    insert_acc_sql = """
    INSERT INTO accidents (
        accident_id, date, time, year, mine, owner, mineral, mine_type,
        coal_noncoal, state, district, location, underground_opencast,
        accident_code, cause, subcause, activity, designation, age, experience,
        fatalities, serious_injuries, accident_description, immediate_cause,
        contributing_factors, root_cause, preventive_action, regulation_reference,
        source_document, source_page
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    
    records_to_insert = [
        tuple(row[col] for col in [
            "accident_id", "date", "time", "year", "mine", "owner", "mineral", "mine_type",
            "coal_noncoal", "state", "district", "location", "underground_opencast",
            "accident_code", "cause", "subcause", "activity", "designation", "age", "experience",
            "fatalities", "serious_injuries", "accident_description", "immediate_cause",
            "contributing_factors", "root_cause", "preventive_action", "regulation_reference",
            "source_document", "source_page"
        ])
        for _, row in df_accidents.iterrows()
    ]
    cursor.executemany(insert_acc_sql, records_to_insert)
    logger.info(f"Inserted {len(records_to_insert)} records into `accidents` table.")

    # Populate major accidents table
    if os.path.exists(MAJOR_ACCIDENTS_CSV):
        df_major = pd.read_csv(MAJOR_ACCIDENTS_CSV)
        df_major.fillna("NULL", inplace=True)
        cursor.execute("DELETE FROM major_accidents;")
        
        insert_major_sql = """
        INSERT INTO major_accidents (
            accident_id, date, year, mine, coal_noncoal, mineral,
            fatalities, serious_injuries, cause, subcause, raw_cause,
            source_document, source_page
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        major_records = [
            tuple(row[col] for col in [
                "accident_id", "date", "year", "mine", "coal_noncoal", "mineral",
                "fatalities", "serious_injuries", "cause", "subcause", "raw_cause",
                "source_document", "source_page"
            ])
            for _, row in df_major.iterrows()
        ]
        cursor.executemany(insert_major_sql, major_records)
        logger.info(f"Inserted {len(major_records)} records into `major_accidents` table.")

    conn.commit()


def verify_database(conn: sqlite3.Connection):
    """Verify row counts and sample records from SQLite."""
    cursor = conn.cursor()
    cursor.execute("SELECT count(*), sum(fatalities), count(DISTINCT mine) FROM accidents;")
    acc_count, fat_sum, mine_count = cursor.fetchone()
    logger.info(f"Verification -> Accidents: {acc_count} rows, Fatalities: {fat_sum}, Unique Mines: {mine_count}")

    cursor.execute("SELECT count(*), sum(fatalities) FROM major_accidents;")
    maj_count, maj_fat_sum = cursor.fetchone()
    logger.info(f"Verification -> Major Accidents: {maj_count} rows, Fatalities: {maj_fat_sum}")

    return {
        "accidents_count": acc_count,
        "fatalities_sum": fat_sum,
        "unique_mines": mine_count,
        "major_accidents_count": maj_count,
        "major_fatalities_sum": maj_fat_sum
    }


def build_database():
    """Main execution entrypoint for SQLite database construction."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        create_schema(conn)
        populate_database(conn)
        stats = verify_database(conn)
        return stats
    finally:
        conn.close()


if __name__ == "__main__":
    stats = build_database()
    print("Database build complete:", stats)
