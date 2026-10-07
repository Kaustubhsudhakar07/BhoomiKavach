"""
BHOOmiKAVACH AI - Phase 1 Test Suite
====================================
Tests:
  - Raw DGMS PDF integrity and presence
  - Extractor output completeness and record counts (30 Coal, 11 Non-Coal = 41 total)
  - 30-column canonical schema adherence
  - Accident code 0111 and Ground Movement classification verification
  - ISO 8601 date normalization and validity
  - Casualty and fatality count consistency
  - No duplicate accident records
  - Strict absence of synthetic data in historical Dataset A
  - SQLite database schema and row count integrity
  - Data quality report generation

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import re
import sqlite3
import pytest
import pandas as pd

COAL_PDF = "data/raw/sanket_coal_2016_2022.pdf"
NONCOAL_PDF = "data/raw/sanket_noncoal_2016_2022.pdf"
CSV_PATH = "data/processed/roof_fall_accidents.csv"
JSON_PATH = "data/processed/roof_fall_accidents.json"
MAJOR_CSV = "data/processed/major_roof_falls_1901_2022.csv"
DB_PATH = "data/processed/bhoomikavach.db"
QUALITY_REPORT = "data/processed/data_quality_report.csv"

EXPECTED_COLUMNS = [
    "accident_id", "date", "time", "year", "mine", "owner", "mineral",
    "mine_type", "coal_noncoal", "state", "district", "location",
    "underground_opencast", "accident_code", "cause", "subcause",
    "activity", "designation", "age", "experience", "fatalities",
    "serious_injuries", "accident_description", "immediate_cause",
    "contributing_factors", "root_cause", "preventive_action",
    "regulation_reference", "source_document", "source_page"
]


def test_source_pdfs_exist():
    """Verify official DGMS PDFs are downloaded and intact."""
    assert os.path.exists(COAL_PDF), f"Missing {COAL_PDF}"
    assert os.path.getsize(COAL_PDF) > 1_000_000, "Coal PDF size too small."
    assert os.path.exists(NONCOAL_PDF), f"Missing {NONCOAL_PDF}"
    assert os.path.getsize(NONCOAL_PDF) > 1_000_000, "Non-Coal PDF size too small."


def test_processed_files_exist():
    """Verify primary and companion processed datasets exist."""
    assert os.path.exists(CSV_PATH), f"Missing {CSV_PATH}"
    assert os.path.exists(JSON_PATH), f"Missing {JSON_PATH}"
    assert os.path.exists(MAJOR_CSV), f"Missing {MAJOR_CSV}"
    assert os.path.exists(DB_PATH), f"Missing {DB_PATH}"
    assert os.path.exists(QUALITY_REPORT), f"Missing {QUALITY_REPORT}"


def test_canonical_schema():
    """Verify exactly 30 canonical columns are present in primary CSV."""
    df = pd.read_csv(CSV_PATH)
    assert list(df.columns) == EXPECTED_COLUMNS, "Schema columns do not match canonical specification."


def test_total_record_counts():
    """Verify total records: 30 Coal + 11 Non-Coal = 41 primary records."""
    df = pd.read_csv(CSV_PATH)
    assert len(df) == 41, f"Expected 41 primary records, got {len(df)}"
    coal_count = (df["coal_noncoal"] == "Coal").sum()
    noncoal_count = (df["coal_noncoal"] == "Non-Coal").sum()
    assert coal_count == 30, f"Expected 30 Coal records, got {coal_count}"
    assert noncoal_count == 11, f"Expected 11 Non-Coal records, got {noncoal_count}"


def test_accident_code_0111():
    """Verify all records strictly pertain to 0111 Fall of Roof."""
    df = pd.read_csv(CSV_PATH)
    codes = df["accident_code"].astype(str).str.zfill(4).unique()
    assert list(codes) == ["0111"], f"Unexpected accident codes present: {codes}"
    assert (df["cause"] == "Ground Movement").all(), "Cause must be Ground Movement"
    assert (df["subcause"] == "Fall of Roof").all(), "Subcause must be Fall of Roof"


def test_date_format():
    """Verify all dates adhere to ISO 8601 YYYY-MM-DD."""
    df = pd.read_csv(CSV_PATH)
    iso_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for d in df["date"]:
        assert iso_pattern.match(str(d)), f"Invalid ISO date: {d}"


def test_fatalities_positive():
    """Verify fatal accident enquiry records have positive fatalities."""
    df = pd.read_csv(CSV_PATH)
    assert (df["fatalities"] >= 1).all(), "Fatal accident record found with < 1 fatality."
    assert df["fatalities"].sum() == 58, f"Expected 58 fatalities total, got {df['fatalities'].sum()}"


def test_no_duplicate_records():
    """Verify unique accident IDs and unique occurrences."""
    df = pd.read_csv(CSV_PATH)
    assert df["accident_id"].nunique() == len(df), "Duplicate accident_ids found."
    assert not df.duplicated(subset=["date", "mine", "time"]).any(), "Duplicate date/mine/time events found."


def test_no_fabrication_experience():
    """Verify experience is not hallucinated (must be null/empty)."""
    df = pd.read_csv(CSV_PATH)
    # Experience must be NaN in pandas or literal 'NULL'
    valid_nulls = df["experience"].isna() | (df["experience"] == "NULL")
    assert valid_nulls.all(), "Synthetic experience values detected in historical dataset."


def test_sqlite_database_integrity():
    """Verify SQLite database holds identical data and indices."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT count(*) FROM accidents;")
    acc_count = cursor.fetchone()[0]
    assert acc_count == 41, f"SQLite accidents count mismatch: {acc_count} vs 41"
    
    cursor.execute("SELECT count(*) FROM major_accidents;")
    maj_count = cursor.fetchone()[0]
    assert maj_count == 144, f"SQLite major_accidents count mismatch: {maj_count} vs 144"
    
    # Check index existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
    indices = [r[0] for r in cursor.fetchall()]
    assert "idx_accidents_year" in indices
    assert "idx_accidents_state" in indices
    assert "idx_accidents_mineral" in indices
    
    conn.close()
