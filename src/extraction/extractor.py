"""
BHOOmiKAVACH AI - Phase 1: DGMS SANKET PDF Extraction Pipeline
==============================================================
Extracts historical DGMS fatal roof-fall accident records (Code 0111)
from SANKET Coal (2016-2022) and SANKET Non-Coal (2016-2022) publications,
plus Chapter 10 Major Accidents Since 1901.

Author: BHOOmiKAVACH AI Engineering Team
"""

import os
import re
import json
import logging
from typing import Dict, List, Any
import fitz  # PyMuPDF
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Standardized Canonical Schema (30 Columns)
CANONICAL_COLUMNS = [
    "accident_id",
    "date",
    "time",
    "year",
    "mine",
    "owner",
    "mineral",
    "mine_type",
    "coal_noncoal",
    "state",
    "district",
    "location",
    "underground_opencast",
    "accident_code",
    "cause",
    "subcause",
    "activity",
    "designation",
    "age",
    "experience",
    "fatalities",
    "serious_injuries",
    "accident_description",
    "immediate_cause",
    "contributing_factors",
    "root_cause",
    "preventive_action",
    "regulation_reference",
    "source_document",
    "source_page"
]


def clean_text(text: str) -> str:
    """Normalize whitespace and strip spurious control characters."""
    if not text:
        return "NULL"
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def normalize_date(raw_date: str) -> str:
    """Convert DD.MM.YY, DD.MM.YYYY, or DD-MM-YYYY to ISO YYYY-MM-DD."""
    if not raw_date or raw_date == "NULL":
        return "NULL"
    raw_date = raw_date.strip()
    # Handle hyphen-separated format: DD-MM-YYYY (used in Chapter 10 tables)
    if "-" in raw_date and "." not in raw_date:
        parts = raw_date.split("-")
        if len(parts) == 3:
            d, m, y = parts[0].zfill(2), parts[1].zfill(2), parts[2].strip()
            if len(y) == 4:
                return f"{y}-{m}-{d}"
            elif len(y) == 2:
                y = "20" + y if int(y) <= 22 else "19" + y
                return f"{y}-{m}-{d}"
    # Handle dot or slash separated format: DD.MM.YY or DD.MM.YYYY
    raw_date = raw_date.replace("/", ".")
    parts = raw_date.split(".")
    if len(parts) == 3:
        d, m, y = parts[0].zfill(2), parts[1].zfill(2), parts[2].strip()
        if len(y) == 2:
            # DGMS reports are up to 2022; 2-digit years <= 22 are 2000-2022, > 22 are 1923-1999
            y = "20" + y if int(y) <= 22 else "19" + y
        return f"{y}-{m}-{d}"
    return "NULL"


def normalize_time(raw_time: str) -> str:
    """Normalize raw DGMS time string into HH:MM (24-hour)."""
    if not raw_time or raw_time == "NULL":
        return "NULL"
    cleaned = raw_time.strip().replace(" ", "")
    if "." in cleaned:
        parts = cleaned.split(".")
        if len(parts) == 2 and parts[0]:
            h = parts[0].zfill(2)
            m = parts[1].ljust(2, "0")[:2]
            return f"{h}:{m}"
    elif ":" in cleaned:
        parts = cleaned.split(":")
        if len(parts) == 2 and parts[0]:
            h = parts[0].zfill(2)
            m = parts[1].ljust(2, "0")[:2]
            return f"{h}:{m}"
    return "NULL"


def extract_regulations(text: str) -> str:
    """Extract all statutory regulation references mentioned in the findings."""
    patterns = [
        r"Regulation\s+[0-9A-Za-z\(\)\s,&]+(?:of\s+(?:the\s+)?(?:Coal\s+Mines\s+Regulations|Metalliferous\s+Mines\s+Regulations|CMR|MMR|Mines\s+Act)[^;\.\n]*)",
        r"Section\s+[0-9A-Za-z\(\)]+\s+of\s+(?:the\s+)?Mines\s+Act[^;\.\n]*",
        r"DGMS(?:\s*\(Tech\.\)|\s*\(Technical\))?\s*Circular[^;\.\n]*",
        r"Mines\s+Vocational\s+Training\s+Rules[^;\.\n]*",
    ]
    citations = []
    for pat in patterns:
        matches = re.findall(pat, text, re.IGNORECASE)
        for m in matches:
            cleaned = clean_text(m).rstrip(",.")
            if cleaned and cleaned not in citations:
                citations.append(cleaned)
    return "; ".join(citations) if citations else "NULL"


def extract_immediate_cause(desc: str) -> str:
    """Extract physical rock-fall mechanism and strata dimensions from description."""
    desc_clean = re.sub(r"\s+", " ", desc)
    # Search for specific rock detachment descriptions
    parts = desc_clean.split(", ")
    for p in parts[1:]:
        if any(w in p.lower() for w in ["parted", "fell", "caved", "dislodged", "collapsed", "separated", "slid"]):
            p_clean = p.strip()
            if len(p_clean) > 20:
                return p_clean[0].upper() + p_clean[1:]
    # Fallback pattern
    m = re.search(r"((?:a\s+)?(?:piece|mass|chunk|portion|block|layer|stone|roof rocks?)[^\.;]+?(?:parted|fell|caved|dislodged|collapsed)[^\.;]*)", desc_clean, re.I)
    if m:
        c = m.group(1).strip()
        return c[0].upper() + c[1:]
    return "Parting and fall of roof strata from exposed gallery/face"


def extract_activity(desc: str) -> str:
    """Extract the mining activity being performed at the time of the event."""
    desc_clean = re.sub(r"\s+", " ", desc)
    m = re.search(r"While\s+(.*?)(?:,\s*(?:a\s+|an\s+|the\s+|suddenly|all\s+of\s+a\s+sudden|one\s+person|two\s+persons))", desc_clean, re.I)
    if m:
        act = m.group(1).strip()
        return act[:140]
    m_simple = re.search(r"While\s+(.*?)(?:,|\.)", desc_clean, re.I)
    if m_simple:
        return m_simple.group(1).strip()[:140]
    return "Underground mining operations"


def extract_location(desc: str) -> str:
    """Extract specific underground location from description."""
    desc_clean = re.sub(r"\s+", " ", desc)
    patterns = [
        r"(?:in|at)\s+(?:a|an)?\s*([a-zA-Z0-9\s\-/]+(?:gallery|panel|district|junction|heading|face|stope|drive|goaf\s+edge|incline|slice|shaft\s+inset))",
        r"(junction\s+of\s+[0-9A-Za-z\s\-/]+)",
        r"(depillaring\s+(?:panel|district|with\s+caving))",
        r"(development\s+(?:gallery|district|face))",
        r"(goaf\s+edge\s+of\s+[a-zA-Z0-9\s\-/]+)"
    ]
    for pat in patterns:
        m = re.search(pat, desc_clean, re.I)
        if m:
            loc = m.group(1).strip()
            return loc[:100]
    return "Underground workings"


def extract_contributing_factors(desc: str, had: str) -> str:
    """Extract geological, geotechnical, and operational precursor factors."""
    factors = []
    text_combined = desc + " " + had
    text_lower = text_combined.lower()
    
    if "blasting" in text_lower or "after blasting" in text_lower or "shot-fir" in text_lower:
        factors.append("Recent blasting causing strata shock and induced stress")
    if "unsupported" in text_lower or "not kept adequately supported" in text_lower or "un-supported" in text_lower:
        factors.append("Deployment of personnel under unsupported or inadequately supported roof")
    if "sandstone" in text_lower or "sand stone" in text_lower:
        factors.append("Sandstone roof strata prone to bed separation")
    if "shale" in text_lower:
        factors.append("Weak shale / laminated roof layer susceptible to weathering")
    if "goaf" in text_lower:
        factors.append("High abutment stress concentration near active goaf edge")
    if "depillaring" in text_lower:
        factors.append("Depillaring extraction inducing roof convergence and deformation")
    if "junction" in text_lower:
        factors.append("Increased diagonal span and stress concentration at gallery junction")
    if "dressing" in text_lower:
        factors.append("Failure to adequately test and dress loose roof before work")
    if "slip" in text_lower or "joint" in text_lower or "fault" in text_lower or "disturbed" in text_lower:
        factors.append("Presence of geological slip plane, fault, or structural weakness")
    if "wider than 4.8" in text_lower or "widened" in text_lower:
        factors.append("Excessive gallery width exceeding standard support density")
    if "prop-withdrawer" in text_lower or "withdrawing support" in text_lower:
        factors.append("Manual support withdrawal without safety prop-withdrawer")
    if "water" in text_lower or "seepage" in text_lower:
        factors.append("Water presence / seepage weakening strata cohesion")
    if "convergence" in text_lower:
        factors.append("Accelerated roof convergence exceeding safety threshold")
        
    return "; ".join(factors) if factors else "Inadequate roof support and dressing"


def extract_root_cause(had: str) -> str:
    """Extract statutory and organizational root cause from enquiry findings."""
    had_lower = had.lower()
    causes = []
    if "systematic support rule" in had_lower or "ssr" in had_lower:
        causes.append("Non-compliance with approved Systematic Support Rules (SSR)")
    if "scamp" in had_lower or "strata control" in had_lower:
        causes.append("Inadequate implementation of Strata Control and Monitoring Plan (SCAMP)")
    if "supervis" in had_lower or "sirdar" in had_lower or "overman" in had_lower:
        causes.append("Lapse in statutory supervision and pre-shift roof inspection")
    if "support plan" in had_lower:
        causes.append("Violation of approved Mine Support Plan")
    if "permission" in had_lower and "condition" in had_lower:
        causes.append("Contravention of DGMS permission conditions for extraction/depillaring")
    if not causes:
        causes.append("Failure to secure roof prior to deploying workers as required by mining regulations")
    return "; ".join(causes)


class DGMSAccidentExtractor:
    """Industrial-grade parser for DGMS SANKET Coal and Non-Coal Publications."""

    def __init__(self, coal_pdf_path: str, noncoal_pdf_path: str):
        self.coal_pdf_path = coal_pdf_path
        self.noncoal_pdf_path = noncoal_pdf_path

    def extract_coal_0111_accidents(self) -> List[Dict[str, Any]]:
        """Extract all 30 fatal roof-fall accidents from SANKET Coal (2016-2022)."""
        logger.info(f"Extracting Coal 0111 records from {self.coal_pdf_path}...")
        doc = fitz.open(self.coal_pdf_path)
        
        year_ranges = [
            (2016, 48, 52),
            (2017, 94, 96),
            (2018, 133, 135),
            (2019, 172, 174),
            (2020, 209, 211),
            (2021, 246, 250),
            (2022, 283, 286)
        ]
        
        records = []
        global_coal_id = 1
        
        for yr, sp, ep in year_ranges:
            page_lines = []
            for p in range(sp, ep + 1):
                text = doc[p].get_text()
                for line in text.split("\n"):
                    page_lines.append((p + 1, line))
                    
            in_0111 = False
            cur_acc = []
            cur_pages = []
            acc_blocks = []
            
            for p_num, line in page_lines:
                if "Code : 0111" in line or "Code: 0111" in line:
                    in_0111 = True
                    continue
                if in_0111 and re.search(r"Code\s*:\s*011[2-9]", line):
                    if cur_acc:
                        acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                        cur_acc = []
                        cur_pages = []
                    in_0111 = False
                    break
                if in_0111:
                    st = line.strip()
                    if st.isdigit() and len(st) <= 3:
                        continue
                    if "STATEMENT NO." in st or "Summary of Findings" in st or st.startswith("---") or st.startswith("==="):
                        continue
                    if re.match(r"^\(\d+\s+Deaths?\)$", st, re.I):
                        continue
                    if re.match(r"^\s*\d+\.\s+Date\s*-", line):
                        if cur_acc:
                            acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                            cur_acc = []
                            cur_pages = []
                    cur_acc.append(line)
                    cur_pages.append(p_num)
                    
            if cur_acc and in_0111:
                acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                
            valid_blocks = [b for b in acc_blocks if re.search(r"Date\s*-", b[3])]
            for yr_b, start_p, end_p, blk in valid_blocks:
                rec = self._parse_block(
                    accident_id=f"DGMS_COAL_{yr_b}_{global_coal_id:03d}",
                    year=yr_b,
                    start_page=start_p,
                    end_page=end_p,
                    raw_block=blk,
                    coal_noncoal="Coal",
                    mineral="Coal",
                    source_doc="SANKET: Coal Mine Accidents (2016-2022)"
                )
                records.append(rec)
                global_coal_id += 1
                
        logger.info(f"Successfully extracted {len(records)} Coal 0111 accident records.")
        return records

    def extract_noncoal_0111_accidents(self) -> List[Dict[str, Any]]:
        """Extract all 11 fatal roof-fall accidents from SANKET Non-Coal (2016-2022)."""
        logger.info(f"Extracting Non-Coal 0111 records from {self.noncoal_pdf_path}...")
        doc = fitz.open(self.noncoal_pdf_path)
        
        year_ranges = [
            (2017, 104, 106),
            (2018, 149, 152),
            (2019, 192, 195),
            (2020, 234, 236),
            (2022, 319, 322)
        ]
        
        records = []
        global_nc_id = 1
        
        for yr, sp, ep in year_ranges:
            page_lines = []
            for p in range(sp, ep + 1):
                text = doc[p].get_text()
                for line in text.split("\n"):
                    page_lines.append((p + 1, line))
                    
            in_0111 = False
            cur_acc = []
            cur_pages = []
            acc_blocks = []
            
            for p_num, line in page_lines:
                if "Code : 0111" in line or "Code: 0111" in line:
                    in_0111 = True
                    continue
                if in_0111 and re.search(r"Code\s*:\s*011[2-9]", line):
                    if cur_acc:
                        acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                        cur_acc = []
                        cur_pages = []
                    in_0111 = False
                    break
                if in_0111:
                    st = line.strip()
                    if st.isdigit() and len(st) <= 3:
                        continue
                    if "STATEMENT NO." in st or "Summary of Findings" in st or st.startswith("---") or st.startswith("==="):
                        continue
                    if re.match(r"^\(\d+\s+Deaths?\)$", st, re.I):
                        continue
                    if re.match(r"^\s*\d+\.\s+Date\s*-", line):
                        if cur_acc:
                            acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                            cur_acc = []
                            cur_pages = []
                    cur_acc.append(line)
                    cur_pages.append(p_num)
                    
            if cur_acc and in_0111:
                acc_blocks.append((yr, min(cur_pages), max(cur_pages), "\n".join(cur_acc)))
                
            valid_blocks = [b for b in acc_blocks if re.search(r"Date\s*-", b[3])]
            for yr_b, start_p, end_p, blk in valid_blocks:
                mineral = "Metalliferous"
                blk_l = blk.lower()
                if "manganese" in blk_l:
                    mineral = "Manganese"
                elif "galena" in blk_l or "zinc" in blk_l or "lead" in blk_l:
                    mineral = "Galena & Sphalerite"
                elif "gold" in blk_l:
                    mineral = "Gold"
                elif "copper" in blk_l:
                    mineral = "Copper"
                elif "uranium" in blk_l:
                    mineral = "Uranium"
                elif "stone" in blk_l:
                    mineral = "Stone"
                    
                rec = self._parse_block(
                    accident_id=f"DGMS_NONCOAL_{yr_b}_{global_nc_id:03d}",
                    year=yr_b,
                    start_page=start_p,
                    end_page=end_p,
                    raw_block=blk,
                    coal_noncoal="Non-Coal",
                    mineral=mineral,
                    source_doc="SANKET: Non-Coal Mine Accidents (2016-2022)"
                )
                records.append(rec)
                global_nc_id += 1
                
        logger.info(f"Successfully extracted {len(records)} Non-Coal 0111 accident records.")
        return records

    def _parse_block(self, accident_id: str, year: int, start_page: int, end_page: int, raw_block: str, coal_noncoal: str, mineral: str, source_doc: str) -> Dict[str, Any]:
        """Parse raw text block into canonical schema."""
        # 1. Date
        m_date = re.search(r"Date\s*-\s*([0-9\.]+)", raw_block)
        raw_date = m_date.group(1).strip() if m_date else "NULL"
        norm_date = normalize_date(raw_date)

        # 2. Time
        m_time = re.search(r"Time\s*-\s*([0-9\.\s]+)", raw_block)
        raw_time = m_time.group(1).strip() if m_time else "NULL"
        norm_time = normalize_time(raw_time)

        # 3. Mine
        m_mine = re.search(r"Mine\s*-\s*([^\n\r]+)", raw_block)
        mine = m_mine.group(1).strip() if m_mine else "NULL"
        mine = re.split(r"\s{2,}", mine)[0].strip()

        # 4. Owner
        m_owner = re.search(r"Owner\s*-\s*([^\n\r]+)", raw_block)
        owner = m_owner.group(1).strip() if m_owner else "NULL"
        owner = re.split(r"\s{2,}", owner)[0].strip()

        # 5. District & State
        m_dist = re.search(r"Dist\.\s*-\s*([^,]+),\s*State\s*-\s*([^\n\r]+)", raw_block)
        if m_dist:
            district = m_dist.group(1).strip()
            state = m_dist.group(2).strip()
        else:
            m_d = re.search(r"Dist\.\s*-\s*([^\n\r]+)", raw_block)
            district = m_d.group(1).strip() if m_d else "NULL"
            m_s = re.search(r"State\s*-\s*([^\n\r]+)", raw_block)
            state = m_s.group(1).strip() if m_s else "NULL"

        # 6. Person(s) Killed
        m_persons = re.search(r"Person\(s\)\s*Killed\s*:(.*?)(?:While|\Z)", raw_block, re.DOTALL)
        persons_raw = m_persons.group(1).strip() if m_persons else ""
        p_lines = [l.strip() for l in persons_raw.split("\n") if l.strip()]
        p_text = " ".join(p_lines)
        p_items = re.split(r"(?=\b\d+\.\s+)", p_text)
        
        designations = []
        ages = []
        fatalities = 0
        for it in p_items:
            it = it.strip()
            if not it:
                continue
            fatalities += 1
            m_p = re.match(r"^\d+\.\s*(.*?)(?:,\s*(Male|Female))?(?:,\s*(\d+)\s*Years)?$", it, re.I)
            if m_p:
                pre_gender = m_p.group(1).strip()
                age = m_p.group(3)
                if age:
                    ages.append(age)
                tokens = [tok.strip() for tok in pre_gender.split(",") if tok.strip()]
                if len(tokens) >= 2:
                    designations.append(tokens[1])
                elif len(tokens) == 1:
                    designations.append(tokens[0])
            else:
                designations.append("Worker")
                
        desig_str = "; ".join(designations) if designations else "NULL"
        age_str = "; ".join(ages) if ages else "NULL"
        if fatalities == 0:
            fatalities = 1

        # 7. Description (While ... up to Had)
        m_desc = re.search(r"(While\s+.*?)(?:Had,?\s*|\Z)", raw_block, re.DOTALL | re.IGNORECASE)
        accident_description = clean_text(m_desc.group(1)) if m_desc else "NULL"

        # 8. Preventive action (Had, ...)
        m_had = re.search(r"(Had,?\s*.*)", raw_block, re.DOTALL | re.IGNORECASE)
        preventive_action = clean_text(m_had.group(1)) if m_had else "NULL"

        # 9. Regulation references
        reg_refs = extract_regulations(raw_block)

        # 10. Serious injuries
        serious_inj = 0
        desc_l = accident_description.lower()
        if "serious bodily injury" in desc_l or "serious injuries" in desc_l:
            if "other two received serious" in desc_l or "two received serious" in desc_l:
                serious_inj = 2
            elif "another received serious" in desc_l or "one received serious" in desc_l or "inflicting serious injuries to dresser and supervisor" in desc_l:
                serious_inj = 1
            elif "three received serious" in desc_l:
                serious_inj = 3
            else:
                serious_inj = 1

        # 11. Activity, Location, Causes
        activity = extract_activity(accident_description)
        location = extract_location(accident_description)
        immediate_cause = extract_immediate_cause(accident_description)
        contributing_factors = extract_contributing_factors(accident_description, preventive_action)
        root_cause = extract_root_cause(preventive_action)

        source_page_str = f"Page {start_page}" if start_page == end_page else f"Pages {start_page}-{end_page}"

        return {
            "accident_id": accident_id,
            "date": norm_date,
            "time": norm_time,
            "year": year,
            "mine": mine,
            "owner": owner,
            "mineral": mineral,
            "mine_type": "Coal" if coal_noncoal == "Coal" else "Non-Coal",
            "coal_noncoal": coal_noncoal,
            "state": state,
            "district": district,
            "location": location,
            "underground_opencast": "Underground",
            "accident_code": "0111",
            "cause": "Ground Movement",
            "subcause": "Fall of Roof",
            "activity": activity,
            "designation": desig_str,
            "age": age_str,
            "experience": "NULL",
            "fatalities": fatalities,
            "serious_injuries": serious_inj,
            "accident_description": accident_description,
            "immediate_cause": immediate_cause,
            "contributing_factors": contributing_factors,
            "root_cause": root_cause,
            "preventive_action": preventive_action,
            "regulation_reference": reg_refs,
            "source_document": source_doc,
            "source_page": source_page_str
        }

    def extract_major_accidents_since_1901(self) -> List[Dict[str, Any]]:
        """Extract all historical major roof-fall accidents from Chapter 10 in both documents."""
        logger.info("Extracting Chapter 10 Major Roof Fall Accidents (1901-2022)...")
        major_records = []
        
        # 1. Coal Chapter 10 (PDF pages 299 to 306)
        doc_coal = fitz.open(self.coal_pdf_path)
        for p in range(298, 306):
            lines = [l.strip() for l in doc_coal[p].get_text().split("\n") if l.strip()]
            i = 0
            while i < len(lines):
                if re.match(r"^\d+$", lines[i]):
                    if i + 1 < len(lines) and re.match(r"^\d{2}[-/]\d{2}[-/]\d{2,4}$", lines[i + 1]):
                        sl = lines[i]
                        dt = lines[i + 1]
                        mine_name = lines[i + 2]
                        if i + 4 < len(lines) and lines[i + 3].isdigit() and lines[i + 4].isdigit():
                            killed = int(lines[i + 3])
                            injured = int(lines[i + 4])
                            cause = lines[i + 5] if i + 5 < len(lines) else "Unknown"
                            if "roof" in cause.lower():
                                dt_norm = normalize_date(dt)
                                yr = int(dt_norm[:4]) if dt_norm != "NULL" else 1900
                                major_records.append({
                                    "accident_id": f"DGMS_MAJOR_COAL_{sl.zfill(3)}",
                                    "date": dt_norm,
                                    "year": yr,
                                    "mine": mine_name,
                                    "coal_noncoal": "Coal",
                                    "mineral": "Coal",
                                    "fatalities": killed,
                                    "serious_injuries": injured,
                                    "cause": "Ground Movement",
                                    "subcause": "Fall of Roof",
                                    "raw_cause": cause,
                                    "source_document": "SANKET Coal (Chapter 10: Major Accidents Since 1901)",
                                    "source_page": f"Page {p + 1}"
                                })
                            i += 6
                            continue
                i += 1
                
        # 2. Non-Coal Chapter 10 (PDF pages 345 to 348)
        doc_nc = fitz.open(self.noncoal_pdf_path)
        for p in range(344, 348):
            lines = [l.strip() for l in doc_nc[p].get_text().split("\n") if l.strip()]
            i = 0
            while i < len(lines):
                if re.match(r"^\d+$", lines[i]):
                    if i + 1 < len(lines) and re.match(r"^\d{2}[-/]\d{2}[-/]\d{2,4}$", lines[i + 1]):
                        sl = lines[i]
                        dt = lines[i + 1]
                        j = i + 2
                        while j < len(lines) and not (lines[j].isdigit() and j + 1 < len(lines) and lines[j + 1].isdigit()):
                            j += 1
                        if j < len(lines) and lines[j].isdigit():
                            mine_name = " ".join(lines[i + 2:j])
                            killed = int(lines[j])
                            injured = int(lines[j + 1])
                            cause = lines[j + 2] if j + 2 < len(lines) else "Unknown"
                            if "roof" in cause.lower():
                                dt_norm = normalize_date(dt)
                                yr = int(dt_norm[:4]) if dt_norm != "NULL" else 1900
                                major_records.append({
                                    "accident_id": f"DGMS_MAJOR_NONCOAL_{sl.zfill(3)}",
                                    "date": dt_norm,
                                    "year": yr,
                                    "mine": mine_name,
                                    "coal_noncoal": "Non-Coal",
                                    "mineral": "Non-Coal",
                                    "fatalities": killed,
                                    "serious_injuries": injured,
                                    "cause": "Ground Movement",
                                    "subcause": "Fall of Roof",
                                    "raw_cause": cause,
                                    "source_document": "SANKET Non-Coal (Chapter 10: Major Accidents Since 1901)",
                                    "source_page": f"Page {p + 1}"
                                })
                            i = j + 3
                            continue
                i += 1
                
        logger.info(f"Extracted {len(major_records)} major historical roof-fall accidents since 1901.")
        return major_records

    def run_full_pipeline(self) -> Dict[str, Any]:
        """Execute extraction and save datasets to data/processed/."""
        os.makedirs("data/processed", exist_ok=True)
        
        coal_records = self.extract_coal_0111_accidents()
        noncoal_records = self.extract_noncoal_0111_accidents()
        combined_records = coal_records + noncoal_records
        
        # Convert to DataFrame
        df_accidents = pd.DataFrame(combined_records, columns=CANONICAL_COLUMNS)
        df_accidents["accident_code"] = "0111"  # ensure string format
        df_accidents.sort_values(by=["date", "accident_id"], inplace=True)
        
        # Save primary CSV and JSON
        csv_path = "data/processed/roof_fall_accidents.csv"
        json_path = "data/processed/roof_fall_accidents.json"
        df_accidents.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(combined_records, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Saved primary historical roof fall dataset: {csv_path} ({len(df_accidents)} rows)")
        logger.info(f"Saved JSON export: {json_path}")
        
        # Major accidents
        major_records = self.extract_major_accidents_since_1901()
        df_major = pd.DataFrame(major_records)
        major_csv_path = "data/processed/major_roof_falls_1901_2022.csv"
        df_major.to_csv(major_csv_path, index=False)
        logger.info(f"Saved historical major roof falls (1901-2022): {major_csv_path} ({len(df_major)} rows)")
        
        return {
            "primary_records_count": len(df_accidents),
            "coal_count": len(coal_records),
            "noncoal_count": len(noncoal_records),
            "total_fatalities": int(df_accidents["fatalities"].sum()),
            "major_accidents_count": len(df_major),
            "csv_path": csv_path,
            "json_path": json_path,
            "major_csv_path": major_csv_path
        }


if __name__ == "__main__":
    extractor = DGMSAccidentExtractor(
        coal_pdf_path="data/raw/sanket_coal_2016_2022.pdf",
        noncoal_pdf_path="data/raw/sanket_noncoal_2016_2022.pdf"
    )
    results = extractor.run_full_pipeline()
    print("\nExtraction Summary:")
    print(json.dumps(results, indent=2))
