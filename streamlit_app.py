"""
BHOOmiKAVACH AI — Streamlit Community Cloud Entrypoint
======================================================
Official entry point for deploying BHOOmiKAVACH AI on Streamlit Community Cloud (share.streamlit.io).
Automatically sets up execution paths and launches the industrial decision-support platform.

Repository: https://github.com/Kaustubhsudhakar07/BhoomiKavach
DGMS Accident Code 0111: Fall of Roof
Standards: CMR 2017 & MMR 1961
"""

import sys
import runpy
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main Streamlit application
app_path = ROOT_DIR / "frontend" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
