"""
BHOOmiKAVACH AI — Background Backend Service Manager
====================================================
Ensures the FastAPI backend is running when the application is launched
via Streamlit Community Cloud or standalone 'streamlit run'.
"""

import sys
import os
import time
import subprocess
import urllib.request
from pathlib import Path
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent


def is_backend_healthy(base_url: str, timeout: float = 1.5) -> bool:
    try:
        health_url = f"{base_url.rstrip('/')}/health"
        req = urllib.request.urlopen(health_url, timeout=timeout)
        return req.status == 200
    except Exception:
        return False


@st.cache_resource(show_spinner="Warming up BHOOmiKAVACH AI FastAPI Backend & Model Vectors...")
def auto_start_local_backend() -> bool:
    """
    Checks if the backend is running. If not, and the target is localhost,
    spawns uvicorn in a background daemon subprocess.
    """
    from frontend.api_client import resolve_backend_url
    base_url = resolve_backend_url()

    # If configured for an external cloud backend (e.g. Render, Railway, AWS), don't launch locally
    if "127.0.0.1" not in base_url and "localhost" not in base_url:
        return True

    # Check if already running
    if is_backend_healthy(base_url, timeout=1.0):
        return True

    # Spawn uvicorn in background
    cmd = [
        sys.executable, "-m", "uvicorn", "backend.app.main:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    try:
        subprocess.Popen(
            cmd,
            cwd=str(ROOT_DIR),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"[BHOOmiKAVACH] Failed to launch background backend: {e}")
        return False

    # Quick poll readiness (if uvicorn succeeds, great; otherwise in-memory client will take over instantly)
    max_wait = 5
    start_time = time.time()
    while time.time() - start_time < max_wait:
        time.sleep(1.0)
        if is_backend_healthy(base_url, timeout=1.0):
            return True

    return False

