"""
BHOOmiKAVACH AI — Unified Services Launcher
============================================
Launches both the FastAPI backend server and the Streamlit decision-support UI
with synchronized lifecycle management, health monitoring, and graceful shutdown.

Usage:
    python run_bhoomikavach.py              # Launch both Backend & Frontend
    python run_bhoomikavach.py --backend    # Launch FastAPI Backend only
    python run_bhoomikavach.py --frontend   # Launch Streamlit Frontend only
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request


def check_backend_health(url="http://127.0.0.1:8000/api/v1/health", max_retries=15, interval=1.0):
    """Wait until backend responds with 200 OK."""
    print(f"[*] Awaiting FastAPI backend readiness at {url}...")
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.urlopen(url, timeout=2.0)
            if req.status == 200:
                print(f"[OK] Backend is healthy and ready (attempt {attempt}/{max_retries}).")
                return True
        except Exception:
            pass
        time.sleep(interval)
    print(f"[WARNING] Backend did not respond within {max_retries * interval:.0f}s. Proceeding anyway.")
    return False


def main():
    parser = argparse.ArgumentParser(description="BHOOmiKAVACH AI Unified Launcher")
    parser.add_argument("--backend", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--frontend", action="store_true", help="Launch Streamlit frontend only")
    parser.add_argument("--backend-port", type=int, default=8000, help="Backend port (default: 8000)")
    parser.add_argument("--frontend-port", type=int, default=8501, help="Frontend port (default: 8501)")
    args = parser.parse_args()

    launch_both = not (args.backend or args.frontend)
    launch_backend = launch_both or args.backend
    launch_frontend = launch_both or args.frontend

    python_exe = sys.executable
    processes = []

    print("=" * 70)
    print("  BHOOmiKAVACH AI: Roof-Fall Risk Decision Support System")
    print("  'SENSE THE RISK. EXPLAIN THE DANGER. PREVENT THE FALL.'")
    print("=" * 70)

    try:
        if launch_backend:
            print(f"[*] Starting FastAPI Backend on http://127.0.0.1:{args.backend_port}...")
            backend_cmd = [
                python_exe, "-m", "uvicorn", "backend.app.main:app",
                "--host", "127.0.0.1",
                "--port", str(args.backend_port)
            ]
            p_back = subprocess.Popen(backend_cmd)
            processes.append(("FastAPI Backend", p_back))

            if launch_frontend:
                check_backend_health(f"http://127.0.0.1:{args.backend_port}/api/v1/health")

        if launch_frontend:
            print(f"[*] Starting Streamlit Frontend on http://127.0.0.1:{args.frontend_port}...")
            frontend_cmd = [
                python_exe, "-m", "streamlit", "run", "frontend/app.py",
                "--server.port", str(args.frontend_port),
                "--server.address", "127.0.0.1",
                "--browser.gatherUsageStats", "false"
            ]
            p_front = subprocess.Popen(frontend_cmd)
            processes.append(("Streamlit Frontend", p_front))

        print("\n" + "=" * 70)
        if launch_backend:
            print(f"  * FastAPI Backend:   http://127.0.0.1:{args.backend_port}")
            print(f"  * Interactive Docs:  http://127.0.0.1:{args.backend_port}/docs")
        if launch_frontend:
            print(f"  * Streamlit UI:      http://127.0.0.1:{args.frontend_port}")
        print("=" * 70)
        print("  Press Ctrl+C to terminate all services.\n")

        # Keep running and monitor child processes
        while True:
            for name, proc in processes:
                ret = proc.poll()
                if ret is not None:
                    print(f"[{name}] Process exited with code {ret}.")
                    raise KeyboardInterrupt
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[*] Stopping all BHOOmiKAVACH AI services...")
        for name, proc in processes:
            if proc.poll() is None:
                print(f"[*] Terminating {name}...")
                proc.terminate()
                try:
                    proc.wait(timeout=4)
                except subprocess.TimeoutExpired:
                    proc.kill()
        print("[OK] All services cleanly stopped. Exiting.")


if __name__ == "__main__":
    main()
