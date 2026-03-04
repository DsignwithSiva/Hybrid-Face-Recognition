"""
Hybrid Face Recognition — Unified Launcher
==========================================
Starts the FastAPI backend and opens the frontend in your browser.

Usage:
    python run.py
    python run.py --port 8080
    python run.py --no-browser
"""

import sys
import os
import time
import threading
import argparse
import webbrowser
import subprocess

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Face Recognition launcher")
    p.add_argument("--port",       type=int, default=8000, help="Port to run on (default: 8000)")
    p.add_argument("--host",       default="0.0.0.0",    help="Host to bind (default: 0.0.0.0)")
    p.add_argument("--no-browser", action="store_true",  help="Don't open browser automatically")
    p.add_argument("--reload",     action="store_true",  help="Enable auto-reload (dev mode)")
    return p.parse_args()


def open_browser(url: str, delay: float = 3.5):
    """Open default browser after a short delay (lets server start first)."""
    def _open():
        time.sleep(delay)
        print(f"\n🌐  Opening browser → {url}")
        webbrowser.open(url)
    t = threading.Thread(target=_open, daemon=True)
    t.start()


def main():
    args = parse_args()
    url  = f"http://localhost:{args.port}"
    print(f"  Backend  : http://{args.host}:{args.port}")
    print(f"  Frontend : {url}")
    print(f"  API docs : {url}/docs")

    if not args.no_browser:
        open_browser(url, delay=4.0)

    # Build uvicorn command
    cmd = [
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", args.host,
        "--port", str(args.port),
        "--log-level", "info",
    ]
    if args.reload:
        cmd.append("--reload")

    # Run in the same process (blocks until Ctrl+C)
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print("\n\n👋  Server stopped. Goodbye!\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
