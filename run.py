import sys
import time
import threading
import argparse
import webbrowser
import subprocess

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid Face Recognition launcher")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--reload", action="store_true")
    return p.parse_args()

def open_browser(url: str, delay: float = 3):
    def _open():
        time.sleep(delay)
        print(f"🌐 Opening browser → {url}")
        webbrowser.open(url)
    threading.Thread(target=_open, daemon=True).start()

def main():
    args = parse_args()
    url = f"http://localhost:{args.port}"

    print(f"🚀 Server starting...")
    print(f"Backend  : http://{args.host}:{args.port}")
    print(f"Frontend : {url}")
    print(f"Docs     : {url}/docs")

    if not args.no_browser:
        open_browser(url)

    cmd = [
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", args.host,
        "--port", str(args.port),
    ]

    if args.reload:
        cmd.append("--reload")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")

if __name__ == "__main__":
    main()