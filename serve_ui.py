from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os
import subprocess
import sys
import webbrowser


HOST = "127.0.0.1"
START_PORT = 8001
PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_DATA_FILES = [
    PROJECT_ROOT / "dataset" / "network_snapshots.json",
    PROJECT_ROOT / "dataset" / "rl_rpl_decisions.csv",
    PROJECT_ROOT / "dataset" / "rl_rpl_round_metrics.csv",
]


def ensure_data_files() -> None:
    if all(path.exists() for path in REQUIRED_DATA_FILES):
        return
    print("Dashboard data not found. Running visual_simulation.py first...")
    subprocess.run([sys.executable, "visual_simulation.py"], cwd=PROJECT_ROOT, check=True)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    ensure_data_files()
    server = None
    port = START_PORT
    for candidate_port in range(START_PORT, START_PORT + 20):
        try:
            server = ThreadingHTTPServer((HOST, candidate_port), SimpleHTTPRequestHandler)
            port = candidate_port
            break
        except OSError:
            continue

    if server is None:
        raise RuntimeError("Could not find an available local port for the dashboard.")

    url = f"http://{HOST}:{port}/ui/index.html"
    print(f"RL-RPL dashboard is running at {url}")
    webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
