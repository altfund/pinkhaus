#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 02:56:17 2025

@author: ess
"""

from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys

PYTHON_EXECUTABLE = sys.executable  # This will resolve to the full path

import os
import time

WATCHED_FILES = [
    "match_markets.py",
    "get_oracle_odds.py",
    "find_opportunities.py",
    "evaluate_open_markets.py",
    "run_scheduler.sh"
]

FILE_HASHES = {}

def file_changed(file_path):
    try:
        with open(file_path, "rb") as f:
            new_hash = hash(f.read())
        old_hash = FILE_HASHES.get(file_path)
        FILE_HASHES[file_path] = new_hash
        return old_hash is not None and new_hash != old_hash
    except Exception as e:
        print(f"Could not check {file_path}: {e}")
        return False

def watch_for_changes_and_restart(base_path="."):
    changed = any(file_changed(os.path.join(base_path, f)) for f in WATCHED_FILES)
    if changed:
        print("🔄 Detected script changes — restarting scheduler...")
        restart_scheduler_service()


def run_script(script_name):
    script_path = SCRIPT_DIR / script_name
    print(f"Running: {script_name}")
    result = subprocess.run([PYTHON_EXECUTABLE, str(script_path)], capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Error:", result.stderr)
    update_last_run(script_name)

def restart_scheduler_service():
    """Restart the systemd user service to reflect updates to scripts/configs."""
    try:
        subprocess.run(["systemctl", "--user", "daemon-reexec"], check=True)
        subprocess.run(["systemctl", "--user", "restart", "altfund_scheduler.service"], check=True)
        print("🔁 altfund_scheduler.service restarted.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to restart service: {e}")

# Configuration of scripts with run intervals (in minutes) and dependencies
SCRIPT_CONFIG = {
    "free_data_pull.py": {"interval": 5, "depends_on": []},
    "match_markets.py": {"interval": 60, "depends_on": ["free_data_pull.py"]},
    "get_oracle_odds.py": {"interval": 60*12, "depends_on": ["match_markets.py"]},
    "find_opportunities.py": {"interval": 60*12, "depends_on": ["get_oracle_odds.py"]},
    "evaluate_open_markets.py": {"interval": 30, "depends_on": ["free_data_pull.py"]},
    "db_inspector.py": {"interval": 60, "depends_on": []}
}

# Define script directory
SCRIPT_DIR = Path("/home/ess/Documents/apps/ominari")

# Path to store last run times
LAST_RUN_PATH = SCRIPT_DIR / "last_run_times.json"

# Load or initialize last run times
if LAST_RUN_PATH.exists():
    with open(LAST_RUN_PATH, "r") as f:
        last_run_times = json.load(f)
else:
    last_run_times = {}
    
LOG_FILE = SCRIPT_DIR / "script_scheduler.log"

def log_message(message):
    with open(LOG_FILE, "a") as log:
        log.write(f"[{datetime.utcnow().isoformat()}] {message}\n")


def get_last_run(script_name):
    ts = last_run_times.get(script_name)
    return datetime.fromisoformat(ts) if ts else None

def update_last_run(script_name):
    last_run_times[script_name] = datetime.utcnow().isoformat()
    with open(LAST_RUN_PATH, "w") as f:
        json.dump(last_run_times, f, indent=2)

def is_ready_to_run(script_name):
    config = SCRIPT_CONFIG[script_name]
    last_run = get_last_run(script_name)
    if last_run is None:
        return True
    next_run_time = last_run + timedelta(minutes=config["interval"])
    return datetime.utcnow() >= next_run_time


def resolve_and_run(script_name, visited=None):
    if visited is None:
        visited = set()

    if script_name in visited:
        return
    visited.add(script_name)

    # Recursively ensure dependencies are met
    for dep in SCRIPT_CONFIG[script_name]["depends_on"]:
        if is_ready_to_run(dep):
            resolve_and_run(dep, visited)

    # Run if it's due and dependencies are up to date
    if is_ready_to_run(script_name):
        run_script(script_name)

# Example usage: resolve and run all scripts
for script in SCRIPT_CONFIG:
    resolve_and_run(script)

watch_for_changes_and_restart("/home/ess/Documents/apps/ominari")
