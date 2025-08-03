import subprocess
from pathlib import Path
from datetime import datetime

APP_DIR = Path(__file__).resolve().parent
SCHEDULER_SCRIPT = APP_DIR / "run_everything.py"
RUN_SCRIPT = APP_DIR / "run_scheduler.sh"
SERVICE_DIR = Path.home() / ".config" / "systemd" / "user"
SERVICE_FILE = SERVICE_DIR / "altfund_scheduler.service"

def detect_python_path():
    try:
        python_path = subprocess.check_output(["which", "python"]).decode().strip()
        print(f"✅ Detected Python path: {python_path}")
        return python_path
    except Exception as e:
        raise RuntimeError("❌ Failed to detect python path using `which python`.") from e

def create_run_script(python_path: str):
    content = f"""#!/bin/bash
cd "{APP_DIR}"
"{python_path}" "{SCHEDULER_SCRIPT.name}" >> scheduler_output.log 2>> scheduler_error.log
"""
    RUN_SCRIPT.write_text(content)
    RUN_SCRIPT.chmod(0o755)
    print(f"✅ Created: {RUN_SCRIPT}")

def create_systemd_service():
    SERVICE_DIR.mkdir(parents=True, exist_ok=True)
    content = f"""[Unit]
Description=altfund Script Scheduler
After=network.target

[Service]
Type=simple
ExecStart={RUN_SCRIPT}
Restart=always
RestartSec=60

[Install]
WantedBy=default.target
"""
    SERVICE_FILE.write_text(content)
    print(f"✅ Created: {SERVICE_FILE}")

def enable_and_start_service():
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", SERVICE_FILE.name], check=True)
    subprocess.run(["systemctl", "--user", "restart", SERVICE_FILE.name], check=True)
    print("✅ Service enabled and started")

def main():
    print(f"🔧 Starting setup at {datetime.now()}")
    python_path = detect_python_path()
    create_run_script(python_path)
    create_systemd_service()
    enable_and_start_service()
    print("🚀 Setup complete. Use `systemctl --user status altfund_scheduler.service` to check status.")

if __name__ == "__main__":
    main()
