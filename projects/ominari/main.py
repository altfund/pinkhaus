#!/usr/bin/env python3
"""
Master startup script for Ominari Trading System
Starts all components in the correct order with proper environment
This is THE MAIN ENTRY POINT - just run this!
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

# Set up environment
os.environ['PG_PORT'] = '5999'
os.environ['PG_DB'] = 'ominari_production'
os.environ['DATABASE_URL'] = 'postgresql://ominari_user:ominari_2025_secure@localhost:5999/ominari_production'

# Track processes for cleanup
processes = []

def cleanup(signum=None, frame=None):
    """Clean shutdown of all processes"""
    print("\n\nShutting down Ominari Trading System...")
    for proc, name in reversed(processes):
        if proc and proc.poll() is None:
            print(f"Stopping {name}...")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    print("Shutdown complete")
    sys.exit(0)

# Set up signal handlers
signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)

print("🚀 Starting Ominari Trading System")
print("=" * 50)

# 1. Check/Start PostgreSQL
print("Checking PostgreSQL...")
try:
    subprocess.run(['psql', '-h', 'localhost', '-p', '5999', '-U', 'ominari_user', '-d', 'ominari_production', '-c', 'SELECT 1'], 
                   capture_output=True, check=True, env={**os.environ, 'PGPASSWORD': 'ominari_2025_secure'})
    print("✓ PostgreSQL is running")
except:
    print("Starting PostgreSQL...")
    pg_proc = subprocess.Popen([sys.executable, 'run_with_postgres.py'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    processes.append((pg_proc, "PostgreSQL"))
    time.sleep(5)

# 2. Run migrations
print("\nRunning database migrations...")
try:
    subprocess.run(['.venv/bin/alembic', 'upgrade', 'head'], check=True)
    print("✓ Migrations complete")
except Exception as e:
    print(f"⚠️  Migration warning: {e}")

# 3. Start main dashboard (which now auto-starts trading)
print("\nStarting Ominari Dashboard with integrated trading...")
dashboard_proc = subprocess.Popen(
    [sys.executable, 'web_dashboard_real_odds.py'],
    env=os.environ.copy()
)
processes.append((dashboard_proc, "Dashboard + Trading"))

print("\n✅ Ominari Trading System Started!")
print("\n📊 Access Points:")
print("  - Main Dashboard: http://localhost:8888")
print("  - Performance Monitor: http://localhost:8889")
print("  - Health Check: http://localhost:8888/health")
print("\n📈 Features:")
print("  - ✓ Real-time market data with edge calculation")
print("  - ✓ Automated paper trading with $10,000 starting bankroll")
print("  - ✓ Blockchain data synchronization")
print("  - ✓ Performance tracking and analytics")
print("  - ✓ Hourly portfolio updates via Discord")
print("\nPress Ctrl+C to stop all services\n")

# Keep main process alive
try:
    while True:
        time.sleep(1)
        # Check if dashboard is still running
        if dashboard_proc.poll() is not None:
            print("⚠️  Dashboard stopped unexpectedly!")
            cleanup()
except KeyboardInterrupt:
    cleanup()

if __name__ == "__main__":
    # This is now the MAIN entry point
    pass