#!/usr/bin/env python3
"""
Development Trading Startup Script
Starts all components needed for automated backtesting and paper trading
"""

import os
import sys
import subprocess
import time
import threading
import signal
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Process list to track started services
processes = []


def run_command(cmd, name):
    """Run a command and track the process"""
    print(f"Starting {name}...")
    try:
        process = subprocess.Popen(cmd, shell=True, cwd=PROJECT_ROOT)
        processes.append((process, name))
        print(f"✓ {name} started (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"✗ Failed to start {name}: {e}")
        return None


def check_database():
    """Check if PostgreSQL is running"""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            port=5999,
            database="database_v2",
            user="ominari_user",
            password="ominari_2025_secure"
        )
        conn.close()
        return True
    except:
        return False


def start_services():
    """Start all required services"""
    print("\n🚀 Starting Ominari Development Trading Environment")
    print("=" * 50)
    
    # 1. Check/Start PostgreSQL
    if not check_database():
        print("Starting PostgreSQL...")
        run_command(".venv/bin/python run_with_postgres.py", "PostgreSQL")
        time.sleep(5)  # Wait for PostgreSQL to start
    else:
        print("✓ PostgreSQL already running")
    
    # 2. Start Redis (optional but recommended)
    redis_process = run_command("redis-server --port 6379", "Redis")
    if redis_process:
        time.sleep(2)
    
    # 3. Run database migrations
    print("\nRunning database migrations...")
    subprocess.run([".venv/bin/alembic", "upgrade", "head"], cwd=PROJECT_ROOT)
    print("✓ Migrations complete")
    
    # 4. Start blockchain sync daemon
    run_command(".venv/bin/python blockchain_reader.py --daemon", "Blockchain Sync")
    time.sleep(3)
    
    # 5. Start web dashboard
    run_command(".venv/bin/python web_dashboard_real_odds.py", "Web Dashboard")
    time.sleep(3)
    
    # 6. Start automated trading system
    run_command(".venv/bin/python automated_trading_system.py", "Automated Trading")
    time.sleep(2)
    
    # 7. Start performance monitor
    run_command(".venv/bin/python trading_performance_monitor.py", "Performance Monitor")
    
    print("\n✅ All services started!")
    print("\n📊 Access Points:")
    print("  - Dashboard: http://localhost:8888")
    print("  - Performance: http://localhost:8889")
    print("  - Health: http://localhost:8888/health")
    print("\n📝 Logs:")
    print("  - Automated Trading: logs/automated_trading.log")
    print("  - Performance: logs/trading_performance.jsonl")
    print("\nPress Ctrl+C to stop all services\n")


def stop_services(signum=None, frame=None):
    """Stop all services gracefully"""
    print("\n\n🛑 Stopping all services...")
    
    for process, name in reversed(processes):
        if process and process.poll() is None:
            print(f"Stopping {name}...")
            process.terminate()
            try:
                process.wait(timeout=5)
                print(f"✓ {name} stopped")
            except subprocess.TimeoutExpired:
                process.kill()
                print(f"✗ {name} force killed")
    
    print("\n✅ All services stopped")
    sys.exit(0)


def monitor_services():
    """Monitor services and restart if needed"""
    while True:
        time.sleep(30)  # Check every 30 seconds
        
        for i, (process, name) in enumerate(processes):
            if process and process.poll() is not None:
                print(f"\n⚠️  {name} crashed! Restarting...")
                
                # Restart the service
                if name == "PostgreSQL":
                    cmd = ".venv/bin/python run_with_postgres.py"
                elif name == "Redis":
                    cmd = "redis-server --port 6379"
                elif name == "Blockchain Sync":
                    cmd = ".venv/bin/python blockchain_reader.py --daemon"
                elif name == "Web Dashboard":
                    cmd = ".venv/bin/python web_dashboard_real_odds.py"
                elif name == "Automated Trading":
                    cmd = ".venv/bin/python automated_trading_system.py"
                elif name == "Performance Monitor":
                    cmd = ".venv/bin/python trading_performance_monitor.py"
                else:
                    continue
                
                new_process = run_command(cmd, name)
                if new_process:
                    processes[i] = (new_process, name)


def main():
    """Main entry point"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, stop_services)
    signal.signal(signal.SIGTERM, stop_services)
    
    # Create necessary directories
    os.makedirs("logs", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    # Start all services
    start_services()
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_services, daemon=True)
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_services()


if __name__ == "__main__":
    main()