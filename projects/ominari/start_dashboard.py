#!/usr/bin/env python3
"""
Start dashboard with proper process management
"""

import os
import subprocess
import time
import socket
import sys

def check_port(port):
    """Check if port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))
    sock.close()
    return result != 0  # True if available

def start_dashboard():
    """Start the dashboard using nohup"""
    port = 8889
    
    print(f"🚀 Starting Ominari Dashboard on port {port}...")
    
    # Kill any existing processes
    print("Cleaning up existing processes...")
    subprocess.run(['pkill', '-f', 'web_monitor_unified'], capture_output=True)
    subprocess.run(['pkill', '-f', 'python.*8889'], capture_output=True)
    time.sleep(2)
    
    # Check port
    if not check_port(port):
        print(f"❌ Port {port} is still in use after cleanup")
        return False
        
    print(f"✅ Port {port} is available")
    
    # Set environment
    env = os.environ.copy()
    env.update({
        'PG_HOST': 'localhost',
        'PG_PORT': '5999',
        'PG_USER': 'ominari_user',
        'PG_PASSWORD': 'ominari_2025_secure',
        'PG_DB': 'ominari_production',
        'USE_POSTGRESQL': '1',
        'DASHBOARD_PORT': str(port),
        'PYTHONUNBUFFERED': '1'
    })
    
    print("\n✅ Features enabled:")
    print("   - ⚽ Soccer-only filter (locked)")
    print("   - 📝 Paper trading mode")
    print("   - 🎯 Underdog betting strategy")
    print("   - 📈 Real-time market data")
    
    # Create start command
    cmd = [
        'nohup',
        sys.executable,
        'web_monitor_unified.py'
    ]
    
    # Start with nohup
    print(f"\n🌐 Starting dashboard at http://localhost:{port}")
    print("📋 Logs will be in nohup.out")
    
    with open('nohup.out', 'w') as out:
        proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True  # Detach from terminal
        )
    
    # Wait a bit and check if it started
    time.sleep(3)
    
    if proc.poll() is None:  # Still running
        print("\n✅ Dashboard started successfully!")
        print(f"   PID: {proc.pid}")
        print(f"   URL: http://localhost:{port}")
        print("\n📋 Commands:")
        print("   View logs: tail -f nohup.out")
        print(f"   Stop dashboard: kill {proc.pid}")
        print("   Check status: ps -p " + str(proc.pid))
        
        # Save PID
        with open('dashboard.pid', 'w') as f:
            f.write(str(proc.pid))
            
        return True
    else:
        print("\n❌ Dashboard failed to start")
        print("Check nohup.out for errors")
        return False

if __name__ == "__main__":
    # Change to the project directory
    os.chdir('/home/ess/Documents/apps/ominari/projects/ominari')
    
    # Activate flox environment
    activate_cmd = 'source .flox/run/x86_64-linux.ominari.dev/activate'
    subprocess.run(activate_cmd, shell=True, executable='/bin/bash')
    
    start_dashboard()