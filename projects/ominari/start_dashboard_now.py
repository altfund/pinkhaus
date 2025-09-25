#!/usr/bin/env python3
"""
Start the web monitor dashboard with the real data we have so far
"""
import os
import sys
import subprocess

# Set environment for PostgreSQL on port 5999
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production',
    'USE_POSTGRESQL': '1'
})

print("🚀 Starting Ominari Dashboard with Real Overtime Data!")
print("=" * 60)

# Check if flox has the required packages
try:
    # Try running with flox first
    print("Starting with flox environment...")
    subprocess.run(["flox", "activate", "&&", "python3", "web_monitor.py"], shell=True)
except:
    # Fall back to uv if available
    print("Starting with uv...")
    try:
        subprocess.run(["uv", "run", "python", "web_monitor.py"])
    except:
        # Last resort - direct python
        print("Starting with python directly...")
        subprocess.run([sys.executable, "web_monitor.py"])