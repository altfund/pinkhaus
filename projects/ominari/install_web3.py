#!/usr/bin/env python3
"""Check if we can install web3"""
import subprocess
import sys

try:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'web3', '--user'])
    print("✅ web3 installed successfully")
except:
    print("❌ Could not install web3")