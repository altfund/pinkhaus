#!/usr/bin/env python3
"""Test the unified dashboard data loading."""

import requests
import json

# Get the unified API data
response = requests.get("http://localhost:8888/api/dashboard/unified")
data = response.json()

print("=== UNIFIED DASHBOARD DATA ===\n")

# Portfolio
portfolio = data.get('portfolio', {})
print("PORTFOLIO:")
print(f"  Total Value: ${portfolio.get('total_value', 0):.2f}")
print(f"  Cash Available: ${portfolio.get('cash_available', 0):.2f}")
print(f"  Positions Value: ${portfolio.get('positions_value', 0):.2f}")
print(f"  Daily Change: ${portfolio.get('daily_change', 0):.2f} ({portfolio.get('daily_change_pct', 0):.1f}%)")
print(f"  Positions Count: {portfolio.get('positions_count', 0)}")
print(f"  Total Exposure: ${portfolio.get('total_exposure', 0):.2f}")

# Performance
perf = data.get('performance', {})
print("\nPERFORMANCE:")
print(f"  Win Rate: {perf.get('win_rate', 0) * 100:.1f}%")
print(f"  ROI: {perf.get('roi', 0):.1f}%")
print(f"  Total P&L: ${perf.get('total_pnl', 0):.2f}")
print(f"  Winning Trades: {perf.get('winning_trades', 0)}")
print(f"  Losing Trades: {perf.get('losing_trades', 0)}")

# System
system = data.get('system', {})
print("\nSYSTEM:")
print(f"  Session ID: {system.get('session', {}).get('id', 'None')}")
print(f"  Status: {system.get('session', {}).get('status', 'inactive')}")
print(f"  Markets: {system.get('database', {}).get('markets', 0)}")

# Positions
positions = data.get('positions', {})
print(f"\nPOSITIONS:")
print(f"  Open: {len(positions.get('open', []))}")
print(f"  Closed: {len(positions.get('closed', []))}")

# Activity
activity = data.get('activity', [])
print(f"\nACTIVITY:")
print(f"  Recent events: {len(activity)}")
if activity:
    print("  Latest 3:")
    for act in activity[:3]:
        print(f"    - [{act.get('type')}] {act.get('message')}")

# Markets
markets = data.get('markets', [])
print(f"\nMARKETS: {len(markets)} available")