#!/usr/bin/env python3
"""Test all dashboard data is piping through correctly."""

import requests
import json

print("=== TESTING ALL DASHBOARD DATA FLOW ===\n")

# Get the unified API data
response = requests.get("http://localhost:8888/api/dashboard/unified")
data = response.json()

# Test each section
sections = {
    'Portfolio': data.get('portfolio', {}),
    'Performance': data.get('performance', {}),
    'System': data.get('system', {}),
    'Markets': data.get('markets', []),
    'Activity': data.get('activity', []),
    'Positions': data.get('positions', {}),
    'Strategy': data.get('strategy', {})
}

print("1. DATA AVAILABILITY CHECK:")
for section, content in sections.items():
    if isinstance(content, dict):
        status = "✓" if content else "✗"
        count = len(content.keys())
        print(f"   {status} {section}: {count} fields")
    elif isinstance(content, list):
        status = "✓" if content else "✗"
        print(f"   {status} {section}: {len(content)} items")

# Detailed checks
print("\n2. PORTFOLIO DATA:")
portfolio = data.get('portfolio', {})
print(f"   Total Value: ${portfolio.get('total_value', 0):.2f}")
print(f"   Cash Available: ${portfolio.get('cash_available', 0):.2f}")
print(f"   Positions Value: ${portfolio.get('positions_value', 0):.2f}")
print(f"   Daily Change: ${portfolio.get('daily_change', 0):.2f} ({portfolio.get('daily_change_pct', 0):.1f}%)")
print(f"   Positions Count: {portfolio.get('positions_count', 0)}")
print(f"   Total Exposure: ${portfolio.get('total_exposure', 0):.2f}")

print("\n3. PERFORMANCE DATA:")
perf = data.get('performance', {})
print(f"   Win Rate: {perf.get('win_rate', 0) * 100:.1f}%")
print(f"   ROI: {perf.get('roi', 0):.1f}%")
print(f"   Total P&L: ${perf.get('total_pnl', 0):.2f}")
print(f"   Total Stake: ${perf.get('total_stake', 0):.2f}")
print(f"   Winning Trades: {perf.get('winning_trades', 0)}")
print(f"   Losing Trades: {perf.get('losing_trades', 0)}")
print(f"   Pending Trades: {perf.get('pending_trades', 0)}")

print("\n4. SYSTEM DATA:")
system = data.get('system', {})
print(f"   Session ID: {system.get('session', {}).get('id', 'None')}")
print(f"   Session Status: {system.get('session', {}).get('status', 'inactive')}")
print(f"   Database Markets: {system.get('database', {}).get('markets', 0)}")
print(f"   Database Odds: {system.get('database', {}).get('odds', 0)}")

print("\n5. MARKETS DATA:")
markets = data.get('markets', [])
print(f"   Total Markets: {len(markets)}")
if markets:
    print("   First 3 markets:")
    for m in markets[:3]:
        print(f"     - {m.get('home_team')} vs {m.get('away_team')}")
        print(f"       Odds count: {len(m.get('odds', []))}")

print("\n6. POSITIONS DATA:")
positions = data.get('positions', {})
open_pos = positions.get('open', [])
closed_pos = positions.get('closed', [])
print(f"   Open Positions: {len(open_pos)}")
if open_pos:
    print("   Open positions details:")
    for pos in open_pos[:3]:
        print(f"     - {pos.get('market_name')} [{pos.get('outcome')}]")
        print(f"       Stake: ${pos.get('stake', 0):.2f}, P&L: ${pos.get('pnl', 0):.2f}")
        
print(f"   Closed Positions: {len(closed_pos)}")
if closed_pos:
    print("   Recent closed positions:")
    for pos in closed_pos[:3]:
        print(f"     - {pos.get('market_name')} [{pos.get('outcome')}]: {pos.get('result')}")

print("\n7. ACTIVITY FEED:")
activity = data.get('activity', [])
print(f"   Total Events: {len(activity)}")
if activity:
    print("   Recent activity:")
    for act in activity[:5]:
        print(f"     - [{act.get('type')}] {act.get('message')[:50]}...")

print("\n8. STRATEGY PARAMETERS:")
strategy = data.get('strategy', {})
print(f"   Kelly Fraction: {strategy.get('kelly_fraction', 0) * 100:.0f}%")
print(f"   Bankroll: ${strategy.get('bankroll', 0):.2f}")
print(f"   Min Bet: ${strategy.get('min_bet', 0):.2f}")
print(f"   Cap Per Game: {strategy.get('cap_per_game', 0) * 100:.0f}%")

# Check for any error fields
if 'error' in data:
    print(f"\n⚠️  API ERROR: {data['error']}")
else:
    print("\n✅ All data sections successfully retrieved!")
    
# Summary
empty_sections = []
for section, content in sections.items():
    if isinstance(content, (dict, list)) and not content:
        empty_sections.append(section)
        
if empty_sections:
    print(f"\n⚠️  Empty sections: {', '.join(empty_sections)}")
    print("   These sections need data to be populated.")
else:
    print("\n✅ All sections contain data!")

print("\n=== END OF DATA FLOW TEST ===")