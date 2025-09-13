#!/usr/bin/env python3
"""Verify the dashboard fixes show all positions with proper totals."""

import requests
import json

print("VERIFYING DASHBOARD FIXES")
print("=" * 80)

# Check unified dashboard
response = requests.get("http://localhost:8888/api/dashboard/unified")
data = response.json()

positions = data.get('positions', {})
open_positions = positions.get('open', [])
closed_positions = positions.get('closed', [])

print(f"Open positions: {len(open_positions)}")
print(f"Closed positions: {len(closed_positions)}")

# Calculate totals using execution_stake
total_stake = 0
total_exec_stake = 0
total_pnl = 0
total_fees = 0

print("\nCalculating totals with execution_stake...")
print("-" * 60)

# Process closed positions
for pos in closed_positions:
    stake = pos.get('stake', 0)
    exec_stake = pos.get('execution_stake', stake)
    pnl = pos.get('pnl', 0)
    fee = exec_stake - stake if exec_stake > stake else 0
    
    total_stake += stake
    total_exec_stake += exec_stake
    total_pnl += pnl
    total_fees += fee

print(f"Total stake (base): ${total_stake:,.2f}")
print(f"Total execution stake: ${total_exec_stake:,.2f}")
print(f"Total fees: ${total_fees:,.2f}")
print(f"Total P&L: ${total_pnl:,.2f}")

# Check if this matches what dashboard should show
print("\nExpected dashboard display:")
print(f"- Should show {len(closed_positions)} closed positions (not just 20)")
print(f"- Total column should show: ${total_exec_stake:,.2f} (including fees)")
print(f"- Result/P&L should show: ${total_pnl:,.2f}")
print(f"- Totals row should be at the TOP of the table")

# Check markets too
markets = data.get('markets', [])
print(f"\nMarkets returned: {len(markets)}")

# Breakdown by result
wins = [p for p in closed_positions if p.get('result') == 'won']
losses = [p for p in closed_positions if p.get('result') == 'lost']

win_exec_stake = sum(p.get('execution_stake', p.get('stake', 0)) for p in wins)
loss_exec_stake = sum(p.get('execution_stake', p.get('stake', 0)) for p in losses)
win_pnl = sum(p.get('pnl', 0) for p in wins)
loss_pnl = sum(p.get('pnl', 0) for p in losses)

print(f"\nWin/Loss breakdown:")
print(f"Wins: {len(wins)} positions")
print(f"  Execution stake: ${win_exec_stake:,.2f}")
print(f"  P&L: ${win_pnl:,.2f}")
print(f"Losses: {len(losses)} positions")
print(f"  Execution stake: ${loss_exec_stake:,.2f}")
print(f"  P&L: ${loss_pnl:,.2f}")