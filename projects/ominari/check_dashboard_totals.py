#!/usr/bin/env python3
"""Check match dashboard totals."""

import requests
import json

# Get closed positions from API
response = requests.get("http://localhost:8888/api/trading/closed_positions?limit=200")
data = response.json()

positions = data.get('positions', [])
summary = data.get('summary', {})

print("MATCH DASHBOARD TOTALS")
print("=" * 60)
print(f"Summary from API: {summary}")
print(f"Total positions: {len(positions)}")

# Calculate totals
total_stake = 0
total_exec_stake = 0
total_pnl = 0
total_fees = 0

for p in positions:
    stake = p.get('stake', 0)
    exec_stake = p.get('execution_stake', 0)
    pnl = p.get('pnl', 0)
    fee = exec_stake - stake if exec_stake and stake else 0
    
    total_stake += stake
    total_exec_stake += exec_stake
    total_pnl += pnl
    total_fees += fee

print(f"\nCalculated totals:")
print(f"Total stake: ${total_stake:,.2f}")
print(f"Total execution stake: ${total_exec_stake:,.2f}")
print(f"Total fees: ${total_fees:,.2f}")
print(f"Total P&L: ${total_pnl:,.2f}")

# Check if this matches what user said
print(f"\nUser mentioned: '$169 lost on $5648'")
print(f"Closest match to $5,648: ???")
print(f"Closest match to $169: ???")

# Maybe it's about specific subsets?
wins = [p for p in positions if p.get('result') == 'won']
losses = [p for p in positions if p.get('result') == 'lost']

win_stake = sum(p.get('stake', 0) for p in wins)
loss_stake = sum(p.get('stake', 0) for p in losses)
win_exec_stake = sum(p.get('execution_stake', 0) for p in wins)
loss_exec_stake = sum(p.get('execution_stake', 0) for p in losses)

print(f"\nBreakdown by outcome:")
print(f"Wins: {len(wins)} positions")
print(f"  Stake: ${win_stake:,.2f}")
print(f"  Execution stake: ${win_exec_stake:,.2f}")
print(f"Losses: {len(losses)} positions") 
print(f"  Stake: ${loss_stake:,.2f}")
print(f"  Execution stake: ${loss_exec_stake:,.2f}")

# Check open positions too
open_response = requests.get("http://localhost:8888/api/dashboard/unified")
open_data = open_response.json()
open_positions = open_data.get('positions', {}).get('open', [])

open_stake = sum(p.get('stake', 0) for p in open_positions)
print(f"\nOpen positions: {len(open_positions)}")
print(f"Open stake at risk: ${open_stake:,.2f}")