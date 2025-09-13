#!/usr/bin/env python3
"""Show what the match dashboard totals should display."""

import requests

response = requests.get("http://localhost:8888/api/dashboard/unified")
data = response.json()

positions = data.get('positions', {})
closed_positions = positions.get('closed', [])

print("EXPECTED MATCH DASHBOARD TOTALS")
print("=" * 60)
print(f"Total positions shown: {len(closed_positions)}")
print()

# Calculate totals by outcome
outcomes = {'Home': {'stake': 0, 'pnl': 0, 'count': 0},
            'Draw': {'stake': 0, 'pnl': 0, 'count': 0},
            'Away': {'stake': 0, 'pnl': 0, 'count': 0}}

for pos in closed_positions:
    outcome = pos.get('outcome', 'Unknown')
    exec_stake = pos.get('execution_stake', pos.get('stake', 0))
    pnl = pos.get('pnl', 0)
    
    if outcome in outcomes:
        outcomes[outcome]['stake'] += exec_stake
        outcomes[outcome]['pnl'] += pnl
        outcomes[outcome]['count'] += 1

# Print by outcome
print("By Outcome:")
for outcome, data in outcomes.items():
    if data['count'] > 0:
        print(f"  {outcome}: ${data['stake']:,.2f} stake, ${data['pnl']:,.2f} P&L ({data['count']} positions)")

# Grand totals
total_stake = sum(d['stake'] for d in outcomes.values())
total_pnl = sum(d['pnl'] for d in outcomes.values())

print(f"\nGRAND TOTALS:")
print(f"  Total: ${total_stake:,.2f}")
print(f"  Result: ${total_pnl:,.2f}")
print()
print("These totals should appear in the FIRST ROW of the match dashboard table.")