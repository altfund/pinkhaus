#\!/usr/bin/env python3
"""Verify that the dashboard displays paper trading data correctly."""

import requests
import json
from datetime import datetime

print("VERIFYING DASHBOARD DISPLAY")
print("=" * 80)

# Check different endpoints
endpoints = [
    ("Unified Dashboard", "http://localhost:8888/api/dashboard/unified"),
    ("Closed Positions", "http://localhost:8888/api/trading/closed_positions?limit=3"),
    ("Trading Portfolio", "http://localhost:8888/api/trading/portfolio"),
    ("Performance", "http://localhost:8888/api/performance")
]

for name, url in endpoints:
    print(f"\n{name}: {url}")
    print("-" * 60)
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if "closed_positions" in url:
                # Check closed positions
                positions = data.get('positions', [])
                summary = data.get('summary', {})
                
                print(f"Total closed: {summary.get('total_count', 0)}")
                print(f"Total P&L: ${summary.get('total_pnl', 0):.2f}")
                print(f"Win rate: {summary.get('win_rate', 0):.1f}%")
                
                if positions:
                    print("\nSample positions:")
                    for i, pos in enumerate(positions[:3]):
                        print(f"\n{i+1}. {pos.get('market_name', 'Unknown')}")
                        print(f"   Outcome: {pos.get('outcome')} -> {pos.get('result')}")
                        print(f"   Score: {pos.get('score')} ({pos.get('final_outcome')})")
                        print(f"   P&L: ${pos.get('pnl', 0):.2f}")
                        
            elif "portfolio" in url:
                # Check portfolio data
                print(f"Total value: ${data.get('total_value', 0):.2f}")
                print(f"Cash: ${data.get('cash_available', 0):.2f}")
                print(f"Positions value: ${data.get('positions_value', 0):.2f}")
                print(f"Daily change: ${data.get('daily_change', 0):.2f} ({data.get('daily_change_pct', 0):.1f}%)")
                
            elif "unified" in url:
                # Check positions in unified dashboard
                positions = data.get('positions', {})
                print(f"Open positions: {len(positions.get('open', []))}")
                print(f"Closed positions: {len(positions.get('closed', []))}")
                
                # Check if closed positions have results
                closed = positions.get('closed', [])
                if closed:
                    with_results = sum(1 for p in closed if p.get('score') != '-')
                    print(f"Closed with results: {with_results}/{len(closed)}")
                    
                    # Show a sample
                    for pos in closed[:2]:
                        print(f"\nSample: {pos.get('match')}")
                        print(f"  Score: {pos.get('score')}")
                        print(f"  Result: {pos.get('outcome_display', '-')}")
                        print(f"  P&L: {pos.get('pnl_display', '-')}")
                        
            else:
                print(f"Response keys: {list(data.keys())[:10]}")
                
        else:
            print(f"ERROR: Status code {response.status_code}")
            print(response.text[:200])
            
    except requests.RequestException as e:
        print(f"ERROR: {e}")
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON response")
        print(f"Response: {response.text[:200]}")

print("\n" + "=" * 80)
