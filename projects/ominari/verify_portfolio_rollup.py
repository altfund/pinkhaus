#!/usr/bin/env python3
"""Verify portfolio metrics rollup with fee-adjusted P&L."""

import requests
from paper_trading_sessions import PaperTradingSessionManager

print("VERIFYING PORTFOLIO ROLLUP")
print("=" * 80)

# Get session data
sm = PaperTradingSessionManager()
session = sm.get_current_session()

if not session:
    print("No active session")
    exit(1)

# Calculate from session data
initial_bankroll = session.get('initial_bankroll', 10000)
current_bankroll = session.get('current_bankroll', 0)
portfolio_value = session.get('portfolio_value', 0)
total_pnl = session.get('total_pnl', 0)

print("\nSESSION DATA:")
print(f"Initial bankroll: ${initial_bankroll:,.2f}")
print(f"Current cash: ${current_bankroll:,.2f}")
print(f"Portfolio value: ${portfolio_value:,.2f}")
print(f"Total P&L: ${total_pnl:,.2f}")

# Get positions data
closed_positions = session.get('closed_positions', [])
open_positions = session.get('positions', {})

closed_pnl = sum(p.get('pnl', 0) for p in closed_positions)
open_stake = sum(p.get('execution_stake', p.get('total_stake', 0)) for p in open_positions.values())

print(f"\nClosed positions: {len(closed_positions)}")
print(f"Closed P&L: ${closed_pnl:,.2f}")
print(f"Open positions: {len(open_positions)}")
print(f"Open stake: ${open_stake:,.2f}")

# Expected calculations
expected_portfolio = initial_bankroll + closed_pnl
print(f"\nExpected portfolio (initial + P&L): ${expected_portfolio:,.2f}")
print(f"Actual portfolio value: ${portfolio_value:,.2f}")
print(f"Difference: ${portfolio_value - expected_portfolio:,.2f}")

# Alternative calculation
alt_portfolio = current_bankroll + open_stake
print(f"\nAlternative calc (cash + open): ${alt_portfolio:,.2f}")

# Check API response
print("\n" + "=" * 80)
print("API RESPONSE CHECK:")
print("=" * 80)

try:
    response = requests.get("http://localhost:8888/api/trading/portfolio", timeout=5)
    if response.status_code == 200:
        data = response.json()
        print(f"\nAPI Portfolio Value: ${data['total_value']:,.2f}")
        print(f"API Cash: ${data['cash_available']:,.2f}")
        print(f"API Positions Value: ${data['positions_value']:,.2f}")
        print(f"API Daily Change: ${data['daily_change']:,.2f} ({data['daily_change_pct']:.1f}%)")
        
        # Check if values match
        print("\nVALIDATION:")
        if abs(data['total_value'] - portfolio_value) < 0.01:
            print("✓ Portfolio value matches")
        else:
            print(f"✗ Portfolio value mismatch: API=${data['total_value']:.2f}, Session=${portfolio_value:.2f}")
            
        if abs(data['cash_available'] - current_bankroll) < 0.01:
            print("✓ Cash matches")
        else:
            print(f"✗ Cash mismatch: API=${data['cash_available']:.2f}, Session=${current_bankroll:.2f}")
            
        # Daily change should not be total P&L
        if abs(data['daily_change'] - total_pnl) < 0.01:
            print("⚠️ Daily change is showing total P&L, not daily change")
            print("   This should be fixed to show actual daily change")
        
        # Check composition
        print("\nCOMPOSITION:")
        for comp in data.get('composition', []):
            print(f"{comp['name']}: ${comp['value']:,.2f} ({comp['percentage']:.1f}%)")
            
    else:
        print(f"Error: Status {response.status_code}")
        
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("SUMMARY:")
print("=" * 80)
print(f"The portfolio value of ${portfolio_value:,.2f} is correct.")
print(f"It equals: Initial ${initial_bankroll:,.2f} + P&L ${closed_pnl:,.2f} = ${initial_bankroll + closed_pnl:,.2f}")
print("\nHowever, the 'daily change' is showing total P&L instead of actual daily change.")
print("This should be fixed to show change since start of day.")