#!/usr/bin/env python3
"""Update trading system to use cached markets"""

import os
import json
from datetime import datetime, timezone

# Load the real soccer markets we found
with open('real_soccer_markets.json', 'r') as f:
    soccer_markets = json.load(f)

print(f"Loaded {len(soccer_markets)} real soccer markets from cache")

# Convert back to proper format with datetime objects
from datetime import datetime
for market in soccer_markets:
    market['maturity_date'] = datetime.fromisoformat(market['maturity_date'])

# Take first 20 for testing
test_markets = soccer_markets[:20]

print("\nTest markets:")
for i, market in enumerate(test_markets[:5]):
    print(f"{i+1}. {market['home_team']} vs {market['away_team']}")
    print(f"   Time: {market['maturity_date']}")
    print(f"   Odds: {market['odds']}")

# Now manually trigger a trade
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from portfolio_trading_engine import PortfolioTradingEngine
from edge_calculator import EdgeCalculator

# Initialize
session_manager = PaperTradingSessionManager()
edge_calculator = EdgeCalculator()
session_id = session_manager.get_current_session()

if session_id:
    print(f"\nUsing session: {session_id}")
    
    # Get session details
    session = session_manager.get_session(session_id)
    
    strategy_config = {
        'bankroll': float(session['current_bankroll']),
        'kelly_fraction': 0.25,
        'min_edge': 0.01,  # Lower threshold for testing
        'cap_per_bet': 0.02,
        'cap_per_game': 0.02,
        'min_bet': 10,
        'max_positions': 20
    }
    
    portfolio_engine = PortfolioTradingEngine(session_manager, edge_calculator, strategy_config)
    
    # Calculate edges
    print("\nCalculating edges...")
    signals = edge_calculator.calculate_edges(test_markets)
    
    # Execute trades
    print("\nExecuting portfolio optimization...")
    result = portfolio_engine.execute_kelly_portfolio_optimization(session_id, test_markets, signals)
    
    if result['success']:
        print(f"\n✅ Success! {len(result.get('trades', []))} trades executed")
        if result.get('trades'):
            for trade in result['trades'][:3]:
                print(f"   {trade['home_team']} vs {trade['away_team']} - {trade['bet_on'].upper()} @ {trade['odds']:.2f} - ${trade['stake']:.2f}")
    else:
        print(f"\n❌ Failed: {result.get('error', 'Unknown error')}")
        print(f"Message: {result.get('message', '')}")