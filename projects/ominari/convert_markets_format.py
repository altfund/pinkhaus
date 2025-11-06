#!/usr/bin/env python3
"""Convert markets to proper format for trading"""

import json
from datetime import datetime

# Load real soccer markets
with open('real_soccer_markets.json', 'r') as f:
    markets = json.load(f)

print(f"Converting {len(markets)} markets to trading format...")

# Convert to the format expected by portfolio trading engine
converted_markets = []

for market in markets[:50]:  # Take first 50 for testing
    # Skip if not proper soccer odds structure
    odds = market.get('odds', {})
    
    # For now, only handle 2-way markets (home/away)
    if 'home' in odds and 'away' in odds:
        # Convert to expected format
        converted_market = {
            'market_id': market['market_id'],
            'match_id': market['match_id'],
            'source': 'overtime_api',
            'sport': 'Soccer',
            'league': market.get('league', 'Unknown'),
            'home_team': market['home_team'],
            'away_team': market['away_team'],
            'maturity_date': datetime.fromisoformat(market['maturity_date']),
            'odds': {
                'home': float(odds['home']),
                'draw': float(odds.get('draw', 3.20)),  # Add draw odds for soccer
                'away': float(odds['away'])
            }
        }
        converted_markets.append(converted_market)

print(f"Converted {len(converted_markets)} markets")

# Show sample
print("\nSample converted markets:")
for i, market in enumerate(converted_markets[:5]):
    print(f"\n{i+1}. {market['home_team']} vs {market['away_team']}")
    print(f"   League: {market['league']}")
    print(f"   Odds: Home {market['odds']['home']:.2f} / Draw {market['odds']['draw']:.2f} / Away {market['odds']['away']:.2f}")

# Now test trading with these
import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from portfolio_trading_engine import PortfolioTradingEngine
from edge_calculator import EdgeCalculator

# Initialize components
session_manager = PaperTradingSessionManager()
edge_calculator = EdgeCalculator()
session_id = session_manager.get_current_session()

if session_id:
    print(f"\n✅ Using session: {session_id}")
    
    session = session_manager.get_session(session_id)
    current_bankroll = float(session['current_bankroll'])
    print(f"   Current bankroll: ${current_bankroll:,.2f}")
    
    # Trading configuration
    strategy_config = {
        'bankroll': current_bankroll,
        'kelly_fraction': 0.25,
        'min_edge': 0.005,  # Very low for testing (0.5%)
        'cap_per_bet': 0.02,  # 2% of bankroll max per bet
        'cap_per_game': 0.02,
        'min_bet': 10,
        'max_positions': 20
    }
    
    portfolio_engine = PortfolioTradingEngine(session_manager, edge_calculator, strategy_config)
    
    # Calculate edges
    print(f"\n📊 Calculating edges for {len(converted_markets)} markets...")
    signals = edge_calculator.calculate_edges(converted_markets)
    print(f"   Generated {len(signals)} signals")
    
    # Show sample signal
    if signals:
        sample = signals[0]
        print(f"\n   Sample signal:")
        print(f"   Home edge: {sample.get('home_edge', 0):.3f}")
        print(f"   Draw edge: {sample.get('draw_edge', 0):.3f}")
        print(f"   Away edge: {sample.get('away_edge', 0):.3f}")
    
    # Execute portfolio optimization
    print(f"\n🎯 Executing portfolio optimization...")
    result = portfolio_engine.execute_kelly_portfolio_optimization(
        session_id, 
        converted_markets, 
        signals
    )
    
    if result['success']:
        trades = result.get('trades', [])
        print(f"\n✅ SUCCESS! {len(trades)} trades executed")
        
        if trades:
            total_stake = sum(t['stake'] for t in trades)
            print(f"   Total staked: ${total_stake:.2f}")
            print(f"\n   First 5 trades:")
            for i, trade in enumerate(trades[:5]):
                print(f"   {i+1}. {trade['home_team']} vs {trade['away_team']}")
                print(f"      Bet: {trade['bet_on'].upper()} @ {trade['odds']:.2f}")
                print(f"      Stake: ${trade['stake']:.2f}")
                print(f"      Edge: {trade.get('edge', 0):.3f}")
        else:
            print("\n   No trades placed - edges may be below threshold")
    else:
        print(f"\n❌ FAILED: {result.get('error', 'Unknown error')}")
        print(f"   Message: {result.get('message', '')}")