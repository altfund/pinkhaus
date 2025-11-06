#!/usr/bin/env python3
"""Test the trading flow directly"""

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
# Signal providers loaded automatically by EdgeCalculator
import requests
from datetime import datetime, timezone

# Initialize components
session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

if not session_id:
    print("No active session")
    exit(1)

print(f"Using session: {session_id}")

# Get session
session = session_manager.get_session(session_id)
print(f"Session bankroll: ${session['current_bankroll']:,.2f}")

# Initialize edge calculator and portfolio engine
edge_calculator = EdgeCalculator()

strategy_config = {
    'bankroll': float(session['current_bankroll']),
    'kelly_fraction': 0.25,
    'min_edge': 0.02,
    'cap_per_bet': 0.01,
    'cap_per_game': 0.02,
    'min_bet': 10,
    'max_positions': 20
}

portfolio_engine = PortfolioTradingEngine(session_manager, edge_calculator, strategy_config)

# Fetch markets
print("\nFetching markets...")
response = requests.get("https://api.overtime.io/overtime-v2/games-info", headers={'accept': 'application/json'})

if response.status_code != 200:
    print(f"Failed to fetch markets: {response.status_code}")
    exit(1)

data = response.json()
games = data.get('games', [])
print(f"Found {len(games)} games from API")

# Filter for upcoming soccer games
now = datetime.now(timezone.utc).timestamp() * 1000  # Convert to milliseconds
soccer_games = []

for game in games:
    if game.get('sport') == 'Soccer' and game.get('maturity', 0) > now:
        soccer_games.append(game)
        
print(f"Found {len(soccer_games)} upcoming soccer games")

if not soccer_games:
    print("No upcoming soccer games found")
    exit(0)

# Take first 5 for testing
test_games = soccer_games[:5]

print("\nTest games:")
for i, game in enumerate(test_games):
    print(f"{i+1}. {game['homeTeam']} vs {game['awayTeam']} - {datetime.fromtimestamp(game['maturity']/1000, tz=timezone.utc)}")

# Convert to market format
markets = []
for game in test_games:
    market = {
        'market_id': game['gameId'],
        'match_id': game['gameId'],
        'source': 'overtime_api',
        'sport': 'Soccer',
        'league': game.get('league', 'Unknown'),
        'home_team': game['homeTeam'],
        'away_team': game['awayTeam'],
        'maturity_date': datetime.fromtimestamp(game['maturity']/1000, tz=timezone.utc),
        'odds': {
            'home': game['odds']['homeOdds'],
            'draw': game['odds']['drawOdds'],
            'away': game['odds']['awayOdds']
        }
    }
    markets.append(market)

# Calculate edges
print("\nCalculating edges...")
signals = edge_calculator.calculate_edges_batch(markets)
print(f"Generated {len(signals)} signals")

# Show a sample signal
if signals:
    sample = signals[0]
    print(f"\nSample signal:")
    print(f"  Market: {markets[0]['home_team']} vs {markets[0]['away_team']}")
    print(f"  Home edge: {sample.get('home_edge', 0):.3f}")
    print(f"  Draw edge: {sample.get('draw_edge', 0):.3f}")
    print(f"  Away edge: {sample.get('away_edge', 0):.3f}")

# Try to place trades
print("\nExecuting portfolio optimization...")
result = portfolio_engine.execute_kelly_portfolio_optimization(session_id, markets, signals)

if result['success']:
    print(f"\n✅ Portfolio optimization complete!")
    print(f"   Trades executed: {len(result.get('trades', []))}")
    print(f"   Message: {result.get('message', '')}")
    
    if result.get('trades'):
        print("\nTrades placed:")
        for trade in result['trades'][:3]:  # Show first 3
            print(f"   {trade['home_team']} vs {trade['away_team']} - Bet {trade['bet_on'].upper()} @ {trade['odds']:.2f} - ${trade['stake']:.2f}")
else:
    print(f"\n❌ Portfolio optimization failed: {result.get('error', 'Unknown error')}")

# Check positions again
positions = session_manager.get_positions(session_id)
open_positions = [p for p in positions if p['status'] == 'pending']
print(f"\nTotal positions now: {len(open_positions)}")

if not open_positions and result['success']:
    print("\n⚠️  No positions were created despite successful optimization")
    print("Possible reasons:")
    print("  - No markets met the minimum edge requirement (2%)")
    print("  - Stakes were below minimum bet ($10)")
    print("  - Position limit already reached")