#!/usr/bin/env python3
"""Test paper trading directly with lower edge threshold"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from paper_trading_live import LivePaperTrader
from database_v2 import db_manager
from models import Market, Odd, BettingSession
from datetime import datetime, timezone
from notifications.discord_notifier import discord_notifier

async def test_paper_trading():
    """Test paper trading with temporary lower edge threshold"""
    trader = LivePaperTrader()
    
    # Temporarily lower the min edge to catch our arbitrage opportunity
    trader.min_edge = -1.0  # Allow any positive or slightly negative edge
    
    print(f"🔍 Testing paper trading with min_edge={trader.min_edge}%")
    print(f"Discord enabled: {discord_notifier.enabled}")
    
    # Create a simple session
    with db_manager.get_db_session() as db:
        session = BettingSession(
            as_of=datetime.now(timezone.utc),
            session_type='paper',
            strategy_name="Test Arbitrage Detection",
            kelly_bankroll=10000.0,
            execution_bankroll=10000.0,
            kelly_fraction=0.25,
            cap_per_game=1000.0,
            cap_per_bet=500.0,
            cap_per_game_market=500.0,
            min_bet_abs=10.0,
            min_bet_pct=0.001,
            min_break_minutes=0.0,
            avg_game_duration_minutes=120.0
        )
        db.add(session)
        db.commit()
        trader.session_id = session.id
    
    # Find opportunities
    opportunities = await trader.find_betting_opportunities()
    
    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities!")
        
        for i, opp in enumerate(opportunities[:3]):
            market = opp['market']
            print(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   Outcome: {opp['outcome']}")
            print(f"   Odds: {opp['odds']}")
            print(f"   Edge: {opp['edge']:.2f}%")
            print(f"   Fair prob: {opp['fair_prob']:.3f}")
            
            # Place the first bet
            if i == 0:
                print(f"\n💰 Placing bet...")
                bet = await trader.place_bet(opp)
                if bet:
                    print(f"✅ Bet placed! ID: {bet.id}")
                    print(f"   Check Discord for notification!")
                else:
                    print(f"❌ Failed to place bet")
    else:
        print("\n❌ No opportunities found")
        
        # Debug specific market
        with db_manager.get_db_session() as db:
            market = db.query(Market).filter(
                Market.home_team == "Portland Thorns FC"
            ).first()
            
            if market:
                print(f"\nDebug - Portland Thorns market:")
                odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
                total_prob = sum(1/o.decimal_odds for o in odds)
                print(f"Total probability: {total_prob:.3f}")
                for odd in odds:
                    print(f"  {odd.outcome}: {odd.decimal_odds}")

if __name__ == "__main__":
    asyncio.run(test_paper_trading())