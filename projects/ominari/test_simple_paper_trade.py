#!/usr/bin/env python3
"""Simple test to place a paper trade and send Discord notification"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from paper_trading_live import LivePaperTrader
from database_v2 import db_manager
from models import BettingSession
from notifications.discord_notifier import discord_notifier

async def place_test_trade():
    """Place a test paper trade"""
    trader = LivePaperTrader()
    trader.min_edge = -1.0  # Allow low/negative edges
    
    print("🎯 Creating paper trading session...")
    
    # Create session
    with db_manager.get_db_session() as db:
        session = BettingSession(
            as_of=datetime.now(timezone.utc),
            session_type='paper',
            strategy_name="Test Paper Trade",
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
    
    print("✅ Session created")
    print("🔍 Finding betting opportunities...")
    
    # Find opportunities
    opportunities = await trader.find_betting_opportunities()
    
    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities!")
        
        # Show top 3
        for i, opp in enumerate(opportunities[:3]):
            market = opp['market']
            print(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   Sport: {market.sport}")
            print(f"   Outcome: {opp['outcome']}")
            print(f"   Odds: {opp['odds']}")
            print(f"   Edge: {opp['edge']:.2f}%")
            print(f"   Fair prob: {opp['fair_prob']:.3f}")
            
            # Place the first bet
            if i == 0:
                print(f"\n💰 Placing paper trade on this market...")
                bet = await trader.place_bet(opp)
                
                if bet:
                    print(f"✅ Bet placed successfully!")
                    print(f"   Bet ID: {bet.id}")
                    print(f"   Stake: ${bet.stake:.2f}")
                    print(f"   Potential win: ${(bet.stake * bet.odds - bet.stake):.2f}")
                    
                    # Send Discord notification manually since we're not using the full system
                    discord_notifier.send_trade_alert({
                        'type': 'BUY',
                        'market': f"{market.home_team} vs {market.away_team}",
                        'outcome': opp['outcome'],
                        'amount': bet.stake,
                        'odds': bet.odds,
                        'edge': opp['edge'],
                        'kelly_pct': (bet.stake / 10000) * 100,
                        'bankroll': 10000.0
                    })
                    
                    print("📢 Discord notification sent!")
                    print("\n🎉 Paper trading is working! Check Discord for the notification.")
                else:
                    print("❌ Failed to place bet")
    else:
        print("\n❌ No opportunities found")
        print("All markets have negative edge")

if __name__ == "__main__":
    asyncio.run(place_test_trade())