#!/usr/bin/env python3
"""Manually trigger a trading check to see if it finds opportunities"""

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

from integrated_trading_system import IntegratedTradingSystem
from notifications.discord_notifier import discord_notifier

async def check_for_trades():
    """Check for trading opportunities"""
    system = IntegratedTradingSystem()
    
    print("🔍 Looking for trading opportunities...")
    print(f"Discord enabled: {discord_notifier.enabled}")
    
    # Find opportunities
    opportunities = await system.paper_system.find_tradeable_opportunities()
    
    if opportunities:
        print(f"\n✅ Found {len(opportunities)} opportunities!")
        
        for i, opp in enumerate(opportunities[:3]):  # Show top 3
            market = opp['market']
            print(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   Outcome: {opp['outcome']}")
            print(f"   Odds: {opp['odds']}")
            print(f"   Edge: {opp['edge']:.2f}%")
            print(f"   Suggested bet: ${opp['adjusted_bet_size']:.2f}")
            
            # Place the bet
            if i == 0:  # Place the first one
                print(f"\n💰 Placing paper trade...")
                result = await system.paper_system.place_liquidity_aware_bet(opp)
                if result:
                    print(f"✅ Trade placed! Check Discord for notification.")
                else:
                    print(f"❌ Trade failed")
    else:
        print("\n❌ No positive edge opportunities found")
        print("All markets still have negative edge")
        
        # Debug: Check our specific market
        from database_v2 import db_manager
        from models import Market, Odd
        
        with db_manager.get_db_session() as db:
            market = db.query(Market).filter(
                Market.home_team == "Portland Thorns FC",
                Market.away_team == "Houston Dash"
            ).first()
            
            if market:
                print(f"\nDebug - Portland Thorns market:")
                odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
                total_prob = sum(1/o.decimal_odds for o in odds)
                print(f"Total probability: {total_prob:.3f}")
                for odd in odds:
                    print(f"  {odd.outcome}: {odd.decimal_odds}")
                    
if __name__ == "__main__":
    asyncio.run(check_for_trades())