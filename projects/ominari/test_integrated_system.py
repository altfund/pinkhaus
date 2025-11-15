#!/usr/bin/env python3
"""Test the integrated trading system directly"""

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

async def test_system():
    """Test the integrated trading system"""
    system = IntegratedTradingSystem()
    
    # Check status
    status = system.get_status()
    print(f"System status: {status}")
    
    # Check Discord
    from notifications.discord_notifier import discord_notifier
    print(f"Discord enabled: {discord_notifier.enabled}")
    
    # Try to find one opportunity
    print("\nLooking for trading opportunities...")
    opportunities = await system.paper_system.find_tradeable_opportunities()
    
    if opportunities:
        print(f"Found {len(opportunities)} opportunities:")
        for opp in opportunities[:3]:
            market = opp['market']
            print(f"  - {market.home_team} vs {market.away_team}")
            print(f"    {opp['outcome']} @ {opp['odds']:.2f} (Edge: {opp['edge']:.2f}%)")
            print(f"    Bet size: ${opp['adjusted_bet_size']:.2f}")
    else:
        print("No opportunities found")
        
        # Debug: Check if there are any markets with odds
        from database_v2 import db_manager
        from models import Market, Odd
        from datetime import datetime, timezone
        
        with db_manager.get_db_session() as db:
            # Get an active market
            market = db.query(Market).filter(
                Market.maturity_date > datetime.now(timezone.utc)
            ).first()
            
            if market:
                print(f"\nSample market: {market.home_team} vs {market.away_team}")
                odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
                print(f"Odds count: {len(odds)}")
                for odd in odds:
                    print(f"  {odd.outcome}: {odd.decimal_odds}")
                    
                # Calculate edge manually
                if len(odds) >= 2:
                    total_prob = sum(1/odd.decimal_odds for odd in odds)
                    print(f"Total probability: {total_prob}")
                    for odd in odds:
                        fair_prob = (1/odd.decimal_odds) / total_prob
                        fair_odds = 1 / fair_prob
                        edge = ((odd.decimal_odds / fair_odds) - 1) * 100
                        print(f"  {odd.outcome}: Edge = {edge:.2f}%")

if __name__ == "__main__":
    asyncio.run(test_system())