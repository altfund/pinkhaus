#!/usr/bin/env python3
"""Test paper trade with real market data"""

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

from database_v2 import db_manager
from models import Market, Odd, BettingSession, Bet
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier

async def place_real_paper_trade():
    """Place a paper trade on a real market"""
    
    # Get the Liverpool market
    with db_manager.get_db_session() as db:
        market = db.query(Market).filter(
            Market.home_team == "Liverpool UY"
        ).first()
        
        if not market:
            print("❌ Liverpool market not found")
            return
            
        print(f"Found market: {market.home_team} vs {market.away_team}")
        print(f"Source ID: {market.source_id}")
        
        # Get odds for home outcome
        odd = db.query(Odd).filter(
            Odd.source_id == market.source_id,
            Odd.outcome == 'home'
        ).first()
        
        if not odd:
            print("❌ No odds found for home outcome")
            return
            
        print(f"Odds for home: {odd.decimal_odds}")
        
        # Create session
        session = BettingSession(
            as_of=datetime.now(timezone.utc),
            session_type='paper',
            strategy_name="Manual Paper Trade Test",
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
        
        print(f"\nCreated session: {session.id}")
        
        # Calculate bet size
        bankroll_config = BankrollConfig()
        bankroll = bankroll_config.get_current_bankroll()
        
        # For high edge (27%), Kelly would suggest a large bet
        # But we'll cap it at 5% of bankroll = $500
        bet_amount = min(500.0, bankroll * 0.05)
        
        print(f"Placing bet of ${bet_amount:.2f} on {market.home_team} to win")
        
        # Place the bet
        bet = Bet(
            session_id=session.id,
            source_id=market.source_id,
            unified_market_type='h2h',
            normalized_outcome='home',
            normalized_line=0.0,
            bet_name=f"{market.home_team} vs {market.away_team} - home",
            probability=0.303,  # Fair probability for 27% edge
            odds=odd.decimal_odds,
            stake=bet_amount,
            execution_stake=bet_amount,
            fee_amount=0.0,
            fee_pct=0.0
        )
        
        db.add(bet)
        db.commit()
        db.refresh(bet)
        
        print(f"\n✅ Bet placed successfully!")
        print(f"   Bet ID: {bet.id}")
        print(f"   Market: {market.home_team} vs {market.away_team}")
        print(f"   Outcome: home")
        print(f"   Odds: {odd.decimal_odds}")
        print(f"   Stake: ${bet.stake:.2f}")
        print(f"   Potential win: ${(bet.stake * bet.odds - bet.stake):.2f}")
        
        # Send Discord notification
        discord_notifier.send_trade_alert({
            'type': 'BUY',
            'market': f"{market.home_team} vs {market.away_team}",
            'outcome': 'home',
            'amount': bet.stake,
            'odds': bet.odds,
            'edge': 27.29,
            'kelly_pct': (bet.stake / bankroll) * 100,
            'bankroll': bankroll
        })
        
        print("\n📢 Discord notification sent!")
        print("🎉 Paper trading is working! Check Discord for the notification.")

if __name__ == "__main__":
    asyncio.run(place_real_paper_trade())