#!/usr/bin/env python3
"""Manually place a test paper trade to verify system is working"""

import asyncio
import os
import sys
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier

async def place_test_trade():
    """Place a test trade for demonstration"""
    bankroll_config = BankrollConfig()
    
    # Get a current session
    with db_manager.get_db_session() as db:
        session = db.query(BettingSession).order_by(
            BettingSession.created_at.desc()
        ).first()
        
        if not session:
            # Create a new session
            session = BettingSession(
                created_at=datetime.now(timezone.utc),
                initial_bankroll=10000.0
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            print(f"Created new betting session: {session.id}")
        
        # Find a market to bet on
        market = db.query(Market).filter(
            Market.maturity_date > datetime.now(timezone.utc),
            Market.sport == 'Soccer'
        ).first()
        
        if not market:
            print("No active markets found")
            return
            
        # Get odds for this market
        odds = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).all()
        
        if len(odds) < 2:
            print("Not enough odds for this market")
            return
            
        # Pick the first outcome
        selected_odd = odds[0]
        
        print(f"\nPlacing test trade:")
        print(f"Market: {market.home_team} vs {market.away_team}")
        print(f"Outcome: {selected_odd.outcome}")
        print(f"Odds: {selected_odd.decimal_odds}")
        print(f"Current bankroll: ${bankroll_config.get_current_bankroll():.2f}")
        
        # Create the bet
        stake = 100.0  # Fixed $100 test bet
        bet = Bet(
            session_id=session.id,
            source_id=market.source_id[:66],  # Truncate to fit the field
            bet_name=f"{market.home_team[:20]} vs {market.away_team[:20]}",
            normalized_outcome=selected_odd.outcome,
            odds=selected_odd.decimal_odds,
            stake=stake,
            created_at=datetime.now(timezone.utc)
        )
        
        db.add(bet)
        db.commit()
        db.refresh(bet)
        
        print(f"✅ Test trade placed! Bet ID: {bet.id}")
        
        # Update bankroll
        bankroll_config.update_bankroll(bankroll_config.get_current_bankroll() - stake)
        
        # Send Discord notification
        if discord_notifier.enabled:
            discord_notifier.send_trade_alert({
                'type': 'NEW',
                'market': f"{market.home_team} vs {market.away_team}",
                'outcome': selected_odd.outcome,
                'amount': stake,
                'odds': selected_odd.decimal_odds,
                'edge': -8.3,
                'bankroll': bankroll_config.get_current_bankroll(),
                'note': 'TEST TRADE - Negative edge trade for demonstration'
            })
            print("📨 Discord notification sent!")
        else:
            print("❌ Discord notifications not enabled")
            
        # Show updated status
        print(f"\nUpdated bankroll: ${bankroll_config.get_current_bankroll():.2f}")
        print("Check your Discord channel for the notification!")
        print("Visit http://localhost:8888 to see the trade in the dashboard")

if __name__ == "__main__":
    asyncio.run(place_test_trade())