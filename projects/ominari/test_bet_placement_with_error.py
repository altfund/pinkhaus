#!/usr/bin/env python3
"""Test bet placement with error details"""

import asyncio
import os
import sys
import traceback
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import BettingSession, Bet
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier

async def test_bet_placement():
    """Test placing a bet directly with error details"""
    
    # Create session
    with db_manager.get_db_session() as db:
        session = BettingSession(
            as_of=datetime.now(timezone.utc),
            session_type='paper',
            strategy_name="Test Direct Bet",
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
        session_id = session.id
    
    print(f"Created session: {session_id}")
    
    # Mock opportunity (Liverpool with 27% edge)
    opportunity = {
        'market': type('obj', (object,), {
            'source_id': 'test_liverpool_001',
            'home_team': 'Liverpool UY',
            'away_team': 'Penarol'
        })(),
        'outcome': 'home',
        'odds': 3.3,
        'fair_prob': 0.303,
        'edge': 27.29
    }
    
    # Calculate bet size
    bankroll_config = BankrollConfig()
    bankroll = bankroll_config.get_current_bankroll()
    
    # Kelly calculation
    b = opportunity['odds'] - 1
    p = opportunity['fair_prob']
    q = 1 - p
    f = (p * b - q) / b
    kelly_fraction = 0.25
    f = f * kelly_fraction
    
    bet_amount = bankroll * f
    bet_amount = min(bet_amount, bankroll * 0.05)  # Max 5%
    bet_amount = max(bet_amount, 10.0)  # Min $10
    bet_amount = round(bet_amount, 2)
    
    print(f"Calculated bet size: ${bet_amount:.2f}")
    
    # Try to place bet
    try:
        with db_manager.get_db_session() as db:
            bet = Bet(
                session_id=session_id,
                source_id=opportunity['market'].source_id,
                unified_market_type='h2h',
                normalized_outcome=opportunity['outcome'],
                normalized_line=0.0,
                bet_name=f"{opportunity['market'].home_team} vs {opportunity['market'].away_team} - {opportunity['outcome']}",
                probability=opportunity['fair_prob'],
                odds=opportunity['odds'],
                stake=bet_amount,
                execution_stake=bet_amount,
                fee_amount=0.0,
                fee_pct=0.0
            )
            
            db.add(bet)
            db.commit()
            db.refresh(bet)
            
            print(f"✅ Bet placed successfully! ID: {bet.id}")
            
            # Send Discord notification
            discord_notifier.send_trade_alert({
                'type': 'BUY',
                'market': f"{opportunity['market'].home_team} vs {opportunity['market'].away_team}",
                'outcome': opportunity['outcome'],
                'amount': bet_amount,
                'odds': opportunity['odds'],
                'edge': opportunity['edge'],
                'kelly_pct': (bet_amount / bankroll) * 100,
                'bankroll': bankroll
            })
            
            print("📢 Discord notification sent!")
            
    except Exception as e:
        print(f"❌ Error placing bet: {type(e).__name__}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_bet_placement())