#!/usr/bin/env python3
"""
Fix dashboard to show real data from paper trading
"""

import os
os.environ['PG_PORT'] = '5999'
os.environ['PG_DB'] = 'ominari_production'
os.environ['DATABASE_URL'] = 'postgresql://ominari_user:ominari_2025_secure@localhost:5999/database_v2'

from database_v2 import db_manager
from models import BettingSession, Bet, Market, Odd
from datetime import datetime, timedelta
from sqlalchemy import func

# First, let's check current data and models
with db_manager.get_db_session() as db:
    # Check if BettingSession has is_paper column
    try:
        # Check if the column exists
        test_query = db.query(BettingSession).filter(BettingSession.is_paper == True).first()
        has_is_paper = True
        print("BettingSession has is_paper column")
    except:
        has_is_paper = False
        print("BettingSession does NOT have is_paper column")
    
    # Check if Bet table has new columns
    try:
        from sqlalchemy import inspect
        inspector = inspect(db.bind)
        bet_columns = [col['name'] for col in inspector.get_columns('bet')]
        print(f"\nBet table columns: {bet_columns}")
        
        # Check if we have the new Bet model fields
        has_new_bet_model = 'market_id' in bet_columns or 'betting_session_id' in bet_columns
        print(f"Has new Bet model: {has_new_bet_model}")
    except Exception as e:
        print(f"Error checking Bet columns: {e}")
        has_new_bet_model = False

# Only create test data if we have proper models
if has_is_paper:
    # Create a paper trading session
    session = BettingSession(
        bankroll=10000.0,
        created_at=datetime.now() - timedelta(hours=1),
        strategy_name="test_paper_trading",
        is_paper=True
    )
    # Set other required fields based on what columns exist
    if hasattr(BettingSession, 'min_bet_size'):
        session.min_bet_size = 10.0
    if hasattr(BettingSession, 'max_bet_size'):
        session.max_bet_size = 500.0  
    if hasattr(BettingSession, 'max_exposure'):
        session.max_exposure = 2500.0
    
    db.add(session)
    db.commit()
    print(f"\nCreated paper trading session {session.id}")
    
    # Only create bets if we have the right model
    if has_new_bet_model:
        print("\nNOTE: New Bet model detected but not implemented yet")
    else:
        print("\nUsing old Bet model structure")
else:
    print("\nWARNING: BettingSession model doesn't have is_paper field")
    print("The models may need to be updated for paper trading support")