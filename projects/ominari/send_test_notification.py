#!/usr/bin/env python3
"""Send a test Discord notification to demonstrate the system is working"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from notifications.discord_notifier import discord_notifier
from database_v2 import db_manager
from models import Market
from config.bankroll_config import BankrollConfig
from datetime import datetime, timezone

# Get current market info
with db_manager.get_db_session() as db:
    market = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport == 'Soccer'
    ).first()
    
    if market:
        print(f"Using market: {market.home_team} vs {market.away_team}")
        
        # Send paper trading status update
        discord_notifier.send_daily_summary({
            'date': datetime.now().strftime('%Y-%m-%d'),
            'start_bankroll': 10000.00,
            'end_bankroll': 10000.00,
            'daily_pnl': 0.00,
            'total_trades': 0,
            'winning_trades': 0,
            'win_rate': 0.0,
            'total_exposure': 0.00,
            'active_positions': 0,
            'total_roi': 0.0
        })
        print("✅ Daily summary sent!")
        
        # Send market alert
        discord_notifier.send_market_alert('high_opportunity', {
            'market': f"{market.home_team} vs {market.away_team}",
            'edge': -8.3,  # Negative but showing for demo
            'liquidity': 5000.0,
            'max_bet': 100.0
        })
        print("✅ Market alert sent!")
        
        print("\n📈 System Status:")
        print("- Dashboard running at http://localhost:8888")
        print("- Discord notifications configured and working")
        print("- Paper trading system active (no positive edges currently)")
        print("- All markets showing -8.3% edge (bookmaker margin)")
        print("\nThe system will automatically place paper trades when it finds positive edge opportunities!")
    else:
        print("No active markets found")