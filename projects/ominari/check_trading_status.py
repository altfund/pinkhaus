#\!/usr/bin/env python3
"""Check trading system status"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from datetime import datetime, timezone, timedelta

with db_manager.get_db_session() as db:
    # Check recent bets
    recent_bets = db.query(Bet).order_by(Bet.created_at.desc()).limit(10).all()
    print(f"Recent bets: {len(recent_bets)}")
    
    # Check betting sessions
    recent_sessions = db.query(BettingSession).order_by(BettingSession.created_at.desc()).limit(5).all()
    print(f"Recent sessions: {len(recent_sessions)}")
    
    # Check markets with positive edge
    recent_time = datetime.now(timezone.utc) - timedelta(hours=24)
    active_markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.created_at > recent_time
    ).limit(100).all()
    
    print(f"Active markets in last 24h: {len(active_markets)}")
    
    # Check for positive edge markets
    positive_edge_count = 0
    for market in active_markets[:20]:  # Check first 20
        odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
        if len(odds) >= 2:
            # Calculate edge
            total_prob = sum(1/odd.decimal_odds for odd in odds)
            for odd in odds:
                fair_prob = (1/odd.decimal_odds) / total_prob
                fair_odds = 1 / fair_prob
                edge = ((odd.decimal_odds / fair_odds) - 1) * 100
                if edge > 2:
                    positive_edge_count += 1
                    print(f"Positive edge found: {market.home_team} vs {market.away_team}, {odd.outcome} @ {odd.decimal_odds} (edge: {edge:.2f}%)")
                    break
    
    print(f"\nTotal markets with positive edge (>2%): {positive_edge_count}")
EOF < /dev/null
