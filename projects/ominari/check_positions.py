#!/usr/bin/env python3
"""Check detailed position information"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from database_v2 import db_manager
from models import Bet

session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

# Get positions with more detail
with db_manager.get_db_session() as db:
    bets = db.query(Bet).filter(
        Bet.session_id == session_id
    ).order_by(Bet.created_at.desc()).limit(20).all()
    
    print('🎲 DETAILED POSITION INFORMATION')
    print('=' * 80)
    
    for i, bet in enumerate(bets[:10]):
        print(f'\nPosition {i+1}:')
        print(f'  Match ID: {bet.match_id}')
        print(f'  Sport: {bet.sport}')
        print(f'  Teams: {bet.home_team} vs {bet.away_team}')
        print(f'  Bet: {bet.bet_on} @ {bet.odds}')
        print(f'  Stake: ${bet.stake:.2f}')
        print(f'  Status: {bet.status}')
        print(f'  Signal: {bet.signal_name} (value: {bet.signal_value:.3f})')
        print(f'  Edge: {bet.edge:.1f}%')
        print(f'  Placed: {bet.created_at.strftime("%Y-%m-%d %H:%M:%S")}')
        if bet.kickoff_time:
            print(f'  Kickoff: {bet.kickoff_time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    # Summary stats
    open_bets = [b for b in bets if b.status in ['pending', 'open']]
    total_stake = sum(b.stake for b in open_bets)
    avg_odds = sum(b.odds for b in open_bets) / len(open_bets) if open_bets else 0
    avg_edge = sum(b.edge for b in open_bets) / len(open_bets) if open_bets else 0
    
    print(f'\n📊 SUMMARY STATISTICS')
    print('=' * 40)
    print(f'Total Open Positions: {len(open_bets)}')
    print(f'Total Exposure: ${total_stake:.2f}')
    print(f'Average Odds: {avg_odds:.2f}')
    print(f'Average Edge: {avg_edge:.1f}%')