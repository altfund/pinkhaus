#!/usr/bin/env python3
"""Fix dashboard position display by updating the data format"""

import os
os.environ['PG_PORT'] = '5999'

# We need to fix the web_monitor.py to properly format position data
# The JavaScript expects: market_name, outcome, total_stake, avg_odds, current_value, pnl
# But we're sending: home_team, away_team, stake, odds, etc.

from database_v2 import db_manager
from models import Market

def get_formatted_positions(positions):
    """Format positions to match what the dashboard expects"""
    formatted_open = {}
    formatted_closed = []
    
    for i, pos in enumerate(positions):
        if pos['status'] in ['pending', 'open']:
            # Get market info if team names are empty
            market_name = f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}"
            
            if market_name == " vs ":  # Empty names, look up from database
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        market_name = f"{market.home_team} vs {market.away_team}"
            
            formatted_open[str(i)] = {
                'market_name': market_name,
                'outcome': pos.get('bet_on', ''),
                'total_stake': float(pos.get('stake', 0)),
                'avg_odds': float(pos.get('odds', 0)),
                'current_value': float(pos.get('stake', 0)),  # For pending, same as stake
                'pnl': 0,  # No P&L until settled
                'status': 'open'
            }
        
        elif pos['status'] in ['settled', 'won', 'lost']:
            market_name = f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}"
            
            if market_name == " vs ":
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        market_name = f"{market.home_team} vs {market.away_team}"
            
            payout = float(pos.get('payout', 0))
            stake = float(pos.get('stake', 0))
            
            formatted_closed.append({
                'market_name': market_name,
                'outcome': pos.get('bet_on', ''),
                'stake': stake,
                'odds': float(pos.get('odds', 0)),
                'payout': payout,
                'pnl': payout - stake,
                'result': 'won' if payout > stake else 'lost',
                'settled_at': pos.get('resolved_at', pos.get('placed_at', ''))
            })
    
    return formatted_open, formatted_closed

# Test the fix
if __name__ == "__main__":
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    
    manager = PaperTradingSessionManager()
    session_id = manager.get_current_session()
    
    if session_id:
        positions = manager.get_positions(session_id)
        formatted_open, formatted_closed = get_formatted_positions(positions)
        
        print(f"Formatted {len(formatted_open)} open positions")
        if formatted_open:
            print("\nFirst open position:")
            for key, value in list(formatted_open.values())[0].items():
                print(f"  {key}: {value}")