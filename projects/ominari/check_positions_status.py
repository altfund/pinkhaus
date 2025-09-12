#!/usr/bin/env python3
"""Check status of current paper trading positions."""

from datetime import datetime, timezone
from paper_trading_sessions import PaperTradingSessionManager
from database_v2 import db_manager
from models import Market

def check_positions():
    """Check current positions and their market status."""
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    
    if not current_session:
        print("No active session found")
        return
        
    print(f"Session: {current_session['session_id']}")
    print(f"Portfolio Value: ${current_session['portfolio_value']:.2f}")
    print(f"Cash Available: ${current_session['current_bankroll']:.2f}")
    print(f"Open Positions: {len(current_session.get('positions', {}))}")
    print()
    
    # Check each position
    with db_manager.get_db_session() as db:
        for position_key, pos in current_session.get('positions', {}).items():
            market_id = pos.get('market_id')
            
            # Get market info from database
            market = db.query(Market).filter(Market.source_id == market_id).first()
            
            if market:
                now_utc = datetime.now(timezone.utc)
                time_diff = market.maturity_date - now_utc
                hours_until = time_diff.total_seconds() / 3600
                
                status = "UPCOMING"
                if hours_until < 0:
                    if market.is_finished:
                        status = "FINISHED"
                    else:
                        status = "IN-PLAY"
                elif hours_until < 0.25:  # 15 minutes
                    status = "STARTING SOON"
                
                print(f"{market.home_team} vs {market.away_team} - {pos.get('outcome')}")
                print(f"  Stake: ${pos.get('total_stake', 0):.2f}")
                print(f"  Avg Odds: {pos.get('avg_odds', 0):.2f}")
                print(f"  Maturity: {market.maturity_date.strftime('%Y-%m-%d %H:%M')} UTC")
                print(f"  Time until: {hours_until:.1f}h")
                print(f"  Status: {status}")
                print(f"  Is Finished: {market.is_finished}")
                if market.is_finished and market.resolved_outcome:
                    print(f"  Result: {market.resolved_outcome} ({market.home_score}-{market.away_score})")
                print()

if __name__ == "__main__":
    check_positions()