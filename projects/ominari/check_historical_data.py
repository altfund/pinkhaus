#!/usr/bin/env python3
"""
Check the status of historical data in the system.
"""

from database_v2 import db_manager
from models import Market
from paper_trading_sessions import PaperTradingSessionManager
from sqlalchemy import func, and_
from datetime import datetime, timedelta

def check_database_results():
    """Check match results in the database."""
    print("\n📊 DATABASE MATCH RESULTS STATUS")
    print("-" * 50)
    
    with db_manager.get_db_session() as db:
        # Total finished matches
        finished_count = db.query(func.count(Market.source_id)).filter(
            Market.is_finished == True
        ).scalar()
        
        # Finished with scores
        with_scores = db.query(func.count(Market.source_id)).filter(
            and_(
                Market.is_finished == True,
                Market.home_score.isnot(None),
                Market.away_score.isnot(None)
            )
        ).scalar()
        
        # Finished without scores
        without_scores = finished_count - with_scores
        
        # Recent finished (last 7 days)
        recent_cutoff = datetime.utcnow() - timedelta(days=7)
        recent_finished = db.query(func.count(Market.source_id)).filter(
            and_(
                Market.is_finished == True,
                Market.last_update >= recent_cutoff
            )
        ).scalar()
        
        print(f"Total finished matches: {finished_count:,}")
        print(f"With complete scores: {with_scores:,} ({with_scores/finished_count*100:.1f}%)")
        print(f"Missing scores: {without_scores:,}")
        print(f"Finished in last 7 days: {recent_finished:,}")

def check_paper_trading_data():
    """Check paper trading positions data."""
    print("\n💰 PAPER TRADING POSITIONS STATUS")
    print("-" * 50)
    
    session_manager = PaperTradingSessionManager()
    sessions = session_manager.sessions.get('sessions', {})
    
    total_positions = 0
    positions_with_results = 0
    positions_with_odds_history = 0
    positions_with_edges = 0
    
    for session_id, session in sessions.items():
        # Count open positions
        for pos in session.get('positions', {}).values():
            total_positions += 1
            if pos.get('historical_odds'):
                positions_with_odds_history += 1
            if pos.get('calculated_edge') or pos.get('edge_stats'):
                positions_with_edges += 1
        
        # Count closed positions
        for pos in session.get('closed_positions', []):
            total_positions += 1
            if pos.get('result') in ['won', 'lost']:
                positions_with_results += 1
            if pos.get('historical_odds') or pos.get('odds_movement'):
                positions_with_odds_history += 1
            if pos.get('calculated_edge') or pos.get('edge_stats'):
                positions_with_edges += 1
    
    print(f"Total sessions: {len(sessions)}")
    print(f"Total positions: {total_positions}")
    print(f"Positions with results: {positions_with_results}")
    print(f"Positions with odds history: {positions_with_odds_history}")
    print(f"Positions with edge data: {positions_with_edges}")
    
    # Check enrichment status
    if total_positions > 0:
        enrichment_pct = (positions_with_odds_history / total_positions) * 100
        print(f"\nEnrichment status: {enrichment_pct:.1f}% complete")
        
        if enrichment_pct < 100:
            print("\n⚠️  Some positions are not fully enriched.")
            print("Run: python run_historical_backfill.py")

def check_catchup_status():
    """Check catch-up service status."""
    print("\n🔄 CATCH-UP SERVICE STATUS")
    print("-" * 50)
    
    try:
        import json
        with open('catchup_state.json', 'r') as f:
            state = json.load(f)
        
        last_run = state.get('last_run')
        if last_run:
            last_run_dt = datetime.fromisoformat(last_run)
            time_since = datetime.now() - last_run_dt.replace(tzinfo=None)
            
            print(f"Last run: {last_run_dt} ({time_since.total_seconds()/3600:.1f} hours ago)")
            print(f"Matches processed: {state.get('matches_processed', 0)}")
            print(f"Positions updated: {state.get('positions_updated', 0)}")
            print(f"Errors: {state.get('errors', 0)}")
        else:
            print("⚠️  Catch-up service has not been run yet")
            
    except FileNotFoundError:
        print("❌ No catch-up state found. Service has not been run.")

def main():
    print("="*60)
    print("HISTORICAL DATA STATUS CHECK")
    print("="*60)
    
    check_database_results()
    check_paper_trading_data()
    check_catchup_status()
    
    print("\n" + "="*60)
    print("To ensure all data is complete:")
    print("1. Run backfill: python run_historical_backfill.py")
    print("2. Set up catch-up: python results_catchup_service.py --mode setup")
    print("3. Check dashboard: http://localhost:8888/unified")

if __name__ == "__main__":
    main()