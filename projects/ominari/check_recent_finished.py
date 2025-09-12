#!/usr/bin/env python3
"""Check for recently finished matches."""

from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market

def check_recent_finished():
    """Check for matches finished in the last 24 hours."""
    with db_manager.get_db_session() as db:
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(hours=24)
        
        # Find matches that should have finished
        finished_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == True,
            Market.resolved_outcome.isnot(None),
            Market.maturity_date > yesterday
        ).order_by(Market.maturity_date.desc()).limit(20).all()
        
        print(f"Recently finished matches (last 24h):")
        print(f"Found {len(finished_markets)} finished matches")
        
        for market in finished_markets:
            time_since = now - market.maturity_date
            hours_ago = time_since.total_seconds() / 3600
            print(f"\n{market.home_team} vs {market.away_team}")
            print(f"  Match time: {market.maturity_date.strftime('%Y-%m-%d %H:%M')} UTC ({hours_ago:.1f}h ago)")
            print(f"  Result: {market.resolved_outcome} ({market.home_score}-{market.away_score})")
            print(f"  Market ID: {market.source_id}")
            
        # Also check matches that should have finished but aren't marked
        should_be_finished = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date < now - timedelta(hours=3),  # 3 hours for game to finish
            Market.maturity_date > yesterday
        ).order_by(Market.maturity_date.desc()).limit(10).all()
        
        if should_be_finished:
            print(f"\n\nMatches that should be finished but aren't marked:")
            for market in should_be_finished:
                time_since = now - market.maturity_date
                hours_ago = time_since.total_seconds() / 3600
                print(f"\n{market.home_team} vs {market.away_team}")
                print(f"  Match time: {market.maturity_date.strftime('%Y-%m-%d %H:%M')} UTC ({hours_ago:.1f}h ago)")
                print(f"  is_finished: {market.is_finished}")
                print(f"  resolved_outcome: {market.resolved_outcome}")

if __name__ == "__main__":
    check_recent_finished()