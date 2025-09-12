#!/usr/bin/env python3
"""Check live scores for current matches."""

from database_v2 import db_manager
from models import Market

with db_manager.get_db_session() as db:
    # Check our matches
    teams = ['Argentina', 'Paraguay', 'Colombia', 'Uruguay']
    markets = db.query(Market).filter(
        Market.home_team.in_(teams),
        Market.sport == 'Soccer'
    ).order_by(Market.maturity_date.desc()).limit(10).all()
    
    print("Recent matches:")
    for m in markets:
        if m.home_score is not None or m.is_finished:
            status = "(Final)" if m.is_finished else "(Live)"
            print(f'{m.home_team} vs {m.away_team}: {m.home_score}-{m.away_score} {status}')
        else:
            print(f'{m.home_team} vs {m.away_team}: Not started or no score data')