#!/usr/bin/env python3
"""Check for real odds in database"""

import os
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Odd, Market
from sqlalchemy import and_, not_, or_

with db_manager.get_db_session() as db:
    # Get markets with real odds (not the default values)
    real_odds = db.query(Odd).filter(
        not_(or_(
            Odd.decimal_odds == 2.5,
            Odd.decimal_odds == 2.8,
            Odd.decimal_odds == 3.0
        ))
    ).limit(50).all()
    
    print(f'Found {len(real_odds)} markets with non-default odds\n')
    
    # Show some examples
    for i, odd in enumerate(real_odds[:20]):
        market = db.query(Market).filter(Market.source_id == odd.source_id).first()
        if market:
            print(f'{i+1}. {market.home_team} vs {market.away_team}')
            print(f'   {odd.outcome}: {odd.decimal_odds}')
            print()
    
    # Check if blockchain markets have real odds
    print("\nChecking blockchain markets specifically...")
    from sqlalchemy import text
    
    result = db.execute(text("""
        SELECT DISTINCT o.decimal_odds, COUNT(*) as count
        FROM odd o
        JOIN blockchain_market_links bl ON o.source_id = bl.source_id
        GROUP BY o.decimal_odds
        ORDER BY count DESC
        LIMIT 10
    """))
    
    print("\nBlockchain market odds distribution:")
    for row in result:
        print(f"  Odds {row[0]}: {row[1]} markets")