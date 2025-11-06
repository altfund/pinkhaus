#!/usr/bin/env python3
"""Check if we're using real market data"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta

# Check recent markets in database
with db_manager.get_db_session() as db:
    markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.maturity_date < datetime.now(timezone.utc) + timedelta(days=7),
        Market.is_finished == False
    ).order_by(Market.maturity_date).limit(10).all()
    
    print('🏟️ UPCOMING MARKETS IN DATABASE:')
    print('=' * 80)
    for m in markets:
        print(f'\nMarket ID: {m.source_id}')
        print(f'  Teams: {m.home_team} vs {m.away_team}')
        print(f'  Sport: {m.sport}')
        print(f'  Source: {m.source}')
        print(f'  Maturity: {m.maturity_date}')
        print(f'  League: {m.league_name if hasattr(m, "league_name") else "N/A"}')
        
        # Check odds for this market
        odds = db.query(Odd).filter(
            Odd.source_id == m.source_id
        ).order_by(Odd.updated_at.desc()).limit(3).all()
        
        if odds:
            print(f'  Latest Odds:')
            for o in odds:
                print(f'    - {o.outcome}: {o.decimal_odds}')
    
    # Check if these are test markets
    print('\n🔍 DATA QUALITY CHECK:')
    test_market_count = db.query(Market).filter(
        Market.source_id.in_(['api_0000000000000000', 'api_777361747a327a79'])
    ).count()
    
    if test_market_count > 0:
        print(f'⚠️  Found {test_market_count} test/dummy markets')
        print('These appear to be test data, not real markets!')
    
    # Check for real sports data
    real_sports_count = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport.in_(['Soccer', 'Basketball', 'Football', 'Hockey', 'Baseball']),
        ~Market.source_id.like('api_0000%'),
        ~Market.source_id.like('api_7773%')
    ).count()
    
    print(f'\n✅ Real sports markets available: {real_sports_count}')