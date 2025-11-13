#!/usr/bin/env python3
"""
Check sport and league data in the database
"""
import os

# Set environment for PostgreSQL on port 5999
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market
from sqlalchemy import func, distinct

def check_sport_league_data():
    with db_manager.get_db_session() as db:
        # Get distinct sports
        sports = db.query(distinct(Market.sport)).filter(Market.sport != None).limit(20).all()
        print('Distinct Sports:')
        for sport in sports:
            if sport[0]:
                count = db.query(Market).filter(Market.sport == sport[0]).count()
                print(f'  - {sport[0]}: {count} markets')
        
        print('\nDistinct Leagues (sample):')
        # Get distinct leagues
        leagues = db.query(distinct(Market.league_name)).filter(Market.league_name != None).limit(20).all()
        for league in leagues:
            if league[0]:
                count = db.query(Market).filter(Market.league_name == league[0]).count()
                print(f'  - {league[0]}: {count} markets')
        
        print('\nSample Markets with Sport/League:')
        markets = db.query(Market).filter(Market.sport != None, Market.league_name != None).limit(10).all()
        for m in markets:
            print(f'  - {m.sport} / {m.league_name}: {m.home_team} vs {m.away_team}')
        
        print('\nMarkets with missing sport/league:')
        missing_sport = db.query(Market).filter(Market.sport == None).count()
        missing_league = db.query(Market).filter(Market.league_name == None).count()
        print(f'  - Missing sport: {missing_sport}')
        print(f'  - Missing league: {missing_league}')

if __name__ == "__main__":
    check_sport_league_data()