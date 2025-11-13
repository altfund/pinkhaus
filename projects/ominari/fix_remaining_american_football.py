import os
os.environ['POSTGRES_HOST'] = 'localhost'
os.environ['POSTGRES_PORT'] = '5999'
os.environ['POSTGRES_DB'] = 'ominari_dev'
os.environ['POSTGRES_USER'] = 'ess'
os.environ['POSTGRES_PASSWORD'] = ''

from models import Market, Odd
from database_v2 import db_manager
from sqlalchemy import or_

american_teams = [
    'Patriots', 'Cowboys', 'Eagles', 'Notre Dame', 'Alabama', 
    'Clemson', 'Georgia', 'Ohio State', 'Michigan', 'LSU',
    'Florida', 'Auburn', 'Penn State', 'Wisconsin', 'Iowa',
    'Texas', 'Oklahoma', 'USC', 'UCLA', 'Stanford',
    'Tennessee', 'Kentucky', 'Mississippi', 'Arkansas', 'Missouri'
]

total_fixed = 0

with db_manager.get_db_session() as db:
    for team in american_teams:
        markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            or_(Market.home_team.ilike(f'%{team}%'),
                Market.away_team.ilike(f'%{team}%'))
        ).all()
        
        if markets:
            print(f'\nFound {len(markets)} markets with {team} classified as Soccer:')
            for m in markets[:3]:  # Show first 3
                print(f'  - {m.home_team} vs {m.away_team} ({m.league_name})')
            
            # Fix them
            for m in markets:
                m.sport = 'American Football'
            
            db.commit()
            print(f'  Fixed {len(markets)} markets')
            total_fixed += len(markets)

print(f'\n✓ Total fixed: {total_fixed} markets')
print('\nClearing dashboard cache...')
import requests
try:
    response = requests.post('http://localhost:8888/clear-cache')
    print('✓ Cache cleared')
except:
    print('\! Could not clear cache (dashboard may not be running)')
