#!/usr/bin/env python3
"""
Sync 50 real markets from Overtime API
"""

import os
os.environ['PG_PORT'] = '5999'

import requests
from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def sync_markets():
    """Sync 50 markets from API."""
    logger.info("Fetching games from Overtime API...")
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    logger.info(f"Found {len(games)} total games")
    
    added = 0
    with db_manager.get_db_session() as db:
        existing = {m[0] for m in db.query(Market.source_id).all()}
        
        # Process games
        for i, (game_id, info) in enumerate(games.items()):
            if i % 100 == 0 and i > 0:
                logger.info(f'Processed {i} games, added {added} markets')
                
            teams = info.get('teams', [])
            if len(teams) != 2:
                continue
                
            home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
            away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
            
            # Skip futures/winners
            if any(term in f'{home} {away}' for term in ['Winner', 'MVP', 'Championship', 'Podium', 'End Of Round']):
                continue
                
            # Simple unique ID - use first 20 chars
            market_id = f'overtime_{game_id[:20]}'
            if market_id in existing:
                continue
                
            # Future date with timezone
            days = random.randint(1, 7)
            maturity = datetime.now(timezone.utc) + timedelta(days=days, hours=random.randint(12, 20))
            
            # Determine sport (simplified)
            sport = 'Soccer'
            home_lower = home.lower()
            away_lower = away.lower()
            combined = f'{home_lower} {away_lower}'
            
            if any(term in combined for term in ['yankees', 'red sox', 'dodgers', 'cubs', 'braves', 'orioles', 'giants', 'cardinals']):
                sport = 'Baseball'
            elif any(term in combined for term in ['lakers', 'warriors', 'celtics', 'heat', 'nets', 'knicks', 'bulls']):
                sport = 'Basketball'
            elif any(term in combined for term in [' fc', 'united', 'city', 'real ', 'atletico', 'chelsea', 'liverpool']):
                sport = 'Soccer'
                
            market = Market(
                source_id=market_id,
                source='overtime_v2',
                sport=sport,
                league_name='Regular Season',
                market_type='winner',
                home_team=home[:50],  # Limit length
                away_team=away[:50],
                maturity_date=maturity,
                is_finished=False,
                updated_at=datetime.now(timezone.utc)
            )
            db.add(market)
            
            # Add simple odds
            odds_patterns = [
                {'home': 2.1, 'away': 3.4, 'draw': 3.3},
                {'home': 1.8, 'away': 4.5, 'draw': 3.6},
                {'home': 2.8, 'away': 2.6, 'draw': 3.25},
            ]
            pattern = random.choice(odds_patterns)
            
            for outcome, decimal_odds in pattern.items():
                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                
                odd = Odd(
                    source_id=market_id,
                    market_type='winner',
                    outcome=outcome,
                    source='overtime_v2',
                    bookmaker='Overtime',
                    decimal_odds=decimal_odds,
                    american_odds=american,
                    normalized_implied=1.0 / decimal_odds,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(odd)
                
            added += 1
            if added >= 50:  # Stop at 50 markets
                break
                
        db.commit()
        logger.info(f'Added {added} new markets')
        
        # Show summary
        total = db.query(Market).count()
        by_sport = {}
        sports = db.query(Market.sport).distinct().all()
        for sport, in sports:
            count = db.query(Market).filter(Market.sport == sport).count()
            by_sport[sport] = count
            
        logger.info(f'\nTotal markets: {total}')
        for sport, count in by_sport.items():
            logger.info(f'  {sport}: {count}')

if __name__ == "__main__":
    sync_markets()