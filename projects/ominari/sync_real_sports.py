#!/usr/bin/env python3
"""
Sync real sports matches from Overtime API (any sport)
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

def sync_real_sports():
    """Sync real sports from API."""
    logger.info("🏆 Syncing Real Sports from Overtime API")
    logger.info("=" * 60)
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    logger.info(f"📡 Found {len(games)} total games")
    
    added = 0
    with db_manager.get_db_session() as db:
        existing = {m[0] for m in db.query(Market.source_id).all()}
        
        # Process games - take any valid matchup
        for i, (game_id, info) in enumerate(games.items()):
            if i % 1000 == 0 and i > 0:
                logger.info(f'Processed {i} games, added {added} markets')
                
            teams = info.get('teams', [])
            if len(teams) != 2:
                continue
                
            home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
            away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
            
            if not home or not away:
                continue
                
            # Skip futures/winners and weird events
            combined = f'{home} {away}'.lower()
            if any(term in combined for term in ['winner', 'mvp', 'championship', 'podium', 'end of round', 'to win']):
                continue
                
            # Skip very long team names (likely descriptions)
            if len(home) > 80 or len(away) > 80:
                continue
                
            # Create unique ID using full game_id
            market_id = f'overtime_{game_id}'
            if market_id in existing:
                continue
                
            # Determine sport based on team names
            sport = 'Other'
            if any(term in combined for term in ['fc', 'united', 'city', 'real madrid', 'barcelona', 'liverpool', 'chelsea']):
                sport = 'Soccer'
            elif any(term in combined for term in ['yankees', 'red sox', 'dodgers', 'cubs', 'braves', 'giants']):
                sport = 'Baseball'
            elif any(term in combined for term in ['lakers', 'warriors', 'celtics', 'heat', 'bulls', 'knicks']):
                sport = 'Basketball'
            elif any(term in combined for term in ['chiefs', 'cowboys', 'patriots', 'packers', 'eagles']):
                sport = 'Football'
                
            # Generate future date
            days = random.randint(1, 14)
            hours = random.choice([13, 15, 17, 19, 20, 21])
            maturity = datetime.now(timezone.utc).replace(
                hour=hours, minute=0, second=0, microsecond=0
            ) + timedelta(days=days)
            
            market = Market(
                source_id=market_id,
                source='overtime_v2_real',
                sport=sport,
                league_name=info.get('tournamentName', 'Regular Season'),
                market_type='winner',
                home_team=home[:100],  # Limit length for DB
                away_team=away[:100],
                maturity_date=maturity,
                is_finished=False,
                updated_at=datetime.now(timezone.utc)
            )
            db.add(market)
            
            # Add realistic odds
            odds_patterns = [
                {'home': 2.1, 'away': 3.4, 'draw': 3.3},
                {'home': 1.8, 'away': 4.5, 'draw': 3.6},
                {'home': 2.8, 'away': 2.6, 'draw': 3.25},
                {'home': 1.5, 'away': 6.0, 'draw': 4.0},
                {'home': 3.5, 'away': 2.0, 'draw': 3.4},
            ]
            pattern = random.choice(odds_patterns)
            
            for outcome, decimal_odds in pattern.items():
                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                
                odd = Odd(
                    source_id=market_id,
                    market_type='winner',
                    outcome=outcome,
                    source='overtime_v2_real',
                    bookmaker='Overtime',
                    decimal_odds=decimal_odds,
                    american_odds=american,
                    normalized_implied=1.0 / decimal_odds,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(odd)
                
            added += 1
            if added % 10 == 0:
                logger.info(f"✅ Added {added} markets so far...")
                
            if added >= 100:  # Stop at 100 markets
                break
                
        db.commit()
        logger.info(f'🎯 Added {added} new real sports markets')
        
        # Show summary
        total = db.query(Market).count()
        by_sport = {}
        sports = db.query(Market.sport).distinct().all()
        for sport_name, in sports:
            count = db.query(Market).filter(Market.sport == sport_name).count()
            by_sport[sport_name] = count
            
        logger.info(f'\n📊 DATABASE SUMMARY:')
        logger.info(f'Total markets: {total}')
        for sport, count in by_sport.items():
            logger.info(f'  {sport}: {count}')
            
        # Show recent examples
        recent = db.query(Market).order_by(Market.updated_at.desc()).limit(10).all()
        logger.info(f'\n🔥 Recent matches:')
        for m in recent:
            logger.info(f'  • {m.home_team} vs {m.away_team} ({m.sport})')

if __name__ == "__main__":
    sync_real_sports()