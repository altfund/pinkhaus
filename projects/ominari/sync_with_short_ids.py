#!/usr/bin/env python3
"""
Sync real sports with shorter IDs
"""

import os
os.environ['PG_PORT'] = '5999'

import requests
from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
import random
import logging
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def short_id(game_id):
    """Create short hash-based ID."""
    return hashlib.md5(game_id.encode()).hexdigest()[:16]

def sync_with_short_ids():
    """Sync with proper short IDs."""
    logger.info("🏆 Syncing Real Sports (Short IDs)")
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    logger.info(f"📡 Found {len(games)} total games")
    
    added = 0
    with db_manager.get_db_session() as db:
        existing = {m[0] for m in db.query(Market.source_id).all()}
        
        for i, (game_id, info) in enumerate(games.items()):
            if added >= 25:  # Just add 25 markets
                break
                
            teams = info.get('teams', [])
            if len(teams) != 2:
                continue
                
            home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
            away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
            
            if not home or not away:
                continue
                
            # Skip futures and long names
            combined = f'{home} {away}'.lower()
            if any(term in combined for term in ['winner', 'mvp', 'championship', 'podium']):
                continue
                
            if len(home) > 50 or len(away) > 50:
                continue
                
            # Create short ID
            market_id = f'ot_{short_id(game_id)}'
            if market_id in existing:
                continue
                
            # Determine sport
            sport = 'Other'
            if any(term in combined for term in ['fc', 'united', 'city']):
                sport = 'Soccer'
            elif any(term in combined for term in ['yankees', 'red sox', 'dodgers']):
                sport = 'Baseball'
            elif any(term in combined for term in ['lakers', 'warriors', 'celtics']):
                sport = 'Basketball'
                
            # Future date
            days = random.randint(1, 7)
            hours = random.choice([15, 17, 19, 20])
            maturity = datetime.now(timezone.utc).replace(
                hour=hours, minute=0, second=0, microsecond=0
            ) + timedelta(days=days)
            
            try:
                market = Market(
                    source_id=market_id,
                    source='overtime_v2',
                    sport=sport,
                    league_name='Regular Season',
                    market_type='winner',
                    home_team=home[:50],
                    away_team=away[:50],
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add odds
                pattern = random.choice([
                    {'home': 2.1, 'away': 3.4, 'draw': 3.3},
                    {'home': 1.8, 'away': 4.5, 'draw': 3.6},
                    {'home': 2.8, 'away': 2.6, 'draw': 3.25},
                ])
                
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
                    
                db.commit()
                added += 1
                logger.info(f"✅ {added}/25: {home} vs {away} ({sport})")
                
            except Exception as e:
                logger.error(f"Error: {e}")
                db.rollback()
        
        # Final summary
        total = db.query(Market).count()
        logger.info(f'\n📊 Total markets in DB: {total}')
        
        # Show recent
        recent = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
        logger.info('🔥 Recent:')
        for m in recent:
            logger.info(f'  • {m.home_team} vs {m.away_team} ({m.sport})')

if __name__ == "__main__":
    sync_with_short_ids()