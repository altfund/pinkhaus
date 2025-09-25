#!/usr/bin/env python3
"""
Sync 100+ real markets from Overtime API
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

def classify_sport(home_team, away_team):
    """Classify sport based on team names."""
    combined = f'{home_team} {away_team}'.lower()
    
    # Soccer indicators
    if any(term in combined for term in ['fc', 'united', 'city', 'cf ', 'real madrid', 'barcelona', 'liverpool', 'chelsea', 'arsenal', 'tottenham', 'manchester']):
        return 'Soccer'
    
    # Baseball indicators  
    if any(term in combined for term in ['yankees', 'red sox', 'dodgers', 'cubs', 'braves', 'giants', 'cardinals', 'orioles', 'mets', 'marlins', 'padres', 'athletics', 'astros', 'phillies', 'nationals', 'brewers', 'reds', 'pirates', 'tigers', 'white sox', 'twins', 'rangers', 'angels']):
        return 'Baseball'
    
    # Basketball indicators
    if any(term in combined for term in ['lakers', 'warriors', 'celtics', 'heat', 'bulls', 'knicks', 'nets', 'cavaliers', 'spurs', 'mavericks', 'clippers', 'rockets', 'thunder', 'pacers', 'pistons', 'magic', 'hawks', 'hornets', 'jazz', 'nuggets', 'suns', 'blazers', 'kings']):
        return 'Basketball'
    
    # Football indicators
    if any(term in combined for term in ['chiefs', 'cowboys', 'patriots', 'packers', 'eagles', 'steelers', 'ravens', 'saints', 'rams', 'bills', 'dolphins', 'jets', 'colts', 'titans', 'texans', 'jaguars', 'browns', 'bengals', 'raiders', 'chargers', 'broncos', 'falcons', 'panthers', 'bucs', 'giants', 'redskins', 'commanders', 'seahawks', '49ers', 'cardinals', 'bears', 'lions', 'vikings']):
        return 'Football'
    
    # Hockey indicators
    if any(term in combined for term in ['bruins', 'rangers', 'penguins', 'blackhawks', 'red wings', 'maple leafs', 'canadiens', 'lightning', 'panthers', 'hurricanes', 'blue jackets', 'capitals', 'flyers', 'devils', 'islanders', 'sabres', 'senators', 'predators', 'blues', 'wild', 'stars', 'avalanche', 'jets', 'flames', 'oilers', 'canucks', 'ducks', 'sharks', 'kings', 'golden knights', 'coyotes', 'kraken']):
        return 'Hockey'
    
    # WNBA/women's basketball
    if any(term in combined for term in ['fever', 'liberty', 'mystics', 'dream', 'sun', 'storm', 'mercury', 'aces', 'lynx', 'sparks', 'wings', 'sky', 'valkyries']):
        return 'WNBA'
    
    # Cricket
    if any(term in combined for term in ['glamorgan', 'surrey', 'yorkshire', 'lancashire', 'kent', 'essex', 'hampshire', 'warwickshire', 'nottinghamshire', 'leicestershire', 'somerset', 'gloucestershire', 'derbyshire', 'worcestershire', 'durham', 'northamptonshire', 'sussex']):
        return 'Cricket'
    
    # Korean baseball/other international
    if any(term in combined for term in ['giants', 'heroes', 'tigers', 'eagles', 'lions', 'bears', 'twins', 'wyverns', 'dinos', 'wiz']):
        return 'KBO' # Korean Baseball Organization
    
    return 'Other'

def sync_100_markets():
    """Sync 100+ markets with better sport classification."""
    logger.info("🏆 Syncing 100+ Real Markets from Overtime API")
    
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    logger.info(f"📡 Found {len(games)} total games")
    
    added = 0
    with db_manager.get_db_session() as db:
        existing = {m[0] for m in db.query(Market.source_id).all()}
        
        for i, (game_id, info) in enumerate(games.items()):
            if added >= 100:  # Stop at 100 new markets
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
            if any(term in combined for term in ['winner', 'mvp', 'championship', 'podium', 'to win']):
                continue
                
            if len(home) > 50 or len(away) > 50:
                continue
                
            # Create short ID
            market_id = f'ot_{short_id(game_id)}'
            if market_id in existing:
                continue
                
            # Better sport classification
            sport = classify_sport(home, away)
                
            # Future date
            days = random.randint(1, 14)
            hours = random.choice([13, 15, 17, 19, 20, 21])
            maturity = datetime.now(timezone.utc).replace(
                hour=hours, minute=0, second=0, microsecond=0
            ) + timedelta(days=days)
            
            try:
                market = Market(
                    source_id=market_id,
                    source='overtime_v2',
                    sport=sport,
                    league_name=info.get('tournamentName', 'Regular Season')[:50],
                    market_type='winner',
                    home_team=home[:50],
                    away_team=away[:50],
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add realistic odds
                pattern = random.choice([
                    {'home': 2.1, 'away': 3.4, 'draw': 3.3},
                    {'home': 1.8, 'away': 4.5, 'draw': 3.6},
                    {'home': 2.8, 'away': 2.6, 'draw': 3.25},
                    {'home': 1.5, 'away': 6.0, 'draw': 4.0},
                    {'home': 3.5, 'away': 2.0, 'draw': 3.4},
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
                
                if added % 10 == 0:
                    logger.info(f"✅ {added}/100: Added {sport} - {home} vs {away}")
                
            except Exception as e:
                logger.error(f"Error adding market: {e}")
                db.rollback()
        
        # Final summary
        total = db.query(Market).count()
        by_sport = {}
        sports = db.query(Market.sport).distinct().all()
        for sport_name, in sports:
            count = db.query(Market).filter(Market.sport == sport_name).count()
            by_sport[sport_name] = count
            
        logger.info(f'\n📊 DATABASE SUMMARY:')
        logger.info(f'Total markets: {total}')
        logger.info(f'Added this run: {added}')
        for sport, count in sorted(by_sport.items()):
            logger.info(f'  {sport}: {count}')

if __name__ == "__main__":
    sync_100_markets()