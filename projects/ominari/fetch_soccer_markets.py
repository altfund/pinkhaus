#!/usr/bin/env python3
"""
Fetch soccer markets specifically from Overtime
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_soccer_markets_from_overtime():
    """Fetch soccer markets from Overtime API"""
    logger.info("⚽ Fetching soccer markets from Overtime...")
    
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            return []
            
        games = response.json()
        logger.info(f"Found {len(games)} total games")
        
        soccer_markets = []
        
        for game_id, info in games.items():
            try:
                # Skip finished games
                if info.get('isGameFinished', True):
                    continue
                
                # Check if it's soccer
                tags = info.get('tags', [])
                tournament = info.get('tournamentName', '').lower()
                sport_id = info.get('sport', info.get('sportId'))
                
                # Sport ID 4 or 9004 is soccer in Overtime
                # Also check tournament names and tags
                is_soccer = (
                    sport_id in [4, 9004] or
                    any('soccer' in str(tag).lower() for tag in tags) or
                    any(league in tournament for league in ['premier league', 'la liga', 'bundesliga', 'serie a', 'ligue 1', 'champions league', 'europa league', 'world cup', 'euro', 'copa', 'mls', 'championship', 'eredivisie', 'primeira liga', 'scottish', 'brazilian', 'argentine']) or
                    ('football' in tournament and 'american' not in tournament)  # International football, not American
                )
                
                # Skip if it's esports
                if any(esport_term in tournament for esport_term in ['cct', 'esport', 'pro league', 'esl', 'blast', 'pgl', 'iem']):
                    continue
                
                if not is_soccer:
                    continue
                
                teams = info.get('teams', [])
                if len(teams) >= 2:
                    home = next((t for t in teams if t.get('isHome')), None)
                    away = next((t for t in teams if not t.get('isHome')), None)
                    
                    if home and away:
                        soccer_markets.append({
                            'game_id': game_id,
                            'home_team': home.get('name', ''),
                            'away_team': away.get('name', ''),
                            'tournament': info.get('tournamentName', ''),
                            'last_update': info.get('lastUpdate', 0),
                            'sport_id': sport_id,
                            'maturity_date': info.get('maturityDate', info.get('startTime'))
                        })
                        
            except Exception as e:
                logger.debug(f"Error parsing game {game_id}: {e}")
                continue
        
        logger.info(f"Found {len(soccer_markets)} soccer markets")
        return soccer_markets
        
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return []

def save_soccer_markets(markets):
    """Save soccer markets to database"""
    logger.info(f"💾 Saving {len(markets)} soccer markets...")
    
    saved_count = 0
    
    with db_manager.get_db_session() as db:
        for market in markets:
            try:
                source_id = f"overtime_soccer_{market['game_id']}"
                
                # Check if exists
                existing = db.query(Market).filter(
                    Market.source_id == source_id
                ).first()
                
                if existing:
                    continue
                
                # Parse maturity date if available
                maturity_str = market.get('maturity_date')
                if maturity_str:
                    try:
                        if isinstance(maturity_str, (int, float)):
                            # Unix timestamp
                            maturity = datetime.fromtimestamp(maturity_str, tz=timezone.utc)
                        else:
                            # ISO string
                            maturity = datetime.fromisoformat(maturity_str.replace('Z', '+00:00'))
                    except:
                        maturity = datetime.now(timezone.utc) + timedelta(hours=24)
                else:
                    maturity = datetime.now(timezone.utc) + timedelta(hours=24)
                
                # Skip if match already started
                if maturity < datetime.now(timezone.utc):
                    continue
                
                # Create market
                new_market = Market(
                    source_id=source_id,
                    home_team=market['home_team'],
                    away_team=market['away_team'],
                    sport='Soccer',
                    league_name=market['tournament'],
                    maturity_date=maturity,
                    source='overtime_soccer',
                    is_finished=False
                )
                db.add(new_market)
                db.commit()
                
                # Add default odds
                for outcome, odds in [('home', 2.1), ('draw', 3.2), ('away', 3.5)]:
                    odd = Odd(
                        source_id=source_id,
                        outcome=outcome,
                        decimal_odds=odds,
                        market_type='winner',
                        source='overtime_soccer',
                        bookmaker='overtime',
                        normalized_implied=1.0/odds
                    )
                    db.add(odd)
                
                db.commit()
                saved_count += 1
                logger.info(f"✅ Saved: {market['home_team']} vs {market['away_team']} ({market['tournament']})")
                
            except Exception as e:
                logger.error(f"Error saving market: {e}")
                db.rollback()
                
    return saved_count

def fetch_from_blockchain():
    """Also check blockchain for soccer markets"""
    logger.info("🔗 Checking blockchain for soccer markets...")
    
    # This would connect to blockchain - simplified for now
    # The sync_blockchain_data.py script already does this
    
    with db_manager.get_db_session() as db:
        blockchain_soccer = db.query(Market).filter(
            Market.source.like('%blockchain%'),
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).count()
        
        logger.info(f"Found {blockchain_soccer} blockchain soccer markets")

def main():
    logger.info("⚽ Soccer Market Fetcher")
    logger.info("=" * 60)
    
    # Fetch from API
    markets = fetch_soccer_markets_from_overtime()
    
    if markets:
        saved = save_soccer_markets(markets)
        logger.info(f"\n✅ Added {saved} new soccer markets")
    else:
        logger.info("\n❌ No soccer markets found")
    
    # Check blockchain too
    fetch_from_blockchain()
    
    # Summary
    with db_manager.get_db_session() as db:
        total_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n📊 Total active soccer markets: {total_soccer}")

if __name__ == "__main__":
    main()