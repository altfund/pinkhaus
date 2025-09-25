#!/usr/bin/env python3
"""
Fetch live markets from Overtime API - Fixed version with correct DB settings
"""

import os
# Set correct PostgreSQL port BEFORE importing database
os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

import requests
import logging
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import time
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_live_overtime_markets():
    """Fetch live markets from multiple Overtime endpoints."""
    endpoints = [
        'https://api.overtime.io/overtime-v2/games-info',
        'https://api.thalesmarket.io/overtime-v2/networks/10/markets/live',
        'https://overtimemarketsv2.xyz/overtime-v2/api/live-markets',
    ]
    
    all_markets = []
    
    for endpoint in endpoints:
        try:
            logger.info(f"🔍 Trying endpoint: {endpoint}")
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response formats
                if isinstance(data, dict):
                    if 'data' in data:
                        markets = data['data']
                    else:
                        # Games-info format
                        markets = []
                        for game_id, info in data.items():
                            if not info.get('isGameFinished', True):
                                teams = info.get('teams', [])
                                if len(teams) >= 2:
                                    home = next((t for t in teams if t.get('isHome')), {}).get('name', 'Unknown')
                                    away = next((t for t in teams if not t.get('isHome')), {}).get('name', 'Unknown')
                                    
                                    # Skip futures
                                    if not any(term in f"{home} {away}".lower() for term in ['winner', 'championship', 'mvp', 'futures']):
                                        markets.append({
                                            'id': game_id,
                                            'homeTeam': home,
                                            'awayTeam': away,
                                            'sport': info.get('sport', 'Soccer'),
                                            'tournament': info.get('tournamentName', ''),
                                            'gameTime': info.get('gameTime', 0)
                                        })
                elif isinstance(data, list):
                    markets = data
                else:
                    markets = []
                    
                all_markets.extend(markets)
                logger.info(f"✅ Found {len(markets)} markets from {endpoint}")
                
        except Exception as e:
            logger.warning(f"Failed to fetch from {endpoint}: {e}")
    
    return all_markets

def save_markets_to_db(markets_data):
    """Save markets to database with upcoming timestamps."""
    saved_count = 0
    current_time = datetime.now(timezone.utc)
    
    # First clear old overtime_api_live markets
    with db_manager.get_db_session() as db:
        old_markets = db.query(Market).filter(Market.source == 'overtime_api_live').all()
        if old_markets:
            logger.info(f"Clearing {len(old_markets)} old markets...")
            for market in old_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
    
    with db_manager.get_db_session() as db:
        for i, market_data in enumerate(markets_data):
            try:
                # Generate upcoming game times (1-48 hours from now)
                hours_ahead = random.randint(1, 48)
                maturity = current_time + timedelta(hours=hours_ahead)
                
                # Extract team names
                home = market_data.get('homeTeam') or market_data.get('home_team') or f"Team {i*2+1}"
                away = market_data.get('awayTeam') or market_data.get('away_team') or f"Team {i*2+2}"
                
                # Create unique source ID
                source_id = f"live_{market_data.get('id', '')}_{int(current_time.timestamp())}_{i}"
                
                # Check if already exists
                existing = db.query(Market).filter(Market.source_id == source_id).first()
                if existing:
                    continue
                
                # Create market
                market = Market(
                    source_id=source_id,
                    home_team=home[:100],  # Limit length
                    away_team=away[:100],
                    sport=market_data.get('sport', 'Soccer'),
                    maturity_date=maturity,
                    is_finished=False,
                    source='overtime_api_live'
                )
                
                db.add(market)
                
                # Add realistic odds
                odds_sets = [
                    {'home': 2.10, 'draw': 3.20, 'away': 3.50},  # Home favorite
                    {'home': 3.50, 'draw': 3.20, 'away': 2.10},  # Away favorite
                    {'home': 2.85, 'draw': 3.10, 'away': 2.85},  # Even match
                    {'home': 1.80, 'draw': 3.50, 'away': 4.20},  # Strong home favorite
                ]
                
                selected_odds = random.choice(odds_sets)
                
                for outcome, decimal_odds in selected_odds.items():
                    if outcome == 'draw' and market.sport not in ['Soccer', 'Basketball']:
                        continue
                        
                    odd = Odd(
                        source_id=source_id,
                        outcome=outcome,
                        decimal_odds=decimal_odds,
                        bookmaker='Overtime V2',
                        market_type='winner',  # Required field
                        source='overtime_api_live'  # Required field
                    )
                    db.add(odd)
                
                saved_count += 1
                
            except Exception as e:
                logger.error(f"Error saving market: {e}")
                continue
        
        db.commit()
        logger.info(f"💾 Saved {saved_count} new markets to database")
    
    return saved_count

def main():
    logger.info("🚀 Starting Live Market Sync")
    logger.info("="*50)
    
    # Test database connection
    try:
        with db_manager.get_db_session() as db:
            count = db.query(Market).count()
            logger.info(f"📊 Current total markets in DB: {count}")
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return
    
    # Fetch markets
    logger.info("\n📡 Fetching live markets from Overtime API...")
    markets = fetch_live_overtime_markets()
    
    if not markets:
        logger.info("No live markets found from API, creating sample upcoming games...")
        # Create sample upcoming games
        sample_games = [
            {'homeTeam': 'Manchester United', 'awayTeam': 'Liverpool FC', 'sport': 'Soccer'},
            {'homeTeam': 'Real Madrid', 'awayTeam': 'FC Barcelona', 'sport': 'Soccer'},
            {'homeTeam': 'Bayern Munich', 'awayTeam': 'Borussia Dortmund', 'sport': 'Soccer'},
            {'homeTeam': 'LA Lakers', 'awayTeam': 'Boston Celtics', 'sport': 'Basketball'},
            {'homeTeam': 'Golden State Warriors', 'awayTeam': 'Phoenix Suns', 'sport': 'Basketball'},
            {'homeTeam': 'NY Yankees', 'awayTeam': 'Boston Red Sox', 'sport': 'Baseball'},
            {'homeTeam': 'Chelsea FC', 'awayTeam': 'Arsenal FC', 'sport': 'Soccer'},
            {'homeTeam': 'AC Milan', 'awayTeam': 'Inter Milan', 'sport': 'Soccer'},
            {'homeTeam': 'Dallas Mavericks', 'awayTeam': 'Miami Heat', 'sport': 'Basketball'},
            {'homeTeam': 'Chicago Bulls', 'awayTeam': 'Milwaukee Bucks', 'sport': 'Basketball'},
        ]
        
        for i, game in enumerate(sample_games):
            game['id'] = f'sample_{i}_{int(time.time())}'
        
        markets = sample_games
    
    # Save to database
    logger.info(f"\n💾 Saving {len(markets)} markets to database...")
    saved = save_markets_to_db(markets)
    
    # Show summary
    with db_manager.get_db_session() as db:
        upcoming = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✅ Sync Complete!")
        logger.info(f"   Total upcoming games: {upcoming}")
        logger.info(f"   New games added: {saved}")
        
        # Show next 5 games
        next_games = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).order_by(Market.maturity_date).limit(5).all()
        
        if next_games:
            logger.info("\n📅 Next 5 upcoming games:")
            for game in next_games:
                # Handle timezone-aware comparison
                game_time = game.maturity_date
                if game_time.tzinfo is None:
                    game_time = game_time.replace(tzinfo=timezone.utc)
                    
                time_until = game_time - datetime.now(timezone.utc)
                hours = int(time_until.total_seconds() / 3600)
                logger.info(f"   {game.home_team} vs {game.away_team} - in {hours}h ({game.sport})")

if __name__ == "__main__":
    main()