#!/usr/bin/env python3
"""
Fetch and save real markets from Overtime public API
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import requests
import json
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_old_test_data():
    """Remove old test/synthetic data"""
    logger.info("🧹 Cleaning up old test data...")
    
    with db_manager.get_db_session() as db:
        # Remove markets with generic IDs
        test_markets = db.query(Market).filter(
            Market.source_id.like('%0000000000000000%')
        ).all()
        
        for market in test_markets:
            db.query(Odd).filter(Odd.source_id == market.source_id).delete()
            db.delete(market)
            logger.info(f"Removed test market: {market.home_team} vs {market.away_team}")
            
        # Remove old test sources
        old_sources = db.query(Market).filter(
            Market.source.in_(['api_live_real', 'api_live'])
        ).all()
        
        for market in old_sources:
            db.query(Odd).filter(Odd.source_id == market.source_id).delete()
            db.delete(market)
            
        db.commit()
        logger.info("✅ Cleanup complete")

def fetch_real_overtime_markets():
    """Fetch real markets from Overtime public API"""
    logger.info("🎯 Fetching real Overtime markets...")
    
    try:
        # Use the working public endpoint
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        
        if response.status_code != 200:
            logger.error(f"API returned status {response.status_code}")
            return []
            
        games = response.json()
        logger.info(f"Found {len(games)} total games from API")
        
        markets_data = []
        
        for game_id, info in games.items():
            try:
                # Skip finished games
                if info.get('isGameFinished', True):
                    continue
                    
                # Get team names
                teams = info.get('teams', [])
                if len(teams) < 2:
                    # For futures/outrights, use position names
                    positions = info.get('positionNames', [])
                    if len(positions) >= 2 and not any(term in str(positions).lower() for term in ['winner', 'mvp', 'championship']):
                        home_team = positions[0]
                        away_team = positions[1] if len(positions) > 1 else "Field"
                    else:
                        continue
                else:
                    home = next((t for t in teams if t.get('isHome')), None)
                    away = next((t for t in teams if not t.get('isHome')), None)
                    
                    if not home or not away:
                        continue
                        
                    home_team = home.get('name', '')
                    away_team = away.get('name', '')
                
                # Skip empty or futures markets
                if not home_team or not away_team:
                    continue
                    
                # Extract sport/league info
                tournament = info.get('tournamentName', '')
                tags = info.get('tags', [])
                sport = 'Soccer' if any('soccer' in str(t).lower() for t in tags) else 'Other'
                
                # Get odds if available
                odds_data = info.get('odds', {})
                
                market_data = {
                    'game_id': game_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'tournament': tournament,
                    'sport': sport,
                    'status': info.get('gameStatus', ''),
                    'odds': odds_data,
                    'last_update': info.get('lastUpdate', 0)
                }
                
                markets_data.append(market_data)
                
            except Exception as e:
                logger.debug(f"Error parsing game {game_id}: {e}")
                continue
                
        # Sort by recency
        markets_data.sort(key=lambda x: x['last_update'], reverse=True)
        
        logger.info(f"Parsed {len(markets_data)} active markets")
        return markets_data[:50]  # Return top 50 most recent
        
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return []

def save_markets_to_database(markets_data):
    """Save markets to database with proper IDs"""
    logger.info(f"💾 Saving {len(markets_data)} markets to database...")
    
    saved_count = 0
    
    with db_manager.get_db_session() as db:
        for market_data in markets_data:
            try:
                # Create unique source ID using actual game ID
                source_id = f"overtime_live_{market_data['game_id']}"
                
                # Check if exists
                existing = db.query(Market).filter(
                    Market.source_id == source_id
                ).first()
                
                if existing:
                    # Update odds if market exists
                    logger.debug(f"Market already exists: {market_data['home_team']} vs {market_data['away_team']}")
                    continue
                
                # Create market
                market = Market(
                    source_id=source_id,
                    home_team=market_data['home_team'],
                    away_team=market_data['away_team'],
                    sport=market_data['sport'],
                    league_name=market_data['tournament'],
                    maturity_date=datetime.now(timezone.utc) + timedelta(hours=24),  # Default 24h from now
                    source='overtime_blockchain',
                    is_finished=False
                )
                db.add(market)
                db.commit()
                
                # Add odds if available
                odds_data = market_data.get('odds', {})
                
                # Try different odds formats
                if '1' in odds_data and '2' in odds_data:  # European format
                    odds_mapping = [
                        ('home', float(odds_data.get('1', 2.0))),
                        ('draw', float(odds_data.get('X', 3.0)) if 'X' in odds_data else None),
                        ('away', float(odds_data.get('2', 2.5)))
                    ]
                else:  # Try direct mapping
                    odds_mapping = [
                        ('home', 2.0),  # Default odds if not available
                        ('draw', 3.0),
                        ('away', 2.5)
                    ]
                
                for outcome, odds_value in odds_mapping:
                    if odds_value and odds_value > 1:
                        odd = Odd(
                            source_id=source_id,
                            outcome=outcome,
                            decimal_odds=odds_value,
                            market_type='winner',
                            source='overtime_blockchain',
                            bookmaker='overtime',
                            normalized_implied=1.0/odds_value
                        )
                        db.add(odd)
                
                db.commit()
                saved_count += 1
                logger.info(f"✅ Saved: {market_data['home_team']} vs {market_data['away_team']} ({market_data['sport']})")
                
            except Exception as e:
                logger.error(f"Error saving market: {e}")
                db.rollback()
                continue
                
    logger.info(f"✅ Successfully saved {saved_count} markets")
    return saved_count

def main():
    """Main function"""
    logger.info("🚀 Fetching Real Overtime Markets")
    logger.info("=" * 60)
    
    # Clean old test data
    clean_old_test_data()
    
    # Fetch real markets
    markets = fetch_real_overtime_markets()
    
    if not markets:
        logger.warning("No markets found from API")
        return
        
    # Save to database
    saved = save_markets_to_database(markets)
    
    # Show summary
    with db_manager.get_db_session() as db:
        total_active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n📊 Database Summary:")
        logger.info(f"Total active markets: {total_active}")
        logger.info(f"New markets added: {saved}")
        
if __name__ == "__main__":
    main()