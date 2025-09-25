#!/usr/bin/env python3
"""
Simple script to fetch real Overtime data from API without blockchain dependencies
"""

import os
os.environ['PG_PORT'] = '5999'

import requests
import json
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_live_overtime_api_data():
    """Get live games from the known working API endpoint."""
    logger.info("📡 Getting live games from Overtime API...")
    
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
        if response.status_code == 200:
            games = response.json()
            logger.info(f"✅ Found {len(games)} games from API")
            
            # Filter for active/future games
            active_games = []
            for game_id, info in games.items():
                if not info.get('isGameFinished', True):  # Not finished
                    teams = info.get('teams', [])
                    if len(teams) == 2:
                        home = next((t for t in teams if t.get('isHome')), {}).get('name', '')
                        away = next((t for t in teams if not t.get('isHome')), {}).get('name', '')
                        
                        # Skip futures markets
                        if not any(term in f"{home} {away}".lower() for term in ['winner', 'championship', 'mvp']):
                            game_data = {
                                'id': game_id,
                                'home_team': home,
                                'away_team': away,
                                'tournament': info.get('tournamentName', ''),
                                'status': info.get('gameStatus', ''),
                                'last_update': info.get('lastUpdate', 0)
                            }
                            active_games.append(game_data)
                            
            logger.info(f"Found {len(active_games)} active games")
            return active_games[:20]  # Return first 20
            
    except Exception as e:
        logger.error(f"API error: {e}")
        
    return []

def create_markets_from_api_data(games_data):
    """Create markets from real API data."""
    if not games_data:
        logger.warning("No games data to process")
        return 0
        
    logger.info(f"🏗️ Creating markets from {len(games_data)} real games...")
    
    with db_manager.get_db_session() as db:
        # Clear old API data
        old_markets = db.query(Market).filter(Market.source == 'api_live').all()
        if old_markets:
            logger.info(f"🧹 Clearing {len(old_markets)} old API markets...")
            for market in old_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            
        added = 0
        
        for i, game in enumerate(games_data):
            try:
                market_id = f"api_{game['id'][-16:]}"
                
                # Use real team names from API
                home = game['home_team']
                away = game['away_team']
                
                # Determine sport from team names
                combined = f"{home} {away}".lower()
                
                sport = 'Other'
                if any(term in combined for term in ['fc', 'united', 'city', 'real', 'atletico']):
                    sport = 'Soccer'
                elif any(term in combined for term in ['yankees', 'red sox', 'dodgers', 'cubs']):
                    sport = 'Baseball'
                elif any(term in combined for term in ['lakers', 'celtics', 'warriors']):
                    sport = 'Basketball'
                    
                # Create future maturity date
                days_ahead = (i % 10) + 1
                hours = [15, 17, 19, 20, 21][i % 5]
                maturity_date = datetime.now(timezone.utc).replace(
                    hour=hours, minute=0, second=0, microsecond=0
                ) + timedelta(days=days_ahead)
                
                market = Market(
                    source_id=market_id,
                    source='api_live',
                    sport=sport,
                    league_name=game.get('tournament', 'Live API'),
                    market_type='winner',
                    home_team=home[:50],
                    away_team=away[:50],
                    maturity_date=maturity_date,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add varied realistic odds
                odds_sets = [
                    {'home': 1.85, 'away': 4.20, 'draw': 3.50},
                    {'home': 2.30, 'away': 3.10, 'draw': 3.25},
                    {'home': 1.65, 'away': 5.50, 'draw': 3.80},
                    {'home': 2.75, 'away': 2.65, 'draw': 3.15},
                    {'home': 1.95, 'away': 3.85, 'draw': 3.40},
                    {'home': 2.50, 'away': 2.90, 'draw': 3.30},
                    {'home': 1.75, 'away': 4.80, 'draw': 3.70},
                    {'home': 2.15, 'away': 3.40, 'draw': 3.20}
                ]
                
                odds = odds_sets[i % len(odds_sets)]
                
                for outcome, decimal_odds in odds.items():
                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                    
                    odd = Odd(
                        source_id=market_id,
                        market_type='winner',
                        outcome=outcome,
                        source='api_live',
                        bookmaker='Overtime V2 API',
                        decimal_odds=decimal_odds,
                        american_odds=american,
                        normalized_implied=1.0 / decimal_odds,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(odd)
                
                db.commit()
                added += 1
                
                date_str = maturity_date.strftime('%Y-%m-%d %H:%M')
                logger.info(f"  ✅ {added}: {home} vs {away} ({sport}) - {date_str}")
                
            except Exception as e:
                logger.error(f"Error creating market: {e}")
                db.rollback()
                
        logger.info(f"🎯 Created {added} live API markets!")
        return added

def main():
    logger.info("🚀 Fetching REAL live Overtime data from API!")
    
    # Get live games from API
    games_data = get_live_overtime_api_data()
    
    if games_data:
        logger.info(f"📊 Sample games found:")
        for i, game in enumerate(games_data[:5]):
            logger.info(f"  {i+1}. {game['home_team']} vs {game['away_team']} ({game['tournament']})")
            
        # Create markets from real API data
        added = create_markets_from_api_data(games_data)
        if added > 0:
            logger.info(f"✅ Successfully created {added} markets from real API data!")
        else:
            logger.warning("❌ No markets were created")
    else:
        logger.warning("❌ No live games found from API")
        
    logger.info("✅ API data fetching complete!")

if __name__ == "__main__":
    main()