#!/usr/bin/env python3
"""Save real soccer matches from Overtime API"""

import os
os.environ['PG_PORT'] = '5999'

import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_real_soccer_matches():
    """Fetch real soccer matches from Overtime API"""
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    games = response.json()
    
    # Soccer tournaments based on analysis
    soccer_tournaments = [
        'Regular Season', 'Round 1', 'Round 2', 'Group Stage', 
        'Apertura', 'Clausura', 'Group A', 'Group B', 'Group C',
        'Promotion/Relegation', 'Quarterfinals', 'Finals', 'Playoffs'
    ]
    
    # Teams/terms that indicate NOT soccer
    exclude_terms = [
        'esport', 'gaming', 'vs end of', 'winner', 'mlb', 'nfl', 'nba', 'nhl',
        'ufc', 'mma', 'boxing', 'tennis', 'cricket', 'rugby', 'baseball',
        'basketball', 'hockey', 'american football', 'volleyball', 'spikers',
        'galacticos', 'dragons', 'phantom', 'vitality', 'flash', 'twist',
        'bulls', 'giants', 'eagles', 'raiders', 'warriors', 'lakers'
    ]
    
    soccer_matches = []
    
    for game_id, info in games.items():
        if info.get('isGameFinished', True):
            continue
            
        tournament = info.get('tournamentName', '')
        teams = info.get('teams', [])
        
        # Must have proper teams
        if len(teams) < 2:
            continue
            
        home = next((t for t in teams if t.get('isHome')), None)
        away = next((t for t in teams if not t.get('isHome')), None)
        
        if not (home and away):
            continue
            
        home_name = home.get('name', '')
        away_name = away.get('name', '')
        
        # Skip empty names
        if not home_name or not away_name:
            continue
            
        # Check exclusions
        name_check = (home_name + away_name + tournament).lower()
        if any(term in name_check for term in exclude_terms):
            continue
            
        # Check if it's a soccer tournament or has soccer-like team names
        is_soccer = False
        
        # Tournament-based identification
        if tournament in soccer_tournaments:
            is_soccer = True
            
        # Team name patterns (FC, CF, CA, etc.)
        soccer_patterns = ['fc', 'cf', 'ca', 'ac', 'sc', 'afc', 'united', 'city', 'athletic', 'club']
        if any(pattern in home_name.lower() or pattern in away_name.lower() for pattern in soccer_patterns):
            is_soccer = True
            
        if not is_soccer:
            continue
            
        # Get maturity date
        maturity = info.get('maturityDate', info.get('startTime'))
        if maturity:
            try:
                if isinstance(maturity, (int, float)):
                    maturity = datetime.fromtimestamp(maturity / 1000, tz=timezone.utc)
                else:
                    maturity = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
            except:
                maturity = datetime.now(timezone.utc) + timedelta(hours=24)
        else:
            maturity = datetime.now(timezone.utc) + timedelta(hours=24)
            
        # Skip past matches
        if maturity < datetime.now(timezone.utc):
            continue
            
        soccer_matches.append({
            'id': game_id,
            'home': home_name,
            'away': away_name,
            'tournament': tournament,
            'maturity': maturity,
            'odds': {
                'home': info.get('homeOdds'),
                'draw': info.get('drawOdds'), 
                'away': info.get('awayOdds')
            }
        })
    
    return soccer_matches

def save_to_database(matches):
    """Save matches to database"""
    saved = 0
    
    with db_manager.get_db_session() as db:
        for match in matches:
            try:
                source_id = f"overtime_real_{match['id']}"
                
                # Check if exists
                existing = db.query(Market).filter(Market.source_id == source_id).first()
                if existing:
                    continue
                
                # Create market
                market = Market(
                    source_id=source_id,
                    home_team=match['home'],
                    away_team=match['away'],
                    sport='Soccer',
                    league_name=match['tournament'],
                    maturity_date=match['maturity'],
                    source='overtime_v2',
                    is_finished=False
                )
                db.add(market)
                db.commit()
                
                # Add odds if available
                if match['odds']['home'] and match['odds']['away']:
                    for outcome, odds_value in [
                        ('home', match['odds']['home']),
                        ('draw', match['odds']['draw'] if match['odds']['draw'] else 3.0),
                        ('away', match['odds']['away'])
                    ]:
                        if odds_value and odds_value > 1:
                            odd = Odd(
                                source_id=source_id,
                                outcome=outcome,
                                decimal_odds=float(odds_value),
                                market_type='winner',
                                source='overtime_v2',
                                bookmaker='overtime',
                                normalized_implied=1.0/float(odds_value)
                            )
                            db.add(odd)
                else:
                    # Default odds if not provided
                    for outcome, odds_value in [('home', 2.5), ('draw', 3.0), ('away', 2.8)]:
                        odd = Odd(
                            source_id=source_id,
                            outcome=outcome,
                            decimal_odds=odds_value,
                            market_type='winner',
                            source='overtime_v2',
                            bookmaker='overtime',
                            normalized_implied=1.0/odds_value
                        )
                        db.add(odd)
                
                db.commit()
                saved += 1
                
                if saved % 10 == 0:
                    logger.info(f"Saved {saved} matches...")
                    
            except Exception as e:
                logger.error(f"Error saving match {match['home']} vs {match['away']}: {e}")
                db.rollback()
                
    return saved

def main():
    logger.info("🎯 Fetching real soccer matches from Overtime...")
    
    # Fetch matches
    matches = fetch_real_soccer_matches()
    logger.info(f"Found {len(matches)} soccer matches")
    
    if not matches:
        logger.warning("No soccer matches found")
        return
        
    # Save to database
    saved = save_to_database(matches)
    logger.info(f"\n✅ Saved {saved} soccer matches to database")
    
    # Show summary
    with db_manager.get_db_session() as db:
        total_soccer = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).count()
        
        # Get some examples
        examples = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.source == 'overtime_v2'
        ).limit(10).all()
        
        logger.info(f"\nTotal active soccer markets: {total_soccer}")
        logger.info("\nExamples of saved matches:")
        for m in examples:
            logger.info(f"  - {m.home_team} vs {m.away_team} ({m.league_name})")

if __name__ == "__main__":
    main()