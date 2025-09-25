#!/usr/bin/env python3
"""
Quick sync to add upcoming soccer games
"""
import os
os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timedelta
import random
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_upcoming_soccer_games():
    """Create 20 upcoming soccer games with realistic data."""
    
    soccer_games = [
        # Premier League
        ("Manchester United", "Liverpool FC", "Premier League"),
        ("Chelsea FC", "Arsenal FC", "Premier League"),
        ("Manchester City", "Tottenham Hotspur", "Premier League"),
        
        # La Liga
        ("Real Madrid", "FC Barcelona", "La Liga"),
        ("Atletico Madrid", "Valencia CF", "La Liga"),
        ("Sevilla FC", "Real Betis", "La Liga"),
        
        # Serie A
        ("AC Milan", "Inter Milan", "Serie A"),
        ("Juventus", "AS Roma", "Serie A"),
        ("Napoli", "Lazio", "Serie A"),
        
        # Bundesliga
        ("Bayern Munich", "Borussia Dortmund", "Bundesliga"),
        ("RB Leipzig", "Bayer Leverkusen", "Bundesliga"),
        
        # Other leagues
        ("Ajax", "PSV Eindhoven", "Eredivisie"),
        ("Benfica", "FC Porto", "Primeira Liga"),
        ("Olympique Marseille", "Paris Saint-Germain", "Ligue 1"),
        ("Celtic FC", "Rangers FC", "Scottish Premiership"),
        ("Club Brugge", "Standard Liege", "Belgian Pro League"),
        ("Galatasaray", "Fenerbahce", "Super Lig"),
        ("CSKA Moscow", "Zenit St Petersburg", "Russian Premier League"),
        ("Shakhtar Donetsk", "Dynamo Kyiv", "Ukrainian Premier League"),
        ("Olympiacos", "Panathinaikos", "Super League Greece"),
    ]
    
    with db_manager.get_db_session() as db:
        # Clear old test data (delete odds first due to foreign key)
        db.query(Odd).filter(Odd.source == 'quick_sync_soccer').delete()
        db.query(Market).filter(Market.source == 'quick_sync_soccer').delete()
        db.commit()
        
        current_time = datetime.now()  # timezone-naive to match DB
        
        for i, (home, away, league) in enumerate(soccer_games):
            # Spread games across next 48 hours
            hours_ahead = 2 + (i * 2.5)
            maturity = current_time + timedelta(hours=hours_ahead)
            
            # Create unique ID
            source_id = f"soccer_{i}_{int(current_time.timestamp())}"
            
            market = Market(
                source_id=source_id,
                source='quick_sync_soccer',
                sport='Soccer',
                league_name=league,
                market_type='winner',
                home_team=home,
                away_team=away,
                maturity_date=maturity,
                is_finished=False
            )
            db.add(market)
            
            # Add realistic odds
            odds_patterns = [
                {'home': 1.85, 'draw': 3.50, 'away': 4.20},  # Home favorite
                {'home': 2.30, 'draw': 3.20, 'away': 3.10},  # Slight home favorite
                {'home': 2.75, 'draw': 3.15, 'away': 2.65},  # Balanced
                {'home': 4.50, 'draw': 3.60, 'away': 1.75},  # Away favorite
            ]
            
            odds = odds_patterns[i % len(odds_patterns)]
            
            for outcome, decimal_odds in odds.items():
                odd = Odd(
                    source_id=source_id,
                    market_type='winner',
                    outcome=outcome,
                    source='quick_sync_soccer',
                    bookmaker='Test Bookmaker',
                    decimal_odds=decimal_odds,
                    normalized_implied=1.0 / decimal_odds
                )
                db.add(odd)
        
        db.commit()
        logger.info(f"✅ Created {len(soccer_games)} upcoming soccer games")
        
        # Show next 5 games
        next_games = db.query(Market).filter(
            Market.source == 'quick_sync_soccer',
            Market.is_finished == False
        ).order_by(Market.maturity_date).limit(5).all()
        
        logger.info("\n📅 Next 5 games:")
        for game in next_games:
            # Both should be timezone-naive now
            time_until = game.maturity_date - current_time
            hours = time_until.total_seconds() / 3600
            logger.info(f"  {game.home_team} vs {game.away_team} - in {hours:.1f}h ({game.league_name})")

if __name__ == "__main__":
    create_upcoming_soccer_games()