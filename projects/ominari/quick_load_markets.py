#!/usr/bin/env python3
"""
Quick script to load a few real markets without timeout
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Load just a few markets quickly."""
    logger.info("🚀 Quick Market Loader")
    
    # Clear existing data
    with db_manager.get_db_session() as db:
        old_count = db.query(Market).count()
        if old_count > 0:
            db.query(Odd).delete()
            db.query(Market).delete()
            db.commit()
            logger.info(f"Cleared {old_count} old markets")
    
    # Try to get just 10 games from API
    try:
        response = requests.get("https://api.overtime.io/overtime-v2/games-info", timeout=10)
        if response.status_code == 200:
            games = response.json()
            
            added = 0
            for game_id, info in list(games.items())[:20]:  # Check first 20
                teams = info.get('teams', [])
                if len(teams) == 2:
                    home = teams[0].get('name', '')
                    away = teams[1].get('name', '') if len(teams) > 1 else ''
                    
                    # Skip futures
                    if 'Winner' in away or 'Championship' in away:
                        continue
                        
                    market_id = f"quick_{game_id[-6:]}"
                    
                    with db_manager.get_db_session() as db:
                        market = Market(
                            source_id=market_id,
                            source="overtime_quick",
                            sport="Soccer",  # Default
                            league_name="Overtime",
                            market_type="winner",
                            home_team=home,
                            away_team=away,
                            maturity_date=datetime.now(timezone.utc) + timedelta(days=2),
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        
                        # Simple odds
                        for outcome, odds in [('home', 1.9), ('away', 2.1)]:
                            odd = Odd(
                                source_id=market_id,
                                market_type="winner",
                                outcome=outcome,
                                source="overtime_quick",
                                bookmaker="Overtime",
                                decimal_odds=odds,
                                american_odds=int((odds - 1) * 100),
                                normalized_implied=1.0 / odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
                        db.commit()
                        added += 1
                        
                        if added >= 5:  # Just load 5 markets
                            break
                            
            logger.info(f"✅ Added {added} markets")
            
    except Exception as e:
        logger.error(f"Error: {e}")
        
    # Show what we have
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        logger.info(f"\n📊 Total markets: {total}")
        
        if total > 0:
            markets = db.query(Market).limit(3).all()
            logger.info("\nExample markets:")
            for m in markets:
                logger.info(f"  • {m.home_team} vs {m.away_team}")

if __name__ == "__main__":
    main()