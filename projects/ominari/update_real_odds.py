#!/usr/bin/env python3
"""
Update database with real odds from Overtime Markets API
This will replace the placeholder odds with actual market data
"""

import os
import sys
import logging
from datetime import datetime, timezone
import requests

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_overtime_markets():
    """Fetch real markets with odds from Overtime API"""
    try:
        # Overtime V2 API endpoint
        url = "https://overtimemarketsv2.xyz/overtime-v2/networks/42161/markets/live"
        
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            markets = data.get('markets', [])
            logger.info(f"Fetched {len(markets)} live markets from Overtime")
            return markets
        else:
            logger.error(f"Failed to fetch markets: {response.status_code}")
            return []
            
    except Exception as e:
        logger.error(f"Error fetching markets: {e}")
        return []

def update_odds():
    """Update database with real odds"""
    markets_data = fetch_overtime_markets()
    
    if not markets_data:
        logger.warning("No markets fetched from API")
        return
        
    with db_manager.get_db_session() as db:
        updated_markets = 0
        created_odds = 0
        
        # First, let's see what the API data looks like
        if markets_data:
            sample = markets_data[0]
            logger.info(f"Sample market data: {list(sample.keys())}")
            
        for market_data in markets_data[:50]:  # Process first 50 markets
            try:
                # Extract market info
                home_team = market_data.get('homeTeam', '').strip()
                away_team = market_data.get('awayTeam', '').strip()
                
                if not home_team or not away_team:
                    continue
                    
                # Find existing market
                market = db.query(Market).filter(
                    Market.home_team == home_team,
                    Market.away_team == away_team,
                    Market.maturity_date > datetime.now(timezone.utc)
                ).first()
                
                if not market:
                    logger.debug(f"Market not found: {home_team} vs {away_team}")
                    continue
                    
                # Extract odds - the structure varies by API
                odds_updated = False
                
                # Try different possible structures
                if 'odds' in market_data:
                    odds_list = market_data['odds']
                    if isinstance(odds_list, list) and len(odds_list) >= 2:
                        # Update home odds
                        if odds_list[0] and odds_list[0] > 1:
                            update_odd(db, market.source_id, 'home', float(odds_list[0]))
                            odds_updated = True
                        # Update away odds
                        if odds_list[1] and odds_list[1] > 1:
                            update_odd(db, market.source_id, 'away', float(odds_list[1]))
                            odds_updated = True
                        # Update draw odds if available
                        if len(odds_list) > 2 and odds_list[2] and odds_list[2] > 1:
                            update_odd(db, market.source_id, 'draw', float(odds_list[2]))
                            odds_updated = True
                            
                elif 'homeOdds' in market_data and 'awayOdds' in market_data:
                    # Alternative structure
                    home_odds = market_data.get('homeOdds')
                    away_odds = market_data.get('awayOdds')
                    draw_odds = market_data.get('drawOdds')
                    
                    if home_odds and home_odds > 1:
                        update_odd(db, market.source_id, 'home', float(home_odds))
                        odds_updated = True
                    if away_odds and away_odds > 1:
                        update_odd(db, market.source_id, 'away', float(away_odds))
                        odds_updated = True
                    if draw_odds and draw_odds > 1:
                        update_odd(db, market.source_id, 'draw', float(draw_odds))
                        odds_updated = True
                        
                elif 'positions' in market_data:
                    # Another possible structure
                    positions = market_data['positions']
                    if isinstance(positions, list):
                        for i, pos in enumerate(positions):
                            if 'quote' in pos and pos['quote'] > 1:
                                outcome = ['home', 'away', 'draw'][i] if i < 3 else None
                                if outcome:
                                    update_odd(db, market.source_id, outcome, float(pos['quote']))
                                    odds_updated = True
                                    
                if odds_updated:
                    updated_markets += 1
                    logger.info(f"Updated odds for: {home_team} vs {away_team}")
                else:
                    logger.debug(f"No odds found in market data for: {home_team} vs {away_team}")
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
                
        db.commit()
        logger.info(f"Updated {updated_markets} markets with real odds")
        
        # Show sample of updated odds
        sample_odds = db.query(Odd).join(Market).filter(
            Market.maturity_date > datetime.now(timezone.utc)
        ).order_by(Odd.updated_at.desc()).limit(10).all()
        
        logger.info("\nSample updated odds:")
        for odd in sample_odds:
            market = db.query(Market).filter(Market.source_id == odd.source_id).first()
            if market:
                logger.info(f"{market.home_team} vs {market.away_team}: {odd.outcome} @ {odd.decimal_odds}")

def update_odd(db, source_id: str, outcome: str, decimal_odds: float):
    """Update or create an odd"""
    # Find existing odd
    existing = db.query(Odd).filter(
        Odd.source_id == source_id,
        Odd.outcome == outcome
    ).first()
    
    if existing:
        # Update if changed significantly
        if abs(existing.decimal_odds - decimal_odds) > 0.01:
            existing.decimal_odds = decimal_odds
            existing.updated_at = datetime.now(timezone.utc)
            logger.debug(f"Updated {outcome} odds to {decimal_odds}")
    else:
        # Create new
        new_odd = Odd(
            source_id=source_id,
            outcome=outcome,
            decimal_odds=decimal_odds,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        db.add(new_odd)
        logger.debug(f"Created new {outcome} odd: {decimal_odds}")

if __name__ == "__main__":
    logger.info("Starting real odds update...")
    update_odds()
    logger.info("Real odds update complete!")