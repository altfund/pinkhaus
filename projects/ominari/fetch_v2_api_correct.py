#!/usr/bin/env python3
"""
Fetch Overtime V2 markets using the correct API structure
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
from web3 import Web3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Based on the documentation - the API base should be:
API_BASE = "https://overtimemarketsv2.xyz"

def fetch_v2_markets():
    """Fetch V2 markets from the correct API."""
    logger.info("🎯 Fetching Overtime V2 Markets")
    logger.info("=" * 60)
    
    # Try different endpoint structures
    endpoints = [
        # Network-specific endpoints
        f"{API_BASE}/overtime-v2/networks/10/games",  # Optimism
        f"{API_BASE}/overtime-v2/networks/42161/games",  # Arbitrum
        
        # Alternative structures  
        f"{API_BASE}/games?network=10",
        f"{API_BASE}/games?network=42161",
        f"{API_BASE}/api/games",
        f"{API_BASE}/v2/games",
        
        # Try api.overtime.io structure
        "https://api.overtime.xyz/overtime-v2/networks/10/games",
        "https://api.overtime.xyz/overtime-v2/networks/42161/games",
    ]
    
    markets_found = False
    
    for endpoint in endpoints:
        try:
            logger.info(f"\nTrying: {endpoint}")
            response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Handle different response structures
                games = []
                if isinstance(data, list):
                    games = data
                elif isinstance(data, dict):
                    games = data.get('games', data.get('data', []))
                    
                if games:
                    logger.info(f"✅ Found {len(games)} games!")
                    process_v2_games(games, endpoint)
                    markets_found = True
                    break
                    
        except Exception as e:
            logger.debug(f"Failed: {e}")
            
    if not markets_found:
        # Try a simple test to see what's available
        logger.info("\n🔍 Trying base URL to discover structure...")
        try:
            response = requests.get(API_BASE, timeout=5)
            logger.info(f"Base URL status: {response.status_code}")
            if response.status_code == 200:
                logger.info(f"Base content preview: {response.text[:200]}...")
        except:
            pass
            
    # Also check if there's a GraphQL endpoint
    logger.info("\n🔍 Checking for GraphQL endpoint...")
    graphql_endpoints = [
        "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-v2-optimism",
        "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-v2-arbitrum",
    ]
    
    for endpoint in graphql_endpoints:
        try:
            # Simple GraphQL query to test
            query = {
                "query": """{
                    games(first: 5, where: {isResolved: false}) {
                        id
                        gameId
                        homeTeam
                        awayTeam
                        startTime
                    }
                }"""
            }
            
            response = requests.post(endpoint, json=query, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'games' in data['data']:
                    games = data['data']['games']
                    logger.info(f"✅ Found {len(games)} games via GraphQL!")
                    # Process GraphQL games...
                    
        except Exception as e:
            logger.debug(f"GraphQL failed: {e}")
            
def process_v2_games(games, source_url):
    """Process V2 game data."""
    markets_added = 0
    
    # Determine chain from URL
    chain = 'optimism' if '10' in source_url else 'arbitrum' if '42161' in source_url else 'unknown'
    
    for game in games[:50]:  # Process up to 50
        try:
            # Extract game details
            game_id = game.get('gameId') or game.get('id')
            home_team = game.get('homeTeam')
            away_team = game.get('awayTeam')
            start_time = game.get('startTime') or game.get('timestamp')
            sport_id = game.get('sport') or game.get('sportId')
            
            if not all([game_id, home_team, away_team, start_time]):
                continue
                
            # Convert timestamp
            if isinstance(start_time, str):
                maturity = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            else:
                maturity = datetime.fromtimestamp(start_time, tz=timezone.utc)
                
            # Skip past games
            if maturity < datetime.now(timezone.utc):
                continue
                
            market_id = f"{chain}_v2_{str(game_id)[-8:]}"
            
            with db_manager.get_db_session() as db:
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # Sport mapping
                sport_map = {
                    1: "American Football",
                    2: "Baseball",
                    3: "Basketball",
                    4: "Soccer",
                    5: "Hockey",
                    6: "MMA", 
                    7: "Boxing",
                    8: "Tennis"
                }
                
                market = Market(
                    source_id=market_id,
                    source=f"{chain}_v2_api",
                    sport=sport_map.get(sport_id, "Other"),
                    league_name=game.get('league', 'Unknown'),
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Process odds/markets
                for mkt in game.get('markets', []):
                    mkt_type = mkt.get('type')
                    odds = mkt.get('odds', [])
                    
                    for i, odd_val in enumerate(odds):
                        if odd_val and odd_val > 0:
                            # Convert from contract format
                            decimal_odds = odd_val / 1e18 if odd_val > 1000 else odd_val
                            
                            if decimal_odds > 1.0 and decimal_odds < 100:
                                outcome = ['home', 'away', 'draw'][i] if i < 3 else f"position_{i}"
                                
                                # American odds
                                if decimal_odds >= 2.0:
                                    american = int((decimal_odds - 1) * 100)
                                else:
                                    american = int(-100 / (decimal_odds - 1))
                                    
                                odd = Odd(
                                    source_id=market.source_id,
                                    market_type=mkt_type or "winner",
                                    outcome=outcome,
                                    source=f"{chain}_v2_api",
                                    bookmaker="Overtime V2",
                                    decimal_odds=decimal_odds,
                                    american_odds=american,
                                    normalized_implied=1.0 / decimal_odds,
                                    updated_at=datetime.now(timezone.utc)
                                )
                                db.add(odd)
                                
                db.commit()
                markets_added += 1
                
                logger.info(f"✅ Added: {home_team} vs {away_team}")
                logger.info(f"   Sport: {sport_map.get(sport_id, 'Other')}")
                logger.info(f"   Date: {maturity}")
                
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            continue
            
    if markets_added > 0:
        # Clear sample data
        with db_manager.get_db_session() as db:
            sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
            for market in sample_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete() 
                db.delete(market)
            db.commit()
            
        logger.info(f"\n🎆 SUCCESS! Added {markets_added} real V2 markets!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
        
    return markets_added

def main():
    """Main function."""
    fetch_v2_markets()
    
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n📊 Database Summary:")
        logger.info(f"Total markets: {total}")
        logger.info(f"Active markets: {active}")
        
        # Show some examples
        if active > 0:
            examples = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).limit(5).all()
            
            logger.info("\n🏆 Example active markets:")
            for m in examples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date} - {m.source}")

if __name__ == "__main__":
    main()