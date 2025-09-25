#!/usr/bin/env python3
"""
Discover Overtime V2 API structure and fetch available markets
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def discover_api_endpoints():
    """Try different API endpoint variations."""
    base_urls = [
        "https://api.overtime.io",
        "https://api.thalesmarket.io",
        "https://overtime-v2-api.xyz",
    ]
    
    endpoints = [
        "/overtime-v2/networks/42161/games",
        "/overtime-v2/networks/42161/markets", 
        "/v2/games",
        "/v2/markets",
        "/api/v2/games",
        "/api/v2/markets",
        "/sports-markets",
        "/sports/markets",
        "/markets",
        "/games",
    ]
    
    logger.info("🔍 Discovering Overtime V2 API endpoints...")
    
    working_endpoints = []
    
    for base in base_urls:
        for endpoint in endpoints:
            url = base + endpoint
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if data:  # Has content
                            logger.info(f"✅ Found working endpoint: {url}")
                            logger.info(f"   Response keys: {list(data.keys())[:5]}")
                            working_endpoints.append((url, data))
                    except:
                        pass
            except:
                pass
                
    return working_endpoints

def check_thales_api():
    """Check Thales/Overtime API endpoints."""
    logger.info("\n🎯 Checking Thales/Overtime specific endpoints...")
    
    # Based on Thales documentation
    urls = [
        "https://api.thalesmarket.io/overtime-v2/optimism-mainnet/games",
        "https://api.thalesmarket.io/overtime-v2/arbitrum-one/games",
        "https://api.thalesmarket.io/overtime/optimism/markets",
        "https://api.thalesmarket.io/overtime/arbitrum/markets",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            logger.info(f"\nChecking: {url}")
            logger.info(f"Status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    logger.info(f"✅ Found {len(data)} items")
                    # Show sample
                    sample = data[0]
                    if isinstance(sample, dict):
                        logger.info(f"Sample keys: {list(sample.keys())[:10]}")
                        
                        # Process if it looks like market data
                        if any(key in sample for key in ['homeTeam', 'awayTeam', 'gameId']):
                            process_api_markets(url, data)
                            
                elif isinstance(data, dict):
                    logger.info(f"Response type: dict with keys {list(data.keys())}")
                    # Check for nested data
                    for key in ['games', 'markets', 'data']:
                        if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                            logger.info(f"Found {len(data[key])} items in '{key}'")
                            process_api_markets(url, data[key])
                            break
                            
        except Exception as e:
            logger.error(f"Error: {e}")
            
def process_api_markets(source_url, markets_data):
    """Process market data from API."""
    logger.info(f"\n💾 Processing {len(markets_data)} potential markets...")
    
    markets_added = 0
    
    for item in markets_data[:20]:  # Process first 20
        try:
            # V2 format
            if 'gameId' in item:
                game_id = item.get('gameId')
                home_team = item.get('homeTeam')
                away_team = item.get('awayTeam')
                start_time = item.get('startTime') or item.get('maturityDate')
                sport_id = item.get('sportId') or item.get('sport')
                
            # V1 format
            elif 'id' in item:
                game_id = item.get('id')
                home_team = item.get('homeTeam')
                away_team = item.get('awayTeam')
                start_time = item.get('maturityDate')
                sport_id = item.get('tags', [None])[0] if 'tags' in item else None
                
            else:
                continue
                
            if not all([home_team, away_team, start_time]):
                continue
                
            # Parse timestamp
            if isinstance(start_time, str):
                # Try ISO format
                try:
                    maturity = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except:
                    continue
            else:
                # Unix timestamp
                maturity = datetime.fromtimestamp(start_time / 1000 if start_time > 1e12 else start_time, tz=timezone.utc)
                
            # Skip past games
            if maturity < datetime.now(timezone.utc):
                continue
                
            # Determine chain from URL
            chain = 'optimism' if 'optimism' in source_url else 'arbitrum'
            
            market_id = f"{chain}_v2_api_{str(game_id)[-8:]}"
            
            with db_manager.get_db_session() as db:
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # Map sport
                sport_map = {
                    1: "American Football", 9001: "American Football",
                    2: "Baseball", 9002: "Baseball",
                    3: "Basketball", 9003: "Basketball", 
                    4: "Soccer", 9004: "Soccer",
                    5: "Hockey", 9005: "Hockey",
                    6: "MMA", 9006: "MMA",
                    7: "Boxing", 9007: "Boxing",
                    8: "Tennis", 9008: "Tennis"
                }
                
                market = Market(
                    source_id=market_id,
                    source=f"{chain}_api",
                    sport=sport_map.get(sport_id, "Other"),
                    league_name=item.get('leagueName', 'Unknown'),
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add odds if available
                if 'markets' in item:
                    for mkt in item['markets']:
                        odds_data = mkt.get('odds', [])
                        for i, odd_val in enumerate(odds_data):
                            if odd_val:
                                decimal_odds = odd_val / 1e18 if odd_val > 100 else odd_val
                                if decimal_odds > 1.0:
                                    outcome = ['home', 'away', 'draw'][i] if i < 3 else f"pos_{i}"
                                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                                    
                                    odd = Odd(
                                        source_id=market.source_id,
                                        market_type="winner",
                                        outcome=outcome,
                                        source=f"{chain}_api",
                                        bookmaker="Overtime",
                                        decimal_odds=decimal_odds,
                                        american_odds=american,
                                        normalized_implied=1.0 / decimal_odds,
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(odd)
                                    
                db.commit()
                markets_added += 1
                
                logger.info(f"✅ Added: {home_team} vs {away_team} on {maturity}")
                
        except Exception as e:
            logger.error(f"Error processing market: {e}")
            continue
            
    if markets_added > 0:
        # Clear sample data
        with db_manager.get_db_session() as db:
            sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
            for market in sample_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            
        logger.info(f"\n🎆 SUCCESS! Added {markets_added} real markets from API!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
        
def main():
    """Main discovery function."""
    logger.info("🎯 Overtime V2 API Discovery Tool")
    logger.info("=" * 60)
    
    # Try to discover endpoints
    working = discover_api_endpoints()
    
    if working:
        logger.info(f"\nFound {len(working)} working endpoints")
        
    # Check Thales-specific endpoints
    check_thales_api()
    
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
        
if __name__ == "__main__":
    main()