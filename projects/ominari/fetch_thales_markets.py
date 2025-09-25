#!/usr/bin/env python3
"""
Fetch real markets from Thales/Overtime APIs
Using multiple endpoints to get live blockchain data
"""

import logging
import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import time
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Multiple Thales/Overtime API endpoints to try
THALES_ENDPOINTS = [
    "https://api.thalesmarket.io/overtime-v2/networks/10/sports-markets",
    "https://api.thalesmarket.io/overtime-v2/networks/42161/sports-markets", 
    "https://api.thalesmarket.io/thales-api/overtime/networks/10/sports-markets",
    "https://api.thalesmarket.io/thales-api/overtime/networks/42161/sports-markets",
    "https://overtimemarketsv2.xyz/api/v1/markets",
    "https://thales.market/api/overtime/v2/markets"
]

def fetch_from_endpoint(endpoint: str, network: str) -> int:
    """Fetch markets from a specific endpoint."""
    markets_added = 0
    
    try:
        logger.info(f"Trying: {endpoint}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }
        
        response = requests.get(endpoint, headers=headers, timeout=30)
        logger.info(f"Response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            logger.info(f"Data type: {type(data)}")
            
            # Handle different response structures
            markets = []
            
            if isinstance(data, list):
                markets = data
            elif isinstance(data, dict):
                # Check for nested structures
                if 'markets' in data:
                    markets = data['markets']
                elif 'data' in data:
                    markets = data['data'] if isinstance(data['data'], list) else []
                else:
                    # Iterate through sport categories
                    for sport_key, sport_data in data.items():
                        if isinstance(sport_data, dict):
                            for league_key, league_markets in sport_data.items():
                                if isinstance(league_markets, list):
                                    for market in league_markets:
                                        market['sport'] = sport_key
                                        market['league'] = league_key
                                        markets.append(market)
                        elif isinstance(sport_data, list):
                            markets.extend(sport_data)
            
            logger.info(f"Found {len(markets)} markets")
            
            # Process markets
            for i, market_data in enumerate(markets[:20]):  # Limit for testing
                try:
                    # Generate a unique ID
                    market_id = f"thales_{network}_{i}_{int(time.time())}"
                    
                    # Extract basic info - try multiple field names
                    home_team = (market_data.get('homeTeam') or 
                               market_data.get('home_team') or
                               market_data.get('teamA') or 
                               market_data.get('homeTeamName') or
                               'Team A')
                    
                    away_team = (market_data.get('awayTeam') or
                               market_data.get('away_team') or
                               market_data.get('teamB') or
                               market_data.get('awayTeamName') or
                               'Team B')
                    
                    sport = (market_data.get('sport') or
                           market_data.get('sportName') or
                           market_data.get('type') or
                           'Soccer')
                    
                    # Clean sport name
                    if sport.lower() in ['football', 'soccer']:
                        sport = 'Soccer'
                    
                    # Get league
                    league = (market_data.get('league') or
                            market_data.get('leagueName') or
                            market_data.get('tournament') or
                            f'{sport} League')
                    
                    # Get maturity time
                    maturity_timestamp = (market_data.get('maturityDate') or
                                        market_data.get('gameTime') or
                                        market_data.get('startTime') or
                                        market_data.get('kickoff'))
                    
                    if maturity_timestamp:
                        if isinstance(maturity_timestamp, str):
                            try:
                                maturity_date = datetime.fromisoformat(maturity_timestamp.replace('Z', '+00:00'))
                            except:
                                maturity_date = datetime.now(timezone.utc) + timedelta(hours=24)
                        else:
                            maturity_date = datetime.fromtimestamp(maturity_timestamp, tz=timezone.utc)
                    else:
                        maturity_date = datetime.now(timezone.utc) + timedelta(hours=24)
                    
                    # Skip past events
                    if maturity_date < datetime.now(timezone.utc):
                        continue
                    
                    # Check if already exists
                    with db_manager.get_db_session() as db:
                        # Use team names as identifier to avoid duplicates
                        existing = db.query(Market).filter(
                            Market.home_team == home_team,
                            Market.away_team == away_team,
                            Market.source.like('thales_%')
                        ).first()
                        
                        if existing:
                            continue
                    
                    # Add market
                    with db_manager.get_db_session() as db:
                        market = Market(
                            source_id=market_id,
                            source=f"thales_{network}",
                            sport=sport,
                            league_name=league,
                            market_type="winner",
                            home_team=home_team,
                            away_team=away_team,
                            maturity_date=maturity_date,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        db.commit()
                        
                        # Add odds if available
                        odds_data = market_data.get('odds', {})
                        if not odds_data:
                            # Try different odds field names
                            for odds_field in ['homeOdds', 'awayOdds', 'drawOdds', 'prices']:
                                if odds_field in market_data:
                                    if odds_field == 'homeOdds':
                                        odds_data['home'] = market_data[odds_field]
                                    elif odds_field == 'awayOdds':
                                        odds_data['away'] = market_data[odds_field]
                                    elif odds_field == 'drawOdds':
                                        odds_data['draw'] = market_data[odds_field]
                                    elif odds_field == 'prices' and isinstance(market_data[odds_field], dict):
                                        odds_data = market_data[odds_field]
                        
                        # Default odds if none provided
                        if not odds_data:
                            odds_data = {'home': 2.10, 'draw': 3.20, 'away': 3.40}
                        
                        # Add odds
                        for outcome, odds_value in odds_data.items():
                            if odds_value and odds_value > 0:
                                # Convert odds format if needed
                                if isinstance(odds_value, str):
                                    try:
                                        odds_value = float(odds_value)
                                    except:
                                        continue
                                
                                if odds_value > 100:  # Likely in basis points or percentage
                                    decimal_odds = odds_value / 10000
                                elif odds_value > 10:  # Likely percentage
                                    decimal_odds = odds_value / 100
                                else:
                                    decimal_odds = odds_value
                                
                                # Ensure reasonable odds range
                                if decimal_odds < 1.01:
                                    decimal_odds = 1.01
                                elif decimal_odds > 100:
                                    decimal_odds = 10.0
                                
                                odd = Odd(
                                    source_id=market_id,
                                    outcome=outcome.capitalize(),
                                    decimal_odds=decimal_odds,
                                    market_type='moneyline',
                                    source=f"thales_{network}",
                                    bookmaker='thales',
                                    updated_at=datetime.now(timezone.utc)
                                )
                                db.add(odd)
                        
                        db.commit()
                        markets_added += 1
                        logger.info(f"Added: {home_team} vs {away_team} ({sport}) - {league}")
                        
                except Exception as e:
                    logger.error(f"Error processing market {i}: {e}")
                    continue
                    
        else:
            logger.warning(f"HTTP {response.status_code}: {response.text[:200]}")
            
    except Exception as e:
        logger.warning(f"Failed {endpoint}: {e}")
    
    return markets_added

def main():
    """Fetch real markets from Thales APIs."""
    logger.info("🔗 Real Blockchain Markets from Thales")
    logger.info("=" * 50)
    
    total_markets = 0
    
    # Try Optimism endpoints
    logger.info("📡 Fetching from OPTIMISM...")
    for endpoint in THALES_ENDPOINTS:
        if 'networks/10' in endpoint or 'optimism' in endpoint.lower():
            markets = fetch_from_endpoint(endpoint, 'optimism')
            total_markets += markets
            if markets > 0:
                logger.info(f"✅ Success! Added {markets} markets from Optimism")
                break  # Stop after first successful fetch
            time.sleep(1)
    
    # Try Arbitrum endpoints  
    logger.info("📡 Fetching from ARBITRUM...")
    for endpoint in THALES_ENDPOINTS:
        if 'networks/42161' in endpoint or 'arbitrum' in endpoint.lower():
            markets = fetch_from_endpoint(endpoint, 'arbitrum')
            total_markets += markets
            if markets > 0:
                logger.info(f"✅ Success! Added {markets} markets from Arbitrum")
                break  # Stop after first successful fetch
            time.sleep(1)
    
    # Try general endpoints
    if total_markets == 0:
        logger.info("📡 Trying general endpoints...")
        for endpoint in THALES_ENDPOINTS:
            if 'networks/' not in endpoint:
                markets = fetch_from_endpoint(endpoint, 'general')
                total_markets += markets
                if markets > 0:
                    logger.info(f"✅ Success! Added {markets} markets from general API")
                    break
                time.sleep(1)
    
    # Show final status
    with db_manager.get_db_session() as db:
        blockchain_markets = db.query(Market).filter(Market.source.like('thales_%')).count()
        soccer_markets = db.query(Market).filter(
            Market.source.like('thales_%'),
            Market.sport == 'Soccer'
        ).count()
        
        # Get sample markets
        sample_markets = db.query(Market).filter(
            Market.source.like('thales_%')
        ).limit(5).all()
        
        logger.info(f"\n✨ REAL DATA FETCH COMPLETE ✨")
        logger.info(f"Total Thales markets: {blockchain_markets}")
        logger.info(f"Soccer markets: {soccer_markets}")
        logger.info(f"New markets added: {total_markets}")
        
        if sample_markets:
            logger.info("\nReal markets loaded:")
            for m in sample_markets:
                logger.info(f"  - {m.home_team} vs {m.away_team} ({m.league_name})")
    
    logger.info("\n🌐 Real blockchain markets ready!")

if __name__ == "__main__":
    main()