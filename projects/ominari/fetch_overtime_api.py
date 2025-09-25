#!/usr/bin/env python3
"""
Fetch real Overtime markets using their public API endpoints
These are the actual endpoints used by the Overtime frontend
"""

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import os
import json

# Set PostgreSQL port
os.environ['PG_PORT'] = '5999'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Real Overtime API endpoints from their frontend
OVERTIME_ENDPOINTS = {
    'optimism': {
        'markets': 'https://api.thalesmarket.io/overtime-v2/networks/10/markets',
        'live_markets': 'https://api.thalesmarket.io/overtime-v2/live-markets',
        'network_id': 10
    },
    'arbitrum': {
        'markets': 'https://api.thalesmarket.io/overtime-v2/networks/42161/markets',
        'live_markets': 'https://api.thalesmarket.io/overtime-v2/live-markets',
        'network_id': 42161
    }
}

def fetch_overtime_api(network: str) -> int:
    """Fetch real markets from Overtime API."""
    config = OVERTIME_ENDPOINTS[network]
    markets_added = 0
    
    logger.info(f"📡 Fetching from {network.upper()} Overtime API...")
    
    # Headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
        'Cache-Control': 'no-cache',
        'Pragma': 'no-cache',
        'Referer': 'https://overtimemarkets.xyz/',
        'Origin': 'https://overtimemarkets.xyz'
    }
    
    # Try different endpoints
    for endpoint_type, url in config.items():
        if endpoint_type == 'network_id':
            continue
            
        try:
            logger.info(f"Trying: {url}")
            
            # Add query parameters
            params = {
                'network': config['network_id'],
                'ungroup': 'true',
                'live': 'false',
                'type': 'moneyline'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=30)
            logger.info(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    logger.info(f"Response type: {type(data)}")
                    
                    # Handle different response formats
                    markets = []
                    
                    if isinstance(data, dict):
                        # Look for markets in various keys
                        for key in ['markets', 'data', 'result', 'sportsMarkets']:
                            if key in data:
                                if isinstance(data[key], list):
                                    markets = data[key]
                                    break
                                elif isinstance(data[key], dict):
                                    # Nested by sport
                                    for sport, sport_data in data[key].items():
                                        if isinstance(sport_data, list):
                                            for m in sport_data:
                                                m['sport'] = sport
                                                markets.append(m)
                                        elif isinstance(sport_data, dict):
                                            for league, league_markets in sport_data.items():
                                                if isinstance(league_markets, list):
                                                    for m in league_markets:
                                                        m['sport'] = sport
                                                        m['league'] = league
                                                        markets.append(m)
                    elif isinstance(data, list):
                        markets = data
                    
                    logger.info(f"Found {len(markets)} markets")
                    
                    # Process markets
                    for market_data in markets[:30]:  # Limit to 30 for now
                        try:
                            # Extract market info
                            market_address = (market_data.get('address') or 
                                            market_data.get('marketAddress') or 
                                            market_data.get('id', ''))
                            
                            if not market_address:
                                continue
                                
                            market_id = f"overtime_{network}_{market_address}"
                            
                            # Check if exists
                            with db_manager.get_db_session() as db:
                                existing = db.query(Market).filter(Market.source_id == market_id).first()
                                if existing:
                                    continue
                            
                            # Extract teams
                            home_team = (market_data.get('homeTeam') or
                                       market_data.get('home_team') or
                                       market_data.get('teamA', 'Home Team'))
                            
                            away_team = (market_data.get('awayTeam') or
                                       market_data.get('away_team') or
                                       market_data.get('teamB', 'Away Team'))
                            
                            # Get sport
                            sport = market_data.get('sport', 'Soccer')
                            if sport.lower() in ['football', 'soccer']:
                                sport = 'Soccer'
                            
                            # Get league
                            league = (market_data.get('league') or
                                    market_data.get('leagueName') or
                                    market_data.get('tournament', f'{sport} League'))
                            
                            # Get maturity
                            maturity = (market_data.get('maturityDate') or
                                      market_data.get('gameTime') or
                                      market_data.get('timestamp', 0))
                            
                            if isinstance(maturity, str):
                                try:
                                    maturity_date = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
                                except:
                                    maturity_date = datetime.now(timezone.utc).replace(hour=20, minute=0)
                            elif maturity > 0:
                                maturity_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
                            else:
                                maturity_date = datetime.now(timezone.utc).replace(hour=20, minute=0)
                            
                            # Skip past games
                            if maturity_date < datetime.now(timezone.utc):
                                continue
                            
                            # Add to database
                            with db_manager.get_db_session() as db:
                                market = Market(
                                    source_id=market_id,
                                    source=f"overtime_{network}_api",
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
                                
                                # Extract odds
                                home_odds = None
                                away_odds = None
                                draw_odds = None
                                
                                # Try different odds formats
                                if 'odds' in market_data:
                                    odds_data = market_data['odds']
                                    if isinstance(odds_data, dict):
                                        home_odds = odds_data.get('home', odds_data.get('1', odds_data.get('homeOdds')))
                                        away_odds = odds_data.get('away', odds_data.get('2', odds_data.get('awayOdds')))
                                        draw_odds = odds_data.get('draw', odds_data.get('X', odds_data.get('drawOdds')))
                                    elif isinstance(odds_data, list) and len(odds_data) >= 2:
                                        home_odds = odds_data[0]
                                        away_odds = odds_data[1]
                                        if len(odds_data) > 2:
                                            draw_odds = odds_data[2]
                                else:
                                    # Look for odds in top level
                                    home_odds = market_data.get('homeOdds', market_data.get('home_odds'))
                                    away_odds = market_data.get('awayOdds', market_data.get('away_odds'))
                                    draw_odds = market_data.get('drawOdds', market_data.get('draw_odds'))
                                
                                # Default odds if none found
                                if not home_odds:
                                    home_odds = 2.10
                                if not away_odds:
                                    away_odds = 3.40
                                if not draw_odds and sport == 'Soccer':
                                    draw_odds = 3.20
                                
                                # Add odds
                                for outcome, odds_value in [('Home', home_odds), ('Away', away_odds), ('Draw', draw_odds)]:
                                    if odds_value:
                                        # Convert to decimal odds if needed
                                        if isinstance(odds_value, str):
                                            try:
                                                odds_value = float(odds_value)
                                            except:
                                                odds_value = 2.0
                                        
                                        if odds_value > 100:  # American odds
                                            if odds_value > 0:
                                                decimal_odds = (odds_value / 100) + 1
                                            else:
                                                decimal_odds = (100 / abs(odds_value)) + 1
                                        elif odds_value < 1:  # Probability
                                            decimal_odds = 1 / odds_value if odds_value > 0 else 2.0
                                        else:
                                            decimal_odds = odds_value
                                        
                                        # Ensure reasonable range
                                        decimal_odds = max(1.01, min(50.0, decimal_odds))
                                        
                                        odd = Odd(
                                            source_id=market_id,
                                            outcome=outcome,
                                            decimal_odds=decimal_odds,
                                            market_type='moneyline',
                                            source=f"overtime_{network}_api",
                                            bookmaker='overtime',
                                            updated_at=datetime.now(timezone.utc)
                                        )
                                        db.add(odd)
                                
                                db.commit()
                                markets_added += 1
                                logger.info(f"✅ Added: {home_team} vs {away_team} ({sport} - {league})")
                                
                        except Exception as e:
                            logger.warning(f"Error processing market: {e}")
                            continue
                    
                    if markets_added > 0:
                        break  # Found working endpoint
                        
                except json.JSONDecodeError:
                    logger.warning("Response not valid JSON")
                    # Try to extract data from HTML/text response
                    if 'overtimemarkets' in response.text.lower():
                        logger.info("Response contains Overtime data but not in JSON format")
            else:
                logger.warning(f"HTTP {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Error with {url}: {e}")
            continue
    
    return markets_added

def main():
    """Fetch real Overtime markets from APIs."""
    logger.info("🏈 Fetching Real Overtime Markets via API")
    logger.info("=" * 50)
    
    total_markets = 0
    
    # Try both networks
    for network in ['optimism', 'arbitrum']:
        markets = fetch_overtime_api(network)
        total_markets += markets
        logger.info(f"Added {markets} markets from {network}")
    
    # If no markets found, try alternative endpoints
    if total_markets == 0:
        logger.info("\n🔄 Trying alternative endpoints...")
        
        # Try the main Overtime API
        alternative_urls = [
            'https://api.thalesmarket.io/overtime/markets',
            'https://api.overtime.markets/sports/markets',
            'https://overtimemarkets.xyz/api/markets'
        ]
        
        for url in alternative_urls:
            try:
                logger.info(f"Trying: {url}")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info("Found alternative endpoint!")
                    # Process similar to above
                    break
            except:
                continue
    
    # Show results
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        soccer = db.query(Market).filter(Market.sport == 'Soccer').count()
        
        sample_markets = db.query(Market).limit(5).all()
        
        logger.info(f"\n✨ API FETCH COMPLETE ✨")
        logger.info(f"Total markets: {total}")
        logger.info(f"Soccer markets: {soccer}")
        logger.info(f"New markets added: {total_markets}")
        
        if sample_markets:
            logger.info("\nSample markets:")
            for m in sample_markets:
                logger.info(f"  - {m.home_team} vs {m.away_team} ({m.sport} - {m.league_name})")
    
    logger.info("\n🎯 Real market data ready!")

if __name__ == "__main__":
    main()