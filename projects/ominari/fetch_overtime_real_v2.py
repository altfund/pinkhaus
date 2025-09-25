#!/usr/bin/env python3
"""
Fetch real Overtime V2 data using public API endpoints
Then optionally enhanced with protected endpoints if API key is available
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
from typing import Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 API Configuration
API_BASE = "https://api.overtime.io/overtime-v2"

# Optional API key from environment
API_KEY = os.environ.get('OVERTIME_API_KEY')

# Chain IDs
CHAINS = {
    'optimism': {'id': 10, 'name': 'Optimism'},
    'arbitrum': {'id': 42161, 'name': 'Arbitrum'}
}

def fetch_public_sports() -> Dict:
    """Fetch sports data from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/sports", timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Error fetching sports: {e}")
        return {}

def fetch_public_games_info() -> List[Dict]:
    """Fetch games info from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/games-info", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Handle different response formats
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('games', data.get('data', []))
        return []
    except Exception as e:
        logger.error(f"Error fetching games info: {e}")
        return []

def fetch_public_live_scores() -> List[Dict]:
    """Fetch live scores from public endpoint."""
    try:
        response = requests.get(f"{API_BASE}/live-scores", timeout=30)
        response.raise_for_status()
        data = response.json()
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('scores', data.get('games', []))
        return []
    except Exception as e:
        logger.error(f"Error fetching live scores: {e}")
        return []

def fetch_protected_markets(chain_id: int) -> List[Dict]:
    """Fetch markets from protected endpoint (requires API key)."""
    if not API_KEY:
        logger.info("No API key available, skipping protected markets endpoint")
        return []
        
    try:
        headers = {"x-api-key": API_KEY}
        url = f"{API_BASE}/networks/{chain_id}/markets?ungroup=true"
        
        response = requests.get(url, headers=headers, timeout=45)
        response.raise_for_status()
        
        data = response.json()
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return data.get('markets', data.get('data', []))
        return []
        
    except Exception as e:
        logger.error(f"Error fetching protected markets: {e}")
        return []

def process_games_into_markets(games: List[Dict], source: str) -> int:
    """Process games data and create markets."""
    markets_added = 0
    
    for game in games:
        try:
            # Extract game details
            game_id = game.get('gameId', game.get('id'))
            home_team = game.get('homeTeam', game.get('home'))
            away_team = game.get('awayTeam', game.get('away'))
            start_time = game.get('startTime', game.get('gameTime', game.get('maturityDate')))
            sport_id = game.get('sportId', game.get('sport'))
            league = game.get('league', game.get('leagueName', 'Unknown'))
            is_resolved = game.get('isResolved', game.get('resolved', False))
            
            if not all([game_id, home_team, away_team, start_time]):
                continue
                
            # Skip resolved games
            if is_resolved:
                continue
                
            # Convert timestamp
            if isinstance(start_time, str):
                try:
                    maturity = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                except:
                    continue
            else:
                maturity = datetime.fromtimestamp(start_time, tz=timezone.utc)
                
            # Skip past games
            if maturity < datetime.now(timezone.utc):
                continue
                
            # Determine chain from game_id or default
            chain = 'optimism'  # Default, could be enhanced
            
            market_id = f"{source}_{str(game_id)[-8:]}"
            
            with db_manager.get_db_session() as db:
                # Skip if exists
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # Sport mapping
                sport_map = {
                    0: "American Football",
                    1: "Basketball",
                    2: "Baseball", 
                    3: "Hockey",
                    4: "Soccer",
                    5: "Boxing/MMA",
                    6: "Tennis",
                    7: "Motorsports",
                    8: "Golf",
                    9: "Cricket"
                }
                
                market = Market(
                    source_id=market_id,
                    source=source,
                    sport=sport_map.get(sport_id, "Soccer"),
                    league_name=league,
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add basic odds if available
                if 'odds' in game:
                    for position, odd_value in enumerate(game['odds']):
                        if odd_value and odd_value > 0:
                            decimal_odds = odd_value / 1e18 if odd_value > 1000 else odd_value
                            if 1.0 < decimal_odds < 100:
                                outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                                outcome = outcome_map.get(position, f'position_{position}')
                                
                                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                                
                                odd = Odd(
                                    source_id=market_id,
                                    market_type="winner",
                                    outcome=outcome,
                                    source=source,
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
                logger.info(f"   Sport: {sport_map.get(sport_id, 'Unknown')}")
                logger.info(f"   Date: {maturity}")
                
        except Exception as e:
            logger.error(f"Error processing game: {e}")
            continue
            
    return markets_added

def resolve_contract_addresses():
    """Resolve V2 contract addresses from redirector."""
    logger.info("\n🔍 Resolving V2 contract addresses...")
    
    contracts = {
        'optimism': {
            'SportsAMMV2': 'https://v2.contracts.overtime.io/mainnet-ovm/SportsAMMV2',
            'Manager': 'https://v2.contracts.overtime.io/mainnet-ovm/SportsAMMV2Manager',
        },
        'arbitrum': {
            'SportsAMMV2': 'https://v2.contracts.overtime.io/arbitrum-mainnet/SportsAMMV2',
            'Manager': 'https://v2.contracts.overtime.io/arbitrum-mainnet/SportsAMMV2Manager',
        }
    }
    
    resolved = {}
    
    for chain, urls in contracts.items():
        resolved[chain] = {}
        for name, url in urls.items():
            try:
                # Follow redirect to get actual address
                response = requests.get(url, allow_redirects=False, timeout=5)
                if response.status_code == 302:
                    location = response.headers.get('Location', '')
                    # Extract address from etherscan URL
                    if '/address/' in location:
                        address = location.split('/address/')[-1].split('#')[0]
                        resolved[chain][name] = address
                        logger.info(f"  {chain} {name}: {address}")
            except Exception as e:
                logger.error(f"Error resolving {chain} {name}: {e}")
                
    return resolved

def main():
    """Main function to fetch real Overtime V2 data."""
    logger.info("🎯 Fetching Real Overtime V2 Data")
    logger.info("=" * 60)
    
    if API_KEY:
        logger.info("🔑 API key found - will use protected endpoints")
    else:
        logger.info("🔓 No API key - using public endpoints only")
        logger.info("Set OVERTIME_API_KEY environment variable for full access")
        
    # Resolve contract addresses
    contracts = resolve_contract_addresses()
    
    total_added = 0
    
    # 1. Fetch public sports data
    logger.info("\n🏅 Fetching sports...")
    sports = fetch_public_sports()
    if sports:
        logger.info(f"Found {len(sports)} sports")
        
    # 2. Fetch public games info
    logger.info("\n🎮 Fetching games info...")
    games = fetch_public_games_info()
    if games:
        logger.info(f"Found {len(games)} games")
        added = process_games_into_markets(games, "v2_public_api")
        total_added += added
        
    # 3. Fetch live scores
    logger.info("\n📺 Fetching live scores...")
    live_scores = fetch_public_live_scores()
    if live_scores:
        logger.info(f"Found {len(live_scores)} live games")
        # Could process these separately if needed
        
    # 4. Fetch protected markets (if API key available)
    if API_KEY:
        for chain_name, chain_config in CHAINS.items():
            logger.info(f"\n🔒 Fetching protected markets for {chain_config['name']}...")
            markets = fetch_protected_markets(chain_config['id'])
            
            if markets:
                logger.info(f"Found {len(markets)} markets on {chain_config['name']}")
                
                # Process market-specific data
                markets_data = []
                for market in markets:
                    # Convert market format to game format
                    game_data = {
                        'gameId': market.get('gameId'),
                        'homeTeam': market.get('homeTeam'),
                        'awayTeam': market.get('awayTeam'),
                        'startTime': market.get('maturityDate'),
                        'sportId': market.get('sport'),
                        'league': market.get('league'),
                        'odds': market.get('odds', [])
                    }
                    markets_data.append(game_data)
                    
                added = process_games_into_markets(markets_data, f"v2_{chain_name}_protected")
                total_added += added
                
    # Clear sample data if we found real markets
    if total_added > 0:
        with db_manager.get_db_session() as db:
            logger.info("\n🧹 Clearing sample data...")
            sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
            for market in sample_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            logger.info(f"Removed {len(sample_markets)} sample markets")
            
        logger.info(f"\n🎆 SUCCESS! Added {total_added} real markets!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
    else:
        logger.info("\nℹ️ No active markets found")
        logger.info("This could be due to:")
        logger.info("  • Off-season (no current games)")
        logger.info("  • Need API key for market data")
        logger.info("  • All games already started/resolved")
        
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
        
        # Show examples
        if active > 0:
            examples = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(3).all()
            
            logger.info("\n🏆 Upcoming markets:")
            for m in examples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    {m.sport} - {m.maturity_date.strftime('%Y-%m-%d %H:%M UTC')}")
                logger.info(f"    Source: {m.source}")
                
        logger.info("\n💡 Next steps:")
        if not API_KEY:
            logger.info("1. Get API key from Overtime Discord")
            logger.info("2. Set environment variable: export OVERTIME_API_KEY='your-key'")
            logger.info("3. Re-run to fetch full market data with odds")
        else:
            logger.info("1. Set up scheduled refresh (cron/scheduler)")
            logger.info("2. Monitor root updates on-chain")
            logger.info("3. Refresh markets when roots change")

if __name__ == "__main__":
    main()