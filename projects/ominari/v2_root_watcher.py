#!/usr/bin/env python3
"""
Overtime V2 Root Watcher - Monitors GameRootUpdated events and fetches market data
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
import time
from eth_utils import encode_hex

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 Configuration
V2_CONFIG = {
    'optimism': {
        'chain_id': 10,
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'explorer_api': 'https://api-optimistic.etherscan.io/api',
        'name': 'Optimism'
    },
    'arbitrum': {
        'chain_id': 42161,
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'explorer_api': 'https://api.arbiscan.io/api',
        'name': 'Arbitrum'
    }
}

API_BASE = 'https://api.overtime.io'

def get_game_root_updated_events(w3, amm_address, from_block, to_block):
    """Get GameRootUpdated events from the AMM."""
    # GameRootUpdated(bytes32,bytes32) - compute topic
    event_signature = Web3.keccak(text='GameRootUpdated(bytes32,bytes32)').hex()
    
    logger.info(f"Scanning for GameRootUpdated events from block {from_block} to {to_block}")
    
    try:
        logs = w3.eth.get_logs({
            'address': amm_address,
            'fromBlock': from_block,
            'toBlock': to_block,
            'topics': [event_signature]
        })
        
        events = []
        for log in logs:
            # Decode the event
            # First topic is event signature
            # Data contains: game_id (bytes32) and root (bytes32)
            data = log['data']
            
            if len(data) >= 130:  # 0x + 64 chars + 64 chars
                game_id = '0x' + data[2:66]
                root = '0x' + data[66:130]
                
                events.append({
                    'game_id': game_id,
                    'root': root,
                    'block': log['blockNumber'],
                    'tx_hash': log['transactionHash'].hex()
                })
                
        return events
        
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        return []

def fetch_markets_from_api(chain_id, game_ids):
    """Fetch market data from Overtime V2 API."""
    if not game_ids:
        return []
        
    # Convert game_ids to comma-separated string
    ids_param = ','.join(game_ids)
    
    url = f"{API_BASE}/overtime-v2/networks/{chain_id}/games"
    params = {'ids': ids_param}
    
    logger.info(f"Fetching markets from API for games: {ids_param[:50]}...")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        return data.get('games', [])
        
    except Exception as e:
        logger.error(f"Error fetching from API: {e}")
        return []

def process_v2_markets(chain_name, markets_data):
    """Process V2 market data and save to database."""
    markets_added = 0
    
    for game in markets_data:
        try:
            game_id = game.get('gameId')
            sport_id = game.get('sportId')
            league_id = game.get('leagueId')
            home_team = game.get('homeTeam')
            away_team = game.get('awayTeam')
            start_time = game.get('startTime')
            
            if not all([game_id, home_team, away_team, start_time]):
                continue
                
            # Convert timestamp to datetime
            maturity = datetime.fromtimestamp(start_time, tz=timezone.utc)
            
            # Skip past games
            if maturity < datetime.now(timezone.utc):
                continue
                
            # Create market ID
            market_id = f"{chain_name}_v2_{game_id[-8:]}"
            
            with db_manager.get_db_session() as db:
                # Check if already exists
                existing = db.query(Market).filter(Market.source_id == market_id).first()
                if existing:
                    continue
                    
                # Map sport ID to name
                sport_map = {
                    1: "American Football",
                    2: "Baseball", 
                    3: "Basketball",
                    4: "Soccer",
                    5: "Hockey",
                    6: "MMA",
                    7: "Boxing",
                    8: "Tennis",
                    9: "Cricket",
                    10: "Golf"
                }
                
                market = Market(
                    source_id=market_id,
                    source=f"{chain_name}_v2_api",
                    sport=sport_map.get(sport_id, "Other"),
                    league_name=game.get('leagueName', 'Unknown League'),
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                db.flush()
                
                # Process markets (lines/odds)
                for market_data in game.get('markets', []):
                    market_type = market_data.get('type')
                    type_id = market_data.get('typeId')
                    line = market_data.get('line', 0)
                    
                    # Get odds for each position
                    for i, odd_data in enumerate(market_data.get('odds', [])):
                        decimal_odds = odd_data / 1e18 if odd_data else None
                        
                        if decimal_odds and decimal_odds > 1.0:
                            # Determine outcome
                            if type_id == 0:  # Winner market
                                outcome = ['home', 'away', 'draw'][i] if i < 3 else None
                            elif type_id in [1, 2]:  # Spread
                                outcome = ['home', 'away'][i] if i < 2 else None
                            elif type_id == 3:  # Total
                                outcome = ['over', 'under'][i] if i < 2 else None
                            else:
                                outcome = f"position_{i}"
                                
                            if outcome:
                                # Convert to American odds
                                if decimal_odds >= 2.0:
                                    american = int((decimal_odds - 1) * 100)
                                else:
                                    american = int(-100 / (decimal_odds - 1))
                                    
                                odd = Odd(
                                    source_id=market.source_id,
                                    market_type=market_type or "winner",
                                    outcome=outcome,
                                    source=f"{chain_name}_v2_api",
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
                logger.info(f"   Start: {maturity}")
                logger.info(f"   Game ID: {game_id}")
                
        except Exception as e:
            logger.error(f"Error processing market: {e}")
            continue
            
    return markets_added

def watch_v2_roots():
    """Main function to watch for root updates and fetch markets."""
    logger.info("🎯 Overtime V2 Root Watcher")
    logger.info("=" * 60)
    logger.info("Monitoring GameRootUpdated events and fetching market data")
    
    # Track last processed block for each chain
    last_blocks = {}
    
    # Initial scan - go back 10k blocks (smaller to avoid RPC limits)
    initial_lookback = 10000
    
    while True:
        try:
            total_markets_added = 0
            all_game_ids = set()
            
            # Process each chain
            for chain_name, config in V2_CONFIG.items():
                logger.info(f"\n🌐 Processing {config['name']}...")
                
                # Connect to RPC
                w3 = Web3(Web3.HTTPProvider(config['rpc']))
                if not w3.is_connected():
                    logger.error(f"Failed to connect to {config['name']}")
                    continue
                    
                current_block = w3.eth.block_number
                
                # Determine from_block
                if chain_name not in last_blocks:
                    from_block = current_block - initial_lookback
                else:
                    from_block = last_blocks[chain_name] + 1
                    
                # Get events
                events = get_game_root_updated_events(
                    w3, 
                    Web3.to_checksum_address(config['amm']),
                    hex(from_block),  # Convert to hex string
                    hex(current_block)  # Convert to hex string
                )
                
                if events:
                    logger.info(f"Found {len(events)} GameRootUpdated events")
                    
                    # Extract unique game IDs
                    game_ids = list(set(e['game_id'] for e in events))
                    all_game_ids.update(game_ids)
                    
                    # Fetch market data from API
                    markets_data = fetch_markets_from_api(config['chain_id'], game_ids)
                    
                    if markets_data:
                        logger.info(f"API returned {len(markets_data)} games")
                        
                        # Process and save markets
                        added = process_v2_markets(chain_name, markets_data)
                        total_markets_added += added
                        
                # Update last processed block
                last_blocks[chain_name] = current_block
                
            # If we added any markets, clear sample data
            if total_markets_added > 0:
                with db_manager.get_db_session() as db:
                    logger.info("\n🧹 Clearing sample data...")
                    sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
                    for market in sample_markets:
                        db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                        db.delete(market)
                    db.commit()
                    logger.info(f"Removed {len(sample_markets)} sample markets")
                    
                logger.info(f"\n🎆 SUCCESS! Added {total_markets_added} real V2 markets!")
                logger.info("Dashboard at http://localhost:8888/unified now shows REAL blockchain data!")
                
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
                logger.info(f"Game IDs being tracked: {len(all_game_ids)}")
                
            # Wait before next poll
            logger.info("\n⏰ Waiting 30 seconds before next check...")
            time.sleep(30)
            
        except KeyboardInterrupt:
            logger.info("\n👋 Stopping root watcher...")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(30)

if __name__ == "__main__":
    watch_v2_roots()