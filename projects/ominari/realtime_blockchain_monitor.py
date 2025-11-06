#!/usr/bin/env python3
"""Real-time blockchain monitor for Overtime V2 markets"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from web3 import Web3
from datetime import datetime, timezone, timedelta
import json
import logging
import time
import asyncio
from database_v2 import db_manager
from models import Market, Odd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RPC endpoints - using public endpoints
RPC_ENDPOINTS = {
    'optimism': 'https://mainnet.optimism.io',
    'arbitrum': 'https://arb1.arbitrum.io/rpc'
}

# Overtime V2 contract
SPORTS_AMM_V2 = "0xFb4e4811C7A811E098A556bD79B64c20b479E431"

# Connect to Optimism
w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINTS['optimism']))
logger.info(f"Connected to Optimism: {w3.is_connected()}")

def decode_game_id(game_id_bytes):
    """Decode game ID from bytes32"""
    try:
        if isinstance(game_id_bytes, bytes):
            decoded = game_id_bytes.hex()
        else:
            decoded = game_id_bytes
        # Remove trailing zeros
        decoded = decoded.rstrip('0')
        if len(decoded) % 2 == 1:
            decoded += '0'
        try:
            game_str = bytes.fromhex(decoded).decode('utf-8').strip('\x00')
            return game_str
        except:
            return decoded[:16]  # Return first 16 chars
    except:
        return str(game_id_bytes)[:16]

def get_market_from_game_id(game_id):
    """Try to get market details using game ID"""
    try:
        # The game IDs look like dates - parse them
        # Format appears to be: YYYYMMDDHEXCODE
        if game_id.startswith('20'):
            date_str = game_id[:8]  # YYYYMMDD
            
            # Try Overtime API directly (might work for some endpoints)
            try:
                # Try to fetch from public endpoints
                for network in ['10', '42161']:  # Optimism, Arbitrum
                    url = f"https://overtimemarketsv2.xyz/live-trading-processor/{network}/game/{game_id}"
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            return {
                                'game_id': game_id,
                                'home_team': data.get('homeTeam', 'Unknown'),
                                'away_team': data.get('awayTeam', 'Unknown'),
                                'sport': data.get('sport', 'Unknown'),
                                'start_time': datetime.fromtimestamp(data.get('startTime', 0), tz=timezone.utc),
                                'odds': data.get('odds', {})
                            }
            except:
                pass
            
            # Return basic info from game ID
            return {
                'game_id': game_id,
                'date': date_str,
                'sport': 'Unknown',
                'league': 'Unknown'
            }
    except Exception as e:
        logger.error(f"Error parsing game ID {game_id}: {e}")
    
    return None

def monitor_live_events():
    """Monitor blockchain events in real-time"""
    logger.info("🔴 Starting real-time blockchain monitoring...")
    
    # Event signature for oracle updates
    event_signature = "0xc6fa3d673d901ef180e5a314ff8ede38ac8ba226ce71c9d822ed8a438020a1ab"
    
    # Track processed games
    processed_games = set()
    
    # Get current block
    current_block = w3.eth.block_number
    
    logger.info(f"Starting from block {current_block}")
    logger.info("Monitoring for new game events...")
    
    markets_added = 0
    
    while True:
        try:
            # Get latest block
            latest_block = w3.eth.block_number
            
            if latest_block > current_block:
                # Get logs for new blocks
                logs = w3.eth.get_logs({
                    'fromBlock': current_block + 1,
                    'toBlock': latest_block,
                    'address': SPORTS_AMM_V2,
                    'topics': [event_signature]
                })
                
                if logs:
                    logger.info(f"\n🎯 Found {len(logs)} new events in blocks {current_block + 1} to {latest_block}")
                    
                    for log in logs:
                        try:
                            # Extract game ID
                            data = log['data']
                            if isinstance(data, str):
                                data = bytes.fromhex(data[2:])
                            
                            game_id_bytes = data[:32]
                            game_id = decode_game_id(game_id_bytes)
                            
                            if game_id not in processed_games:
                                processed_games.add(game_id)
                                logger.info(f"\n🆕 New Game ID: {game_id}")
                                
                                # Try to get market details
                                market_info = get_market_from_game_id(game_id)
                                if market_info and 'home_team' in market_info:
                                    # Save to database
                                    if save_market_to_db(market_info):
                                        markets_added += 1
                                        logger.info(f"✅ Added market: {market_info['home_team']} vs {market_info['away_team']}")
                                
                        except Exception as e:
                            logger.error(f"Error processing log: {e}")
                
                current_block = latest_block
            
            # Show status
            if markets_added > 0:
                logger.info(f"\n📊 Total markets added: {markets_added}")
            
            # Wait before next check
            time.sleep(10)  # Check every 10 seconds
            
        except KeyboardInterrupt:
            logger.info("\n⛔ Monitoring stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            time.sleep(5)

def save_market_to_db(market_info):
    """Save market to database"""
    try:
        with db_manager.get_db_session() as db:
            source_id = f"overtime_live_{market_info['game_id']}"
            
            # Check if exists
            existing = db.query(Market).filter(
                Market.source_id == source_id
            ).first()
            
            if not existing:
                market = Market(
                    source_id=source_id,
                    home_team=market_info.get('home_team', 'TBD'),
                    away_team=market_info.get('away_team', 'TBD'),
                    sport=market_info.get('sport', 'Unknown'),
                    league_name=market_info.get('league', ''),
                    maturity_date=market_info.get('start_time', datetime.now(timezone.utc) + timedelta(days=1)),
                    source='blockchain_live',
                    is_finished=False
                )
                db.add(market)
                db.commit()
                
                # Add odds if available
                odds_data = market_info.get('odds', {})
                if odds_data:
                    for outcome, odds_value in odds_data.items():
                        if odds_value and float(odds_value) > 1:
                            odd = Odd(
                                source_id=source_id,
                                outcome=outcome.lower(),
                                decimal_odds=float(odds_value),
                                market_type='winner',
                                source='blockchain_live',
                                bookmaker='overtime_v2',
                                normalized_implied=1.0/float(odds_value)
                            )
                            db.add(odd)
                    db.commit()
                
                return True
                
    except Exception as e:
        logger.error(f"Error saving market: {e}")
        
    return False

def check_existing_games():
    """Check recently discovered game IDs"""
    game_ids = [
        '202510317EFAD2A8',
        '202511023E760198',
        '2025110268F3E34E',
        '2025103143790DE8',
        '20251031B1BA8034',
        '202510311C9DB34D',
        '202511011B429F79',
        '2025103076829B99'
    ]
    
    logger.info("🔍 Checking discovered game IDs...")
    
    for game_id in game_ids[:5]:  # Check first 5
        logger.info(f"\nChecking: {game_id}")
        market_info = get_market_from_game_id(game_id)
        if market_info:
            logger.info(f"  Date: {market_info.get('date', 'Unknown')}")
            if 'home_team' in market_info:
                logger.info(f"  Teams: {market_info['home_team']} vs {market_info['away_team']}")

if __name__ == "__main__":
    logger.info("🚀 Overtime V2 Real-Time Monitor")
    logger.info("=" * 60)
    
    # First check some existing games
    check_existing_games()
    
    # Then start real-time monitoring
    logger.info("\n" + "=" * 60)
    monitor_live_events()