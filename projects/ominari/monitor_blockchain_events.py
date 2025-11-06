#!/usr/bin/env python3
"""Monitor Overtime V2 blockchain events for real market data"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from web3 import Web3
from datetime import datetime, timezone
import json
import logging
from database_v2 import db_manager
from models import Market, Odd
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RPC endpoints
RPC_ENDPOINTS = {
    'optimism': 'https://mainnet.optimism.io',
    'arbitrum': 'https://arb1.arbitrum.io/rpc'
}

# Overtime V2 contract addresses (verified from documentation)
SPORTS_AMM_V2 = "0xFb4e4811C7A811E098A556bD79B64c20b479E431"
LIVE_TRADING_PROCESSOR = "0x3b834149F21B9A6C2DDC9F6ce97F2FD1097F8EAB"

# Connect to Optimism
w3 = Web3(Web3.HTTPProvider(RPC_ENDPOINTS['optimism']))
logger.info(f"Connected to Optimism: {w3.is_connected()}")

# Load ABI for SportsAMM V2
try:
    with open('abi/SportsAMMV2.json', 'r') as f:
        sports_amm_abi = json.load(f)
except:
    # Minimal ABI for event monitoring
    sports_amm_abi = [
        {
            "anonymous": False,
            "inputs": [
                {"indexed": False, "name": "gameId", "type": "bytes32"},
                {"indexed": False, "name": "merkleRoot", "type": "bytes32"}
            ],
            "name": "GameOddsAdded",
            "type": "event"
        },
        {
            "anonymous": False,
            "inputs": [
                {"indexed": True, "name": "buyer", "type": "address"},
                {"indexed": False, "name": "market", "type": "address"},
                {"indexed": False, "name": "position", "type": "uint8"},
                {"indexed": False, "name": "amount", "type": "uint256"},
                {"indexed": False, "name": "sUSDPaid", "type": "uint256"}
            ],
            "name": "BoughtFromAMM",
            "type": "event"
        },
        {
            "inputs": [{"name": "_gameId", "type": "bytes32"}],
            "name": "getGameDetails",
            "outputs": [
                {"name": "gameId", "type": "bytes32"},
                {"name": "startTime", "type": "uint256"},
                {"name": "homeTeam", "type": "string"},
                {"name": "awayTeam", "type": "string"},
                {"name": "homeOdds", "type": "uint256"},
                {"name": "awayOdds", "type": "uint256"},
                {"name": "drawOdds", "type": "uint256"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]

# Initialize contract
contract = w3.eth.contract(address=SPORTS_AMM_V2, abi=sports_amm_abi)

def decode_game_id(game_id_bytes):
    """Decode game ID from bytes32"""
    try:
        # Try to decode as UTF-8
        decoded = game_id_bytes.hex()
        # Remove trailing zeros
        decoded = decoded.rstrip('0')
        if len(decoded) % 2 == 1:
            decoded += '0'
        try:
            return bytes.fromhex(decoded).decode('utf-8').strip('\x00')
        except:
            return decoded
    except:
        return game_id_bytes.hex()

def monitor_recent_events():
    """Monitor recent blockchain events for market data"""
    logger.info("🔍 Scanning recent Overtime V2 events...")
    
    # Get recent block range
    latest_block = w3.eth.block_number
    from_block = latest_block - 1000  # Last 1000 blocks (~4 hours on Optimism)
    
    logger.info(f"Scanning blocks {from_block} to {latest_block}")
    
    # Look for oracle update events (most common pattern found)
    event_signature = "0xc6fa3d673d901ef180e5a314ff8ede38ac8ba226ce71c9d822ed8a438020a1ab"
    
    # Get logs
    logs = w3.eth.get_logs({
        'fromBlock': from_block,
        'toBlock': latest_block,
        'address': SPORTS_AMM_V2,
        'topics': [event_signature]
    })
    
    logger.info(f"Found {len(logs)} oracle events")
    
    # Track unique game IDs
    game_ids = set()
    
    for log in logs[-20:]:  # Last 20 events
        try:
            # Extract game ID from data
            data = log['data']
            if isinstance(data, str):
                data = bytes.fromhex(data[2:])
            
            game_id_bytes = data[:32]
            game_id = decode_game_id(game_id_bytes)
            game_ids.add(game_id_bytes)
            
            logger.info(f"Game ID: {game_id}")
            
        except Exception as e:
            logger.error(f"Error processing log: {e}")
    
    # Try to get details for discovered games
    logger.info(f"\n🎮 Attempting to fetch details for {len(game_ids)} games...")
    
    markets_found = []
    
    for game_id_bytes in list(game_ids)[:5]:  # Try first 5
        try:
            # Call getGameDetails
            result = contract.functions.getGameDetails(game_id_bytes).call()
            
            if result and len(result) >= 7:
                game_id, start_time, home_team, away_team, home_odds, away_odds, draw_odds = result
                
                # Convert odds from contract format (usually 18 decimals)
                home_odds_decimal = home_odds / 1e18 if home_odds > 0 else 0
                away_odds_decimal = away_odds / 1e18 if away_odds > 0 else 0
                draw_odds_decimal = draw_odds / 1e18 if draw_odds > 0 else 0
                
                if home_odds_decimal > 1 and away_odds_decimal > 1:
                    market_data = {
                        'game_id': decode_game_id(game_id),
                        'start_time': datetime.fromtimestamp(start_time, tz=timezone.utc),
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_odds': home_odds_decimal,
                        'away_odds': away_odds_decimal,
                        'draw_odds': draw_odds_decimal
                    }
                    
                    markets_found.append(market_data)
                    logger.info(f"✅ Found market: {home_team} vs {away_team}")
                    logger.info(f"   Start: {market_data['start_time']}")
                    logger.info(f"   Odds: H:{home_odds_decimal:.2f} D:{draw_odds_decimal:.2f} A:{away_odds_decimal:.2f}")
                    
        except Exception as e:
            logger.debug(f"Could not fetch details for game: {e}")
    
    return markets_found

def monitor_buy_events():
    """Monitor recent buy events to find active markets"""
    logger.info("\n🛒 Looking for recent betting activity...")
    
    latest_block = w3.eth.block_number
    from_block = latest_block - 5000  # Last ~20 hours
    
    # BoughtFromAMM event
    buy_event = contract.events.BoughtFromAMM()
    
    try:
        events = buy_event.get_logs(fromBlock=from_block, toBlock=latest_block)
        logger.info(f"Found {len(events)} buy events")
        
        unique_markets = set()
        for event in events[-10:]:  # Last 10 buys
            market_address = event['args']['market']
            unique_markets.add(market_address)
            
        logger.info(f"Found {len(unique_markets)} unique markets with activity")
        
        # Try to get market details
        for market_addr in list(unique_markets)[:3]:
            logger.info(f"Market with activity: {market_addr}")
            
    except Exception as e:
        logger.error(f"Error fetching buy events: {e}")

def save_real_markets(markets):
    """Save discovered markets to database"""
    if not markets:
        return
        
    logger.info(f"\n💾 Saving {len(markets)} real blockchain markets...")
    
    with db_manager.get_db_session() as db:
        for market_data in markets:
            try:
                # Create unique source ID
                source_id = f"overtime_v2_{market_data['game_id']}"
                
                # Check if exists
                existing = db.query(Market).filter(
                    Market.source_id == source_id
                ).first()
                
                if not existing:
                    market = Market(
                        source_id=source_id,
                        home_team=market_data['home_team'],
                        away_team=market_data['away_team'],
                        sport='Soccer',  # Default, would need sport detection
                        league_name='',
                        maturity_date=market_data['start_time'],
                        source='blockchain_v2_live',
                        is_finished=False
                    )
                    db.add(market)
                    db.commit()
                    
                    # Add odds
                    for outcome, odds_value in [
                        ('home', market_data['home_odds']),
                        ('draw', market_data['draw_odds']),
                        ('away', market_data['away_odds'])
                    ]:
                        if odds_value > 1:
                            odd = Odd(
                                source_id=source_id,
                                outcome=outcome,
                                decimal_odds=float(odds_value),
                                market_type='winner',
                                source='blockchain_v2',
                                bookmaker='overtime_v2',
                                normalized_implied=1.0/float(odds_value)
                            )
                            db.add(odd)
                    
                    db.commit()
                    logger.info(f"✅ Added: {market_data['home_team']} vs {market_data['away_team']}")
                    
            except Exception as e:
                logger.error(f"Error saving market: {e}")
                db.rollback()

def main():
    """Main monitoring function"""
    logger.info("🚀 Overtime V2 Blockchain Monitor")
    logger.info("=" * 60)
    
    # Monitor recent events
    markets = monitor_recent_events()
    
    # Also check buy events
    monitor_buy_events()
    
    if markets:
        save_real_markets(markets)
        logger.info(f"\n✅ Successfully found {len(markets)} real blockchain markets!")
    else:
        logger.info("\n⚠️  No market details could be retrieved")
        logger.info("The contract is active but requires specific game IDs or parameters")
        logger.info("Consider:")
        logger.info("  1. Monitoring events in real-time for new games")
        logger.info("  2. Getting Overtime API access for easier integration")
        logger.info("  3. Using alternative blockchain betting protocols")
    
    # Show current database status
    with db_manager.get_db_session() as db:
        blockchain_markets = db.query(Market).filter(
            Market.source.like('%blockchain%'),
            Market.is_finished == False
        ).count()
        
        logger.info(f"\n📊 Database Status:")
        logger.info(f"Total blockchain markets: {blockchain_markets}")

if __name__ == "__main__":
    main()