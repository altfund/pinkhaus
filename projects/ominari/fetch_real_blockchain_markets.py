#!/usr/bin/env python3
"""
Fetch real Overtime V2 markets using blockchain events + public API
No API keys required - uses public endpoints
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chain configurations
CHAINS = {
    'optimism': {
        'id': 10,
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'id': 42161, 
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

# Try multiple API endpoints
API_ENDPOINTS = [
    'https://api.overtime.io',
    'https://overtimemarketsv2.xyz',
    'https://api.thalesmarket.io'
]

def get_recent_game_ids_from_chain(w3, amm_address, hours_back=48):
    """Get game IDs from recent GameRootUpdated events."""
    game_ids = set()
    
    try:
        current_block = w3.eth.block_number
        blocks_per_hour = 300  # ~12s blocks on Arbitrum/Optimism
        from_block = current_block - (blocks_per_hour * hours_back)
        
        # GameRootUpdated(bytes32,bytes32)
        event_sig = Web3.keccak(text='GameRootUpdated(bytes32,bytes32)').hex()
        
        # Scan in chunks to avoid RPC limits
        chunk_size = 5000
        
        for start in range(from_block, current_block, chunk_size):
            end = min(start + chunk_size, current_block)
            
            try:
                logs = w3.eth.get_logs({
                    'address': amm_address,
                    'fromBlock': hex(start),
                    'toBlock': hex(end),
                    'topics': [event_sig]
                })
                
                for log in logs:
                    if len(log['data']) >= 130:  # Has both game_id and root
                        game_id = '0x' + log['data'][2:66]
                        game_ids.add(game_id)
                        
            except Exception as e:
                logger.debug(f"Error in block range {start}-{end}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error getting game IDs: {e}")
        
    return list(game_ids)

def fetch_markets_from_api(chain_id, game_ids=None):
    """Fetch market data from public API."""
    markets = []
    
    for api_base in API_ENDPOINTS:
        try:
            # Try with game IDs if provided
            if game_ids:
                ids_str = ','.join(game_ids[:20])  # Limit to 20
                endpoints = [
                    f"{api_base}/overtime-v2/networks/{chain_id}/games?ids={ids_str}",
                    f"{api_base}/v2/games?network={chain_id}&ids={ids_str}",
                    f"{api_base}/games?chainId={chain_id}&gameIds={ids_str}"
                ]
            else:
                # Try to get all recent games
                endpoints = [
                    f"{api_base}/overtime-v2/networks/{chain_id}/games",
                    f"{api_base}/v2/games?network={chain_id}",
                    f"{api_base}/games?chainId={chain_id}&resolved=false",
                    f"{api_base}/sports-markets?network={chain_id}"
                ]
                
            for endpoint in endpoints:
                try:
                    logger.debug(f"Trying: {endpoint}")
                    response = requests.get(endpoint, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Handle different response formats
                        if isinstance(data, list) and len(data) > 0:
                            markets = data
                            logger.info(f"✅ Found {len(markets)} markets from {api_base}")
                            return markets
                        elif isinstance(data, dict):
                            for key in ['games', 'markets', 'data']:
                                if key in data and isinstance(data[key], list) and len(data[key]) > 0:
                                    markets = data[key]
                                    logger.info(f"✅ Found {len(markets)} markets from {api_base}")
                                    return markets
                                    
                except Exception as e:
                    continue
                    
        except Exception as e:
            continue
            
    return markets

def get_recent_trades_from_chain(w3, amm_address, hours_back=24):
    """Get market addresses from recent trade events."""
    market_addresses = set()
    
    try:
        current_block = w3.eth.block_number
        blocks_per_hour = 300
        from_block = current_block - (blocks_per_hour * hours_back)
        
        # Look for any events from AMM
        logs = w3.eth.get_logs({
            'address': amm_address,
            'fromBlock': hex(from_block),
            'toBlock': hex(current_block)
        })
        
        logger.info(f"Found {len(logs)} total events from AMM in last {hours_back} hours")
        
        # Analyze logs for market addresses
        for log in logs:
            # Check topics for addresses (skip event signature)
            for topic in log['topics'][1:]:
                topic_hex = topic.hex() if hasattr(topic, 'hex') else topic
                addr = '0x' + topic_hex[-40:]
                try:
                    addr = Web3.to_checksum_address(addr)
                    # Check if it's a minimal proxy (market)
                    code = w3.eth.get_code(addr)
                    if code and len(code) == 45:
                        market_addresses.add(addr)
                except:
                    pass
                    
            # Check data for addresses
            if log['data']:
                data = log['data'].hex() if hasattr(log['data'], 'hex') else log['data']
                data = data[2:] if data.startswith('0x') else data
                
                # Each parameter is 64 hex chars
                for i in range(0, len(data), 64):
                    param = data[i:i+64]
                    if len(param) == 64:
                        addr = '0x' + param[-40:]
                        try:
                            addr = Web3.to_checksum_address(addr)
                            if addr != '0x' + '0' * 40:
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:
                                    market_addresses.add(addr)
                        except:
                            pass
                            
    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        
    return list(market_addresses)

def get_market_details_onchain(w3, market_address):
    """Get market details directly from blockchain."""
    try:
        # getGameDetails() - 0x5ee0fe28
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5ee0fe28'
        })
        
        details = {}
        if result and len(result) > 96:
            # Decode game label
            offset = int.from_bytes(result[32:64], 'big')
            length = int.from_bytes(result[offset:offset+32], 'big')
            if 0 < length < 1000:
                game_label = result[offset+32:offset+32+length].decode('utf-8', errors='ignore').strip()
                details['gameLabel'] = game_label
                
                # Parse teams
                if ' vs ' in game_label:
                    parts = game_label.split(' vs ')
                    details['homeTeam'] = parts[0].strip()
                    details['awayTeam'] = parts[1].strip()
                elif ' @ ' in game_label:
                    parts = game_label.split(' @ ')
                    details['awayTeam'] = parts[0].strip()
                    details['homeTeam'] = parts[1].strip()
                    
        # times() - 0xd0370218
        result = w3.eth.call({
            'to': market_address,
            'data': '0xd0370218'
        })
        if result and len(result) >= 32:
            maturity = int.from_bytes(result[:32], 'big')
            if maturity > 0:
                details['maturity'] = maturity
                
        # resolved() - 0x5fe138b5
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5fe138b5'
        })
        if result:
            details['resolved'] = result[-1] == 1
            
        return details
        
    except:
        return None

def process_markets(chain_name, markets_data, source="api"):
    """Process and save market data."""
    markets_added = 0
    chain_id = CHAINS[chain_name]['id']
    
    for market in markets_data:
        try:
            # Extract details based on source
            if source == "api":
                game_id = market.get('gameId', market.get('id'))
                home_team = market.get('homeTeam')
                away_team = market.get('awayTeam')
                start_time = market.get('startTime', market.get('maturityDate'))
                sport_id = market.get('sport', market.get('sportId'))
                league = market.get('league', market.get('leagueName', 'Unknown'))
                
                # Convert timestamp
                if isinstance(start_time, str):
                    maturity = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                else:
                    maturity = datetime.fromtimestamp(start_time, tz=timezone.utc)
                    
            else:  # blockchain source
                game_id = market.get('address', '')
                home_team = market.get('homeTeam')
                away_team = market.get('awayTeam')
                maturity = datetime.fromtimestamp(market.get('maturity', 0), tz=timezone.utc)
                sport_id = None
                league = 'Overtime V2'
                
            if not all([home_team, away_team, maturity]):
                continue
                
            # Skip past games
            if maturity < datetime.now(timezone.utc):
                continue
                
            # Create unique ID
            market_id = f"{chain_name}_v2_{str(game_id)[-8:]}"
            
            with db_manager.get_db_session() as db:
                # Skip if exists
                if db.query(Market).filter(Market.source_id == market_id).first():
                    continue
                    
                # Sport mapping
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
                
                market_obj = Market(
                    source_id=market_id,
                    source=f"{chain_name}_v2_blockchain",
                    sport=sport_map.get(sport_id, "Soccer"),
                    league_name=league,
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market_obj)
                
                # Add odds if available (from API)
                if 'markets' in market:
                    for mkt in market['markets']:
                        odds = mkt.get('odds', [])
                        for i, odd_val in enumerate(odds):
                            if odd_val and odd_val > 0:
                                decimal_odds = odd_val / 1e18 if odd_val > 1000 else odd_val
                                if 1.0 < decimal_odds < 100:
                                    outcome = ['home', 'away', 'draw'][i] if i < 3 else f"pos_{i}"
                                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                                    
                                    odd_obj = Odd(
                                        source_id=market_obj.source_id,
                                        market_type="winner",
                                        outcome=outcome,
                                        source=f"{chain_name}_v2_blockchain",
                                        bookmaker="Overtime V2",
                                        decimal_odds=decimal_odds,
                                        american_odds=american,
                                        normalized_implied=1.0 / decimal_odds,
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(odd_obj)
                                    
                db.commit()
                markets_added += 1
                
                logger.info(f"✅ Added: {home_team} vs {away_team}")
                logger.info(f"   Date: {maturity}")
                logger.info(f"   Chain: {chain_name}")
                
        except Exception as e:
            logger.error(f"Error processing market: {e}")
            continue
            
    return markets_added

def fetch_real_blockchain_markets():
    """Main function to fetch real blockchain markets."""
    logger.info("🎯 Fetching Real Overtime V2 Markets from Blockchain")
    logger.info("=" * 60)
    logger.info("Using blockchain events + public API (no keys required)")
    
    total_added = 0
    
    for chain_name, config in CHAINS.items():
        logger.info(f"\n🌐 Processing {config['name']}...")
        
        try:
            # Connect to chain
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            if not w3.is_connected():
                logger.error(f"Failed to connect to {config['name']}")
                continue
                
            amm = Web3.to_checksum_address(config['amm'])
            
            # Method 1: Get game IDs from recent events
            logger.info("\n1️⃣ Checking GameRootUpdated events...")
            game_ids = get_recent_game_ids_from_chain(w3, amm, hours_back=24)
            
            if game_ids:
                logger.info(f"Found {len(game_ids)} games with recent updates")
                
                # Fetch details from API
                markets = fetch_markets_from_api(config['id'], game_ids)
                if markets:
                    added = process_markets(chain_name, markets, source="api")
                    total_added += added
                    
            # Method 2: Get markets from recent trades
            logger.info("\n2️⃣ Checking recent trading activity...")
            market_addrs = get_recent_trades_from_chain(w3, amm, hours_back=12)
            
            if market_addrs:
                logger.info(f"Found {len(market_addrs)} markets from trades")
                
                # Get details directly from blockchain
                blockchain_markets = []
                for addr in market_addrs[:10]:  # Check first 10
                    details = get_market_details_onchain(w3, addr)
                    if details:
                        details['address'] = addr
                        blockchain_markets.append(details)
                        
                if blockchain_markets:
                    added = process_markets(chain_name, blockchain_markets, source="blockchain")
                    total_added += added
                    
            # Method 3: Try to get all markets from API
            logger.info("\n3️⃣ Checking API for all active markets...")
            all_markets = fetch_markets_from_api(config['id'])
            
            if all_markets:
                logger.info(f"API returned {len(all_markets)} markets")
                added = process_markets(chain_name, all_markets, source="api")
                total_added += added
                
        except Exception as e:
            logger.error(f"Error processing {chain_name}: {e}")
            continue
            
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
            
        logger.info(f"\n🎉 SUCCESS! Added {total_added} real blockchain markets!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
    else:
        logger.info("\n⚠️ No active markets found")
        logger.info("This could be due to:")
        logger.info("  • Off-season period (no current games)")
        logger.info("  • Markets already resolved")
        logger.info("  • API endpoint changes")
        
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
            ).order_by(Market.maturity_date).limit(5).all()
            
            logger.info("\n🏆 Upcoming markets:")
            for m in examples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    {m.sport} - {m.maturity_date.strftime('%Y-%m-%d %H:%M UTC')}")
                logger.info(f"    Source: {m.source}")

if __name__ == "__main__":
    fetch_real_blockchain_markets()