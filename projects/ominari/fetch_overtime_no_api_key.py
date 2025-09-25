#!/usr/bin/env python3
"""
Fetch Overtime V2 markets WITHOUT API key
Uses blockchain analysis and public data sources
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chain configurations
CHAINS = {
    'optimism': {
        'id': 10,
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'explorer': 'https://optimistic.etherscan.io',
        'name': 'Optimism'
    },
    'arbitrum': {
        'id': 42161,
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'explorer': 'https://arbiscan.io',
        'name': 'Arbitrum'
    }
}

def scan_all_amm_events(w3, amm_address, hours=24):
    """Scan ALL events from AMM to find market addresses."""
    markets_found = set()
    
    try:
        current_block = w3.eth.block_number
        blocks_per_hour = 300
        from_block = current_block - (blocks_per_hour * hours)
        
        logger.info(f"Scanning blocks {from_block} to {current_block}")
        
        # Get ALL logs from AMM
        chunk_size = 2000
        total_logs = 0
        
        for start in range(from_block, current_block, chunk_size):
            end = min(start + chunk_size, current_block)
            
            try:
                logs = w3.eth.get_logs({
                    'address': amm_address,
                    'fromBlock': hex(start),
                    'toBlock': hex(end)
                })
                
                total_logs += len(logs)
                
                for log in logs:
                    # Check topics for addresses (skip event signature in topic[0])
                    for i, topic in enumerate(log['topics'][1:], 1):
                        topic_hex = topic.hex() if hasattr(topic, 'hex') else str(topic)
                        # Extract potential address
                        if len(topic_hex) >= 40:
                            addr = '0x' + topic_hex[-40:]
                            try:
                                addr = Web3.to_checksum_address(addr)
                                # Check if it's a minimal proxy (market)
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:
                                    markets_found.add(addr)
                                    logger.debug(f"Found market in topic[{i}]: {addr}")
                            except:
                                pass
                    
                    # Check data for addresses
                    if log['data']:
                        data = log['data'].hex() if hasattr(log['data'], 'hex') else log['data']
                        data = data[2:] if data.startswith('0x') else data
                        
                        # Each parameter is 32 bytes (64 hex chars)
                        for i in range(0, len(data), 64):
                            param = data[i:i+64]
                            if len(param) == 64:
                                # Address would be in last 40 chars
                                addr = '0x' + param[-40:]
                                try:
                                    addr = Web3.to_checksum_address(addr)
                                    if addr != '0x' + '0' * 40:
                                        code = w3.eth.get_code(addr)
                                        if code and len(code) == 45:
                                            markets_found.add(addr)
                                            logger.debug(f"Found market in data: {addr}")
                                except:
                                    pass
                                    
            except Exception as e:
                logger.debug(f"Error in chunk {start}-{end}: {e}")
                
        logger.info(f"Scanned {total_logs} total events")
        
    except Exception as e:
        logger.error(f"Error scanning events: {e}")
        
    return list(markets_found)

def scan_recent_trades(w3, amm_address, hours=48):
    """Scan recent trades to find market addresses."""
    markets_found = set()
    
    try:
        current_block = w3.eth.block_number
        blocks_per_hour = 300
        from_block = current_block - (blocks_per_hour * hours)
        
        # BoughtFromAMM event signature
        bought_sig = Web3.keccak(text='BoughtFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)').hex()
        
        # Get logs in chunks
        chunk_size = 2000
        for start in range(from_block, current_block, chunk_size):
            end = min(start + chunk_size, current_block)
            
            try:
                # Get all logs from AMM
                logs = w3.eth.get_logs({
                    'address': amm_address,
                    'fromBlock': hex(start),
                    'toBlock': hex(end)
                })
                
                for log in logs:
                    # Check if it's a trade event
                    if log['topics'] and log['topics'][0].hex() == bought_sig:
                        # Second topic is buyer, third is market
                        if len(log['topics']) >= 3:
                            market_addr = '0x' + log['topics'][2].hex()[-40:]
                            try:
                                market_addr = Web3.to_checksum_address(market_addr)
                                # Verify it's a contract
                                code = w3.eth.get_code(market_addr)
                                if code and len(code) == 45:  # Minimal proxy
                                    markets_found.add(market_addr)
                            except:
                                pass
                                
            except Exception as e:
                logger.debug(f"Error in chunk {start}-{end}: {e}")
                
    except Exception as e:
        logger.error(f"Error scanning trades: {e}")
        
    return list(markets_found)

def get_market_details(w3, market_address):
    """Get market details from blockchain."""
    try:
        details = {'address': market_address}
        
        # getGameDetails() - 0x5ee0fe28
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5ee0fe28'
        })
        
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
            
        # tags() - 0x4bbf5252 (for sport)
        result = w3.eth.call({
            'to': market_address,
            'data': '0x4bbf5252'
        })
        if result and len(result) >= 96:
            # First tag is usually sport ID
            offset = int.from_bytes(result[:32], 'big')
            length = int.from_bytes(result[offset:offset+32], 'big')
            if length > 0:
                sport_id = int.from_bytes(result[offset+64:offset+96], 'big')
                details['sportId'] = sport_id
                
        return details
        
    except Exception as e:
        logger.error(f"Error getting details for {market_address}: {e}")
        return None

def get_odds_from_amm(w3, amm_address, market_address):
    """Try to get odds from AMM."""
    odds = {}
    
    try:
        # getMarketDefaultOdds(address,bool) - 0xb12df8e3
        selector = '0xb12df8e3'
        encoded_market = market_address[2:].lower().rjust(64, '0')
        encoded_bool = '0' * 64  # false
        call_data = selector + encoded_market + encoded_bool
        
        result = w3.eth.call({
            'to': amm_address,
            'data': call_data
        })
        
        if result and len(result) >= 96:
            # Decode array
            offset = int.from_bytes(result[0:32], 'big')
            length = int.from_bytes(result[offset:offset+32], 'big')
            
            if length >= 2:
                for i in range(min(length, 3)):  # Home, Away, Draw
                    odds_val = int.from_bytes(result[offset+64+(i*32):offset+96+(i*32)], 'big')
                    if odds_val > 0:
                        odds[i] = odds_val / 1e18
                        
    except:
        # Try simpler price method
        try:
            # price(address,uint8) - multiple calls
            for position in [0, 1, 2]:
                selector = '0xa035b1fe'
                call_data = selector + encoded_market + str(position).rjust(64, '0')
                
                result = w3.eth.call({
                    'to': amm_address,
                    'data': call_data
                })
                
                if result:
                    price = int.from_bytes(result, 'big')
                    if price > 0:
                        odds[position] = 1 / (price / 1e18) if price < 1e18 else 2.0
        except:
            pass
            
    return odds

def check_public_endpoints():
    """Check for any public endpoints that don't need API key."""
    logger.info("\n🌐 Checking public data sources...")
    
    # Try GraphQL endpoints
    graphql_urls = [
        "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-optimism",
        "https://api.thegraph.com/subgraphs/name/thales-markets/overtime-arbitrum",
        "https://api.thegraph.com/subgraphs/name/overtime/overtime-v2-optimism",
        "https://api.thegraph.com/subgraphs/name/overtime/overtime-v2-arbitrum",
    ]
    
    for url in graphql_urls:
        try:
            query = {
                "query": """{
                    sportMarkets(first: 10, where: {isResolved: false}) {
                        id
                        address
                        maturityDate
                        homeTeam
                        awayTeam
                        tags
                    }
                }"""
            }
            
            response = requests.post(url, json=query, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and data['data']:
                    markets = data['data'].get('sportMarkets', [])
                    if markets:
                        logger.info(f"✅ Found {len(markets)} markets from GraphQL")
                        return markets
        except:
            pass
            
    # Try public API endpoints that might work
    api_urls = [
        "https://api.overtime.io/overtime-v2/games-info",
        "https://api.overtime.io/overtime/markets",
        "https://api.thalesmarket.io/overtime/markets",
    ]
    
    for url in api_urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data:
                    logger.info(f"✅ Found data from {url}")
                    return data
        except:
            pass
            
    return None

def main():
    """Main function."""
    logger.info("🎯 Fetching Overtime V2 Markets WITHOUT API Key")
    logger.info("=" * 60)
    logger.info("Using blockchain analysis and public sources")
    
    total_added = 0
    
    # First try public endpoints
    public_data = check_public_endpoints()
    
    # Process each chain
    for chain_name, config in CHAINS.items():
        logger.info(f"\n🌐 Processing {config['name']}...")
        
        try:
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            if not w3.is_connected():
                logger.error(f"Failed to connect to {config['name']}")
                continue
                
            amm = Web3.to_checksum_address(config['amm'])
            
            # Method 1: Scan ALL AMM events
            logger.info("\n🔍 Scanning ALL AMM events for market addresses...")
            market_addrs = scan_all_amm_events(w3, amm, hours=12)
            
            if not market_addrs:
                # Method 2: Scan specific trade events
                logger.info("\n🔍 Scanning recent trades for markets...")
                market_addrs = scan_recent_trades(w3, amm, hours=48)
            
            if market_addrs:
                logger.info(f"Found {len(market_addrs)} markets from trades")
                
                for addr in market_addrs[:20]:  # Process up to 20
                    details = get_market_details(w3, addr)
                    
                    if not details or not details.get('homeTeam'):
                        continue
                        
                    # Skip resolved markets
                    if details.get('resolved', False):
                        continue
                        
                    # Skip past games
                    maturity = details.get('maturity', 0)
                    if maturity == 0:
                        continue
                        
                    maturity_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
                    if maturity_date < datetime.now(timezone.utc):
                        continue
                        
                    market_id = f"{chain_name}_blockchain_{addr[-8:]}"
                    
                    with db_manager.get_db_session() as db:
                        if db.query(Market).filter(Market.source_id == market_id).first():
                            continue
                            
                        # Sport mapping
                        sport_map = {
                            9001: "American Football",
                            9002: "Baseball",
                            9003: "Basketball",
                            9004: "Soccer",
                            9005: "Hockey",
                            9006: "MMA",
                            9007: "Boxing",
                            9008: "Tennis"
                        }
                        
                        sport_id = details.get('sportId', 9004)
                        
                        market = Market(
                            source_id=market_id,
                            source=f"{chain_name}_blockchain_no_api",
                            sport=sport_map.get(sport_id, "Soccer"),
                            league_name="Overtime V2",
                            market_type="winner",
                            home_team=details.get('homeTeam', 'Unknown'),
                            away_team=details.get('awayTeam', 'Unknown'),
                            maturity_date=maturity_date,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        
                        # Try to get odds
                        odds = get_odds_from_amm(w3, amm, addr)
                        
                        for position, decimal_odds in odds.items():
                            if 1.0 < decimal_odds < 100:
                                outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                                outcome = outcome_map.get(position)
                                
                                if outcome:
                                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
                                    
                                    odd = Odd(
                                        source_id=market_id,
                                        market_type="winner",
                                        outcome=outcome,
                                        source=f"{chain_name}_blockchain_no_api",
                                        bookmaker="Overtime V2",
                                        decimal_odds=decimal_odds,
                                        american_odds=american,
                                        normalized_implied=1.0 / decimal_odds,
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(odd)
                                    
                        db.commit()
                        total_added += 1
                        
                        logger.info(f"✅ Added: {details.get('gameLabel', 'Unknown')}")
                        logger.info(f"   Date: {maturity_date}")
                        
            else:
                logger.info("No markets found in recent trades")
                
        except Exception as e:
            logger.error(f"Error processing {chain_name}: {e}")
            
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
            
        logger.info(f"\n🎆 SUCCESS! Added {total_added} real blockchain markets!")
        logger.info("All data fetched directly from blockchain - NO API KEY REQUIRED!")
        logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
    else:
        logger.info("\nℹ️ No active markets found")
        logger.info("This could be due to:")
        logger.info("  • Low trading activity (try scanning more hours)")
        logger.info("  • Off-season period")
        logger.info("  • Markets already resolved")
        
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
        
        if active > 0:
            examples = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).order_by(Market.maturity_date).limit(5).all()
            
            logger.info("\n🏆 Active markets:")
            for m in examples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    {m.sport} - {m.maturity_date.strftime('%Y-%m-%d %H:%M UTC')}")

if __name__ == "__main__":
    main()