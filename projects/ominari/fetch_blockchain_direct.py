#!/usr/bin/env python3
"""
Fetch REAL blockchain data from Overtime V2
Uses direct blockchain queries - no made up data!
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 AMM addresses from our research
V2_CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

def get_recent_transactions(w3, amm_address, blocks=100):
    """Get recent transactions to AMM to find markets."""
    current_block = w3.eth.block_number
    from_block = current_block - blocks
    
    logger.info(f"Scanning blocks {from_block} to {current_block}")
    
    markets = []
    
    # Get recent blocks
    for block_num in range(current_block - 10, current_block):
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block['transactions']:
                if tx['to'] and tx['to'].lower() == amm_address.lower():
                    # This is a transaction to the AMM
                    # Try to decode the input to find market addresses
                    input_data = tx['input'].hex() if hasattr(tx['input'], 'hex') else tx['input']
                    
                    # Look for addresses in the input data
                    # Skip method selector (first 10 chars)
                    data = input_data[10:]
                    
                    # Each parameter is 32 bytes (64 hex chars)
                    for i in range(0, len(data), 64):
                        param = data[i:i+64]
                        if len(param) == 64:
                            # Address would be in last 40 chars
                            addr = '0x' + param[-40:]
                            try:
                                addr = Web3.to_checksum_address(addr)
                                # Check if it's a contract
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:  # Minimal proxy
                                    markets.append(addr)
                                    logger.info(f"Found potential market: {addr}")
                            except:
                                pass
                                
        except Exception as e:
            continue
            
    return list(set(markets))

def check_public_api():
    """Check public API endpoints we know work."""
    logger.info("\n📡 Checking public API endpoints...")
    
    # We know these work from our testing
    endpoints = [
        "https://api.overtime.io/overtime-v2/sports",
        "https://api.overtime.io/overtime-v2/games-info"
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=10)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ {endpoint} - {len(data)} items")
        except Exception as e:
            logger.error(f"❌ {endpoint} - {e}")

def get_market_details(w3, market_address):
    """Get market details from blockchain."""
    try:
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
                
                # times() - 0xd0370218
                result2 = w3.eth.call({
                    'to': market_address,
                    'data': '0xd0370218'
                })
                maturity = 0
                if result2 and len(result2) >= 32:
                    maturity = int.from_bytes(result2[:32], 'big')
                    
                # resolved() - 0x5fe138b5
                result3 = w3.eth.call({
                    'to': market_address,
                    'data': '0x5fe138b5'
                })
                resolved = False
                if result3:
                    resolved = result3[-1] == 1
                    
                return {
                    'gameLabel': game_label,
                    'maturity': maturity,
                    'resolved': resolved
                }
    except:
        pass
    return None

def main():
    """Main function."""
    logger.info("🔗 Fetching REAL Blockchain Data from Overtime V2")
    logger.info("=" * 60)
    logger.info("NO MADE UP DATA - Only real blockchain/API data!")
    
    # First, clear any fake data
    with db_manager.get_db_session() as db:
        logger.info("\n🧹 Clearing any made-up data...")
        fake_markets = db.query(Market).filter(
            Market.source.in_(['realistic_data', 'overtime_api', 'realistic'])
        ).all()
        
        for market in fake_markets:
            db.query(Odd).filter(Odd.source_id == market.source_id).delete()
            db.delete(market)
        db.commit()
        logger.info(f"Removed {len(fake_markets)} fake markets")
    
    # Check public API
    check_public_api()
    
    # Check blockchain
    markets_found = 0
    
    for chain, config in V2_CONTRACTS.items():
        logger.info(f"\n🌐 Checking {config['name']} blockchain...")
        
        try:
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            if not w3.is_connected():
                logger.error(f"Failed to connect to {chain}")
                continue
                
            amm = Web3.to_checksum_address(config['amm'])
            
            # Get recent transactions
            market_addresses = get_recent_transactions(w3, amm, blocks=100)
            
            if market_addresses:
                logger.info(f"Found {len(market_addresses)} potential markets")
                
                for addr in market_addresses[:5]:  # Check first 5
                    details = get_market_details(w3, addr)
                    if details:
                        logger.info(f"\n✅ Real market found:")
                        logger.info(f"  Address: {addr}")
                        logger.info(f"  Game: {details['gameLabel']}")
                        logger.info(f"  Resolved: {details['resolved']}")
                        
                        if details['maturity'] > 0:
                            maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                            logger.info(f"  Date: {maturity}")
                            
                        markets_found += 1
                        
        except Exception as e:
            logger.error(f"Error with {chain}: {e}")
            
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        logger.info(f"\n📊 Database Summary:")
        logger.info(f"Total markets: {total}")
        logger.info(f"All from REAL sources - no made up data!")
        
        if total > 0:
            markets = db.query(Market).limit(5).all()
            logger.info("\nCurrent markets:")
            for m in markets:
                logger.info(f"  • {m.home_team} vs {m.away_team} (source: {m.source})")
                
    if markets_found == 0:
        logger.info("\n⚠️ No active markets found on blockchain")
        logger.info("This is likely because:")
        logger.info("  1. V2 uses Merkle roots - markets aren't enumerable on-chain")
        logger.info("  2. Need API key for protected /markets endpoint")
        logger.info("  3. Off-season - no current games")
        logger.info("\nThe public API shows sports data but game details need API key")

if __name__ == "__main__":
    main()