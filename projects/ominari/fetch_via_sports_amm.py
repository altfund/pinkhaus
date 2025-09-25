#!/usr/bin/env python3
"""
Fetch real Overtime markets using the SportsAMM contract
Based on the user's documentation showing the correct approach
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

# From the user's documentation - verified SportsAMM addresses
SPORTS_AMM_ADDRESSES = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',  # From docs
        'explorer_api': 'https://api-optimistic.etherscan.io/api',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xd375572a9d6f6f464dd315d53053cf8183fb392e',  # From user's guide
        'explorer_api': 'https://api.arbiscan.io/api',
        'name': 'Arbitrum'
    }
}

# SportsAMM ABI - key methods for finding markets
SPORTS_AMM_ABI = [
    # Get market addresses by date and teams
    {
        "inputs": [
            {"name": "_sportId", "type": "uint256"},
            {"name": "_date", "type": "uint256"},
            {"name": "_homeTeam", "type": "string"},
            {"name": "_awayTeam", "type": "string"}
        ],
        "name": "getMarketAddressesByTagDateAndTeams",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Buy from AMM quote
    {
        "inputs": [
            {"name": "market", "type": "address"},
            {"name": "position", "type": "uint8"},
            {"name": "amount", "type": "uint256"}
        ],
        "name": "buyFromAmmQuote",
        "outputs": [{"name": "quote", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Manager address
    {
        "inputs": [],
        "name": "manager",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Game Market ABI - minimal interface
GAME_MARKET_ABI = [
    {"inputs": [], "name": "homeTeam", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "awayTeam", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "times", "outputs": [{"name": "", "type": "uint256"}, {"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "resolved", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "cancelled", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "tags", "outputs": [{"name": "", "type": "uint256[]"}], "stateMutability": "view", "type": "function"},
    {"inputs": [], "name": "getGameDetails", "outputs": [{"name": "", "type": "bytes32"}, {"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
    # For proxy contracts
    {"inputs": [], "name": "implementation", "outputs": [{"name": "", "type": "address"}], "stateMutability": "view", "type": "function"}
]

def scan_amm_events(w3, amm_address, from_block, to_block):
    """
    Scan AMM events to find market addresses.
    """
    markets = set()
    
    # BuyFromAMM event signature
    buy_sig = Web3.keccak(text="BuyFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
    # SellToAMM event signature
    sell_sig = Web3.keccak(text="SellToAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
    
    try:
        # Get buy events
        buy_logs = w3.eth.get_logs({
            'fromBlock': from_block,
            'toBlock': to_block,
            'address': amm_address,
            'topics': [buy_sig]
        })
        
        # Get sell events
        sell_logs = w3.eth.get_logs({
            'fromBlock': from_block,
            'toBlock': to_block,
            'address': amm_address,
            'topics': [sell_sig]
        })
        
        all_logs = buy_logs + sell_logs
        logger.info(f"  Found {len(all_logs)} AMM events in blocks {from_block}-{to_block}")
        
        for log in all_logs:
            # Market address is in topics[2]
            if len(log['topics']) > 2:
                market_addr = '0x' + log['topics'][2].hex()[-40:]
                markets.add(Web3.to_checksum_address(market_addr))
                
    except Exception as e:
        logger.debug(f"Error scanning events: {e}")
        
    return markets

def get_market_implementation(w3, proxy_address):
    """
    Get implementation address for a proxy contract.
    """
    # EIP-1967 implementation slot
    IMPL_SLOT = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
    
    try:
        impl_data = w3.eth.get_storage_at(proxy_address, IMPL_SLOT)
        if impl_data != b'\x00' * 32:
            impl_addr = '0x' + impl_data.hex()[-40:]
            return Web3.to_checksum_address(impl_addr)
    except:
        pass
        
    return None

def read_market_via_implementation(w3, proxy_address, impl_address):
    """
    Read market data by calling the implementation directly.
    """
    try:
        # Encode the call data for the proxy
        # Using homeTeam() selector: 0x8de859d8
        home_team_selector = '0x8de859d8'
        
        # Call the proxy with the selector
        result = w3.eth.call({
            'to': proxy_address,
            'data': home_team_selector
        })
        
        if result and len(result) > 0:
            # Decode string result
            # Skip offset (32 bytes) and length (32 bytes)
            if len(result) > 64:
                string_length = int.from_bytes(result[32:64], 'big')
                home_team = result[64:64+string_length].decode('utf-8').strip()
                return home_team
                
    except Exception as e:
        logger.debug(f"Error reading via implementation: {e}")
        
    return None

def get_market_details(w3, market_address):
    """
    Get details from a market contract (handling proxies).
    """
    try:
        # Check if it's a proxy (45 bytes)
        code = w3.eth.get_code(market_address)
        if len(code) == 45:
            logger.debug(f"  Proxy contract detected")
            impl = get_market_implementation(w3, market_address)
            if impl:
                logger.debug(f"  Implementation: {impl}")
                
        details = {'address': market_address}
        
        # Try to read teams directly
        contract = w3.eth.contract(address=market_address, abi=GAME_MARKET_ABI)
        
        # Try different approaches
        # 1. Direct call
        try:
            details['homeTeam'] = contract.functions.homeTeam().call()
            details['awayTeam'] = contract.functions.awayTeam().call()
        except:
            # 2. Try game details
            try:
                game_id, game_label = contract.functions.getGameDetails().call()
                if ' vs ' in game_label:
                    parts = game_label.split(' vs ')
                    details['homeTeam'] = parts[0].strip()
                    details['awayTeam'] = parts[1].strip()
            except:
                # 3. Try raw call with selector
                home = read_market_via_implementation(w3, market_address, None)
                if home:
                    details['homeTeam'] = home
                    # Try away team with selector 0x36c78516
                    try:
                        result = w3.eth.call({'to': market_address, 'data': '0x36c78516'})
                        if result and len(result) > 64:
                            string_length = int.from_bytes(result[32:64], 'big')
                            details['awayTeam'] = result[64:64+string_length].decode('utf-8').strip()
                    except:
                        pass
                        
        if not details.get('homeTeam'):
            return None
            
        # Get timing
        try:
            times = contract.functions.times().call()
            details['maturity'] = times[0]
        except:
            # Try raw call with times() selector: 0xd0370218
            try:
                result = w3.eth.call({'to': market_address, 'data': '0xd0370218'})
                if result and len(result) >= 64:
                    details['maturity'] = int.from_bytes(result[0:32], 'big')
            except:
                pass
                
        # Get status
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        # Get sport
        try:
            tags = contract.functions.tags().call()
            if tags:
                details['sportId'] = tags[0]
        except:
            details['sportId'] = 9004
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market: {e}")
        return None

def process_chain(chain):
    """
    Process markets from a specific chain using SportsAMM.
    """
    config = SPORTS_AMM_ADDRESSES[chain]
    logger.info(f"\n🌐 Processing {config['name']}...")
    logger.info(f"SportsAMM: {config['amm']}")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {chain}")
            return 0
            
        current_block = w3.eth.block_number
        logger.info(f"Connected at block {current_block:,}")
        
        # Scan recent blocks for market activity
        markets = set()
        
        # Scan in chunks
        chunk_size = 1000
        total_blocks = 5000  # Last 5000 blocks
        
        for i in range(0, total_blocks, chunk_size):
            from_block = current_block - total_blocks + i
            to_block = min(from_block + chunk_size - 1, current_block)
            
            chunk_markets = scan_amm_events(w3, config['amm'], from_block, to_block)
            markets.update(chunk_markets)
            
            if len(markets) >= 50:  # Enough markets found
                break
                
        logger.info(f"\n🎯 Found {len(markets)} unique markets from AMM activity")
        
        # Process each market
        for i, market_addr in enumerate(list(markets)[:50]):
            logger.info(f"\nProcessing market {i+1}: {market_addr}")
            
            try:
                details = get_market_details(w3, market_addr)
                
                if not details or not details.get('homeTeam'):
                    logger.warning("  No team data found")
                    continue
                    
                if details.get('resolved'):
                    logger.info("  Market resolved, skipping")
                    continue
                    
                if not details.get('maturity'):
                    logger.warning("  No maturity date")
                    continue
                    
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    logger.info("  Past game, skipping")
                    continue
                    
                market_id = f"blockchain_{chain}_amm_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        logger.info("  Already in database")
                        continue
                        
                    # Map sport ID
                    sport_map = {
                        9001: "American Football",
                        9002: "Baseball",
                        9003: "Basketball",
                        9004: "Soccer",
                        9005: "Hockey",
                        9006: "MMA",
                        9008: "Tennis"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{chain}_amm",
                        sport=sport_map.get(details.get('sportId', 9004), 'Soccer'),
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=details['homeTeam'],
                        away_team=details['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"  ✅ Added: {details['homeTeam']} vs {details['awayTeam']}")
                    logger.info(f"     Sport: {sport_map.get(details.get('sportId', 9004), 'Soccer')}")
                    logger.info(f"     Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing {chain}: {e}")
        
    return markets_added

def main():
    """
    Main function.
    """
    logger.info("🎯 Overtime Market Fetcher via SportsAMM")
    logger.info("=" * 60)
    logger.info("Using verified SportsAMM contracts from documentation")
    
    total_added = 0
    
    # Process each chain
    for chain in ['optimism', 'arbitrum']:
        added = process_chain(chain)
        total_added += added
        time.sleep(2)
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ SPORTS AMM FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")
                logger.info(f"    ID: {m.source_id}")

if __name__ == "__main__":
    main()