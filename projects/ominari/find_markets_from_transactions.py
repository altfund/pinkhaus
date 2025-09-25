#!/usr/bin/env python3
"""
Find real Overtime markets by analyzing recent SportsAMM transactions
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SportsAMM contracts we know are active
AMM_CONTRACTS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'name': 'Optimism'
    }
}

# Common AMM method signatures
METHOD_SIGNATURES = {
    'buyFromAMM': '0x942b67dc',  # buyFromAMM(address,uint8,uint256)
    'sellToAMM': '0x8cc7bcc5',   # sellToAMM(address,uint8,uint256)
    'buyFromAMMWithDifferentCollateral': '0x1ed66ad7',
    'sellToAMMWithDifferentCollateral': '0x4917dc72'
}

# Market ABI for reading details
MARKET_ABI = [
    {
        "inputs": [],
        "name": "homeTeam",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "awayTeam",
        "outputs": [{"name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "times",
        "outputs": [
            {"name": "", "type": "uint256"},
            {"name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "resolved",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Alternative method names
    {
        "inputs": [],
        "name": "getGameDetails",
        "outputs": [
            {"name": "", "type": "bytes32"},
            {"name": "", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

def find_markets_in_recent_blocks(w3, amm_address):
    """Find market addresses from recent AMM transactions."""
    logger.info("🔍 Analyzing recent AMM transactions...")
    
    markets = set()
    current_block = w3.eth.block_number
    
    # Check last 100 blocks
    for block_num in range(current_block - 100, current_block):
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block['transactions']:
                # Check if transaction is to AMM
                if tx['to'] and tx['to'].lower() == amm_address.lower():
                    input_data = tx['input']
                    
                    # Check if it's one of our known methods
                    for method_name, sig in METHOD_SIGNATURES.items():
                        if input_data.startswith(sig):
                            # Extract market address (first parameter)
                            try:
                                # Skip method signature (4 bytes = 8 hex + 0x = 10)
                                data = input_data[10:]
                                if len(data) >= 64:
                                    # First 32 bytes is the address (padded)
                                    addr_hex = '0x' + data[24:64]
                                    market_addr = Web3.to_checksum_address(addr_hex)
                                    
                                    # Verify it's a contract
                                    code = w3.eth.get_code(market_addr)
                                    if code and len(code) > 100:
                                        markets.add(market_addr)
                                        logger.debug(f"Found market from {method_name}: {market_addr}")
                            except Exception as e:
                                logger.debug(f"Error parsing tx: {e}")
                                
        except Exception as e:
            logger.debug(f"Error processing block {block_num}: {e}")
            
        # Show progress
        if (current_block - block_num) % 20 == 0:
            logger.info(f"  Scanned {current_block - block_num}/100 blocks, found {len(markets)} markets")
            
    return list(markets)

def get_market_details(w3, market_address):
    """Get details from a market contract."""
    try:
        contract = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        details = {'address': market_address}
        
        # Try standard methods
        try:
            details['homeTeam'] = contract.functions.homeTeam().call()
            details['awayTeam'] = contract.functions.awayTeam().call()
        except:
            # Try alternative method
            try:
                game_id, game_label = contract.functions.getGameDetails().call()
                if ' vs ' in game_label:
                    parts = game_label.split(' vs ')
                    details['homeTeam'] = parts[0].strip()
                    details['awayTeam'] = parts[1].strip()
                else:
                    return None
            except:
                return None
                
        # Get timing
        times = contract.functions.times().call()
        details['maturity'] = times[0]
        
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
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def process_network(network):
    """Process markets from a specific network."""
    config = AMM_CONTRACTS[network]
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network} at block {w3.eth.block_number:,}")
        
        # Find markets from recent transactions
        market_addresses = find_markets_in_recent_blocks(w3, config['amm'])
        
        logger.info(f"\n🎯 Found {len(market_addresses)} unique markets")
        
        # Process each market
        for i, market_addr in enumerate(market_addresses):
            if i >= 30:  # Limit
                break
                
            try:
                details = get_market_details(w3, market_addr)
                
                if not details or not details.get('homeTeam'):
                    continue
                    
                # Skip resolved
                if details.get('resolved'):
                    continue
                    
                # Check maturity
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_tx_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Map sport
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
                        source=f"blockchain_{network}_tx",
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
                    logger.info(f"✅ Added: {details['homeTeam']} vs {details['awayTeam']}")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Sport: {sport_map.get(details.get('sportId', 9004), 'Soccer')}")
                    logger.info(f"   Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                
    except Exception as e:
        logger.error(f"Error processing {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Overtime Market Discovery via Transactions")
    logger.info("=" * 60)
    logger.info("Finding real markets from SportsAMM transaction history")
    
    total = 0
    
    # Process each network
    for network in ['arbitrum', 'optimism']:
        added = process_network(network)
        total += added
        time.sleep(2)
        
    # Summary
    with db_manager.get_db_session() as db:
        count = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ TRANSACTION DISCOVERY COMPLETE ✨")
        logger.info(f"Total markets in database: {count}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total}")
        
        if count > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")

if __name__ == "__main__":
    main()