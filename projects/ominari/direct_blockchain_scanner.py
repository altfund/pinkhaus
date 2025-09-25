#!/usr/bin/env python3
"""
Direct blockchain scanner for Overtime markets
Using transaction analysis and known patterns
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
CONFIGS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'market_factory': '0x33b68A42584343F956e5FEFB6e1f0D56c8f5D4c5',  # Possible factory
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm_v2': '0xfb64E79A562F7250131cf528242CEB10fDC82395', 
        'market_factory': '0x2Aed181563D59e8e3D11ecdcfb9eeF9CA23FBe6e',  # Possible factory
        'name': 'Arbitrum'
    }
}

# Known method signatures for market creation
METHOD_SIGS = {
    'createMarket': '0x130e5939',  # createMarket(bytes32,uint256,uint256,uint256)
    'buyFromAMM': '0xe9331db5',    # buyFromAMM(address,uint8,uint256)
    'exerciseOptions': '0x8bced212'  # exerciseOptions()
}

def scan_recent_transactions(network):
    """Scan recent transactions to find market interactions."""
    config = CONFIGS[network]
    logger.info(f"\n🔍 Scanning {config['name']} transactions...")
    
    markets_found = set()
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        current_block = w3.eth.block_number
        logger.info(f"Connected at block {current_block:,}")
        
        # Scan last 100 blocks (more recent activity)
        for block_num in range(current_block - 100, current_block):
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block['transactions']:
                    # Check if it's to the AMM contract
                    if tx['to'] and tx['to'].lower() == config['sports_amm_v2'].lower():
                        # Check if it's a buyFromAMM transaction
                        if tx['input'][:10] == METHOD_SIGS['buyFromAMM']:
                            # Extract market address from input data
                            # buyFromAMM(address market, uint8 position, uint256 amount)
                            try:
                                # Market address is first parameter (after method sig)
                                market_addr = '0x' + tx['input'][34:74]
                                markets_found.add(Web3.to_checksum_address(market_addr))
                            except:
                                pass
                                
            except Exception as e:
                continue
                
        logger.info(f"Found {len(markets_found)} unique markets from transactions")
        
        # Process found markets
        markets_added = 0
        for market_addr in list(markets_found)[:20]:  # Process first 20
            try:
                # Get market details
                details = get_basic_market_info(w3, market_addr)
                if details and details.get('valid'):
                    market_id = f"blockchain_{network}_tx_{market_addr.lower()}"
                    
                    with db_manager.get_db_session() as db:
                        if not db.query(Market).filter(Market.source_id == market_id).first():
                            market = Market(
                                source_id=market_id,
                                source=f"blockchain_{network}_direct",
                                sport="Soccer",  # Default
                                league_name="Overtime Markets",
                                market_type="winner",
                                home_team=details.get('home', f"Team A {market_addr[-6:]}"),
                                away_team=details.get('away', f"Team B {market_addr[-6:]}"),
                                maturity_date=details.get('maturity', datetime.now(timezone.utc) + timedelta(days=1)),
                                is_finished=False,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(market)
                            db.commit()
                            
                            markets_added += 1
                            logger.info(f"✅ Added market from tx: {market_addr}")
                            
            except Exception as e:
                logger.debug(f"Error processing market {market_addr}: {e}")
                
        return markets_added
        
    except Exception as e:
        logger.error(f"Error scanning {network}: {e}")
        return 0

def get_basic_market_info(w3, market_address):
    """Try to get basic info from a market contract."""
    try:
        # Check if it's a contract
        code = w3.eth.get_code(market_address)
        if code == b'':
            return None
            
        # Try to read storage slots directly
        # Slot 0-10 often contain important data
        details = {'valid': True}
        
        # Try common storage patterns
        try:
            # Maturity date often in early slots
            slot_data = w3.eth.get_storage_at(market_address, 2)
            timestamp = int.from_bytes(slot_data, 'big')
            if 1600000000 < timestamp < 2000000000:  # Valid timestamp range
                details['maturity'] = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except:
            pass
            
        return details
        
    except Exception as e:
        return None

def find_markets_via_logs(network):
    """Find markets by looking for specific event patterns."""
    config = CONFIGS[network]
    logger.info(f"\n📋 Scanning {config['name']} event logs...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            return 0
            
        current_block = w3.eth.block_number
        
        # Look for OptionsBought events (indicates active markets)
        options_bought_sig = Web3.keccak(text="OptionsBought(address,address,uint256,uint8,uint256)").hex()
        
        # Scan recent blocks
        logs = w3.eth.get_logs({
            'fromBlock': current_block - 500,
            'toBlock': 'latest',
            'address': config['sports_amm_v2'],
            'topics': [options_bought_sig]
        })
        
        logger.info(f"Found {len(logs)} OptionsBought events")
        
        market_addresses = set()
        for log in logs:
            try:
                # Market address is second topic
                if len(log['topics']) > 2:
                    market_addr = '0x' + log['topics'][2].hex()[-40:]
                    market_addresses.add(Web3.to_checksum_address(market_addr))
            except:
                pass
                
        logger.info(f"Found {len(market_addresses)} unique markets from events")
        
        # Add markets
        for market_addr in list(market_addresses)[:10]:
            market_id = f"blockchain_{network}_event_{market_addr.lower()}"
            
            with db_manager.get_db_session() as db:
                if not db.query(Market).filter(Market.source_id == market_id).first():
                    # Create basic market entry
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_events",
                        sport="Soccer",
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=f"Home {market_addr[-6:]}",
                        away_team=f"Away {market_addr[-6:]}",
                        maturity_date=datetime.now(timezone.utc) + timedelta(days=2),
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"✅ Added market from event: {market_addr}")
                    
    except Exception as e:
        logger.error(f"Error scanning logs: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🔍 Direct Blockchain Scanner")
    logger.info("=" * 60)
    
    total_added = 0
    
    for network in ['optimism', 'arbitrum']:
        # Try transaction scanning
        added = scan_recent_transactions(network)
        total_added += added
        
        # Try event log scanning
        added = find_markets_via_logs(network)
        total_added += added
    
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        
        logger.info(f"\n✨ SCAN COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total > 0:
            samples = db.query(Market).limit(5).all()
            logger.info("\n📊 Sample markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    Source: {m.source}")

if __name__ == "__main__":
    main()