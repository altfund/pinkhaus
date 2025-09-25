#!/usr/bin/env python3
"""
Remove placeholder data and fetch ONLY real blockchain market contracts
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blockchain configurations
CONFIGS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'chain_id': 10,
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm_v2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'chain_id': 42161,
        'name': 'Arbitrum'
    }
}

# Proper event signatures for Overtime Markets
MARKET_EVENTS = {
    # MarketCreated event
    'MarketCreated': 'MarketCreated(address,bytes32,string,uint256)',
    # GameCreated event  
    'GameCreated': 'GameCreated(bytes32,string,uint256,uint256,string,string)',
    # CreateSportsMarket event (older version)
    'CreateSportsMarket': 'CreateSportsMarket(address,bytes32,string,uint256)',
}

def clean_placeholder_data():
    """Remove all placeholder/known market data."""
    logger.info("🧹 Cleaning placeholder data...")
    
    with db_manager.get_db_session() as db:
        # Count before
        before_count = db.query(Market).count()
        
        # First, delete odds for placeholder markets
        odds_deleted = db.query(Odd).filter(
            Odd.source.like('%_known')
        ).delete()
        
        # Delete odds for non-blockchain markets
        odds_deleted += db.query(Odd).filter(
            ~Odd.source_id.like('%0x%')
        ).delete()
        
        logger.info(f"Deleted {odds_deleted} placeholder odds")
        
        # Now delete the markets
        deleted = db.query(Market).filter(
            Market.source.like('%_known')
        ).delete()
        
        # Also delete markets without proper blockchain addresses
        deleted += db.query(Market).filter(
            ~Market.source_id.like('%0x%')
        ).delete()
        
        db.commit()
        
        after_count = db.query(Market).count()
        logger.info(f"Deleted {deleted} placeholder markets")
        logger.info(f"Markets: {before_count} → {after_count}")

def get_market_created_events(w3, contract_address, from_block, to_block):
    """Get MarketCreated events from the blockchain."""
    events = []
    
    # Calculate event topic hashes
    event_sigs = {}
    for name, sig in MARKET_EVENTS.items():
        event_sigs[name] = Web3.keccak(text=sig).hex()
    
    logger.info(f"Scanning for events from block {from_block:,} to {to_block:,}")
    
    # Scan in smaller chunks to avoid timeouts
    chunk_size = 500
    for start in range(from_block, to_block + 1, chunk_size):
        end = min(start + chunk_size - 1, to_block)
        
        try:
            # Get logs for any of our known events
            logs = w3.eth.get_logs({
                'fromBlock': start,
                'toBlock': end,
                'address': contract_address,
                'topics': [[sig for sig in event_sigs.values()]]
            })
            
            if logs:
                logger.info(f"Found {len(logs)} events in blocks {start:,}-{end:,}")
                events.extend(logs)
                
        except Exception as e:
            logger.debug(f"Error scanning blocks {start}-{end}: {e}")
            
    return events

def decode_market_address(log):
    """Extract market contract address from event log."""
    # MarketCreated event has market address as first indexed parameter
    if len(log['topics']) > 1:
        # Remove 0x prefix and padding
        address_hex = log['topics'][1].hex()
        if len(address_hex) >= 40:
            return '0x' + address_hex[-40:]
    
    # Also check data field for non-indexed parameters
    if log['data'] and len(log['data']) >= 40:
        data_hex = log['data'].hex() if isinstance(log['data'], bytes) else log['data']
        # Market address might be in the data
        if len(data_hex) >= 42:  # 0x + 40 chars
            return Web3.to_checksum_address('0x' + data_hex[26:66])
            
    return None

def get_market_details_from_contract(w3, market_address):
    """Get market details by reading the market contract directly."""
    try:
        # Basic market contract ABI
        market_abi = [
            {
                "inputs": [],
                "name": "gameId",
                "outputs": [{"name": "", "type": "bytes32"}],
                "stateMutability": "view",
                "type": "function"
            },
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
                "name": "maturityDate",
                "outputs": [{"name": "", "type": "uint256"}],
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
                "name": "cancelled",
                "outputs": [{"name": "", "type": "bool"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
        contract = w3.eth.contract(address=Web3.to_checksum_address(market_address), abi=market_abi)
        
        # Try to read basic info
        details = {}
        
        try:
            details['gameId'] = contract.functions.gameId().call()
        except:
            pass
            
        try:
            details['homeTeam'] = contract.functions.homeTeam().call()
        except:
            pass
            
        try:
            details['awayTeam'] = contract.functions.awayTeam().call()
        except:
            pass
            
        try:
            details['maturityDate'] = contract.functions.maturityDate().call()
        except:
            pass
            
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        try:
            details['cancelled'] = contract.functions.cancelled().call()
        except:
            details['cancelled'] = False
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market contract {market_address}: {e}")
        return None

def fetch_real_blockchain_markets(network: str) -> int:
    """Fetch real market contracts from blockchain events."""
    config = CONFIGS[network]
    logger.info(f"\n🔗 Fetching REAL markets from {config['name']}...")
    
    markets_added = 0
    
    try:
        # Connect
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        current_block = w3.eth.block_number
        logger.info(f"Connected to {network} at block {current_block:,}")
        
        # Scan last 7 days of blocks
        blocks_per_day = 43200 if network == 'optimism' else 345600  # Approximate
        from_block = current_block - (blocks_per_day * 7)
        
        # Get market creation events
        events = get_market_created_events(
            w3, 
            config['sports_amm_v2'],
            from_block,
            current_block
        )
        
        logger.info(f"Found {len(events)} total events to process")
        
        # Process each event
        market_addresses = set()
        
        for event in events:
            market_address = decode_market_address(event)
            if market_address and market_address not in market_addresses:
                market_addresses.add(market_address)
                
        logger.info(f"Found {len(market_addresses)} unique market contracts")
        
        # Get details for each market
        for market_address in list(market_addresses)[:50]:  # Limit to 50 for now
            try:
                details = get_market_details_from_contract(w3, market_address)
                
                if not details or not details.get('homeTeam') or not details.get('awayTeam'):
                    continue
                    
                # Skip if resolved or cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                market_id = f"blockchain_{network}_contract_{market_address.lower()}"
                
                # Check if exists
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Create market
                    maturity = datetime.fromtimestamp(
                        details.get('maturityDate', 0), 
                        tz=timezone.utc
                    ) if details.get('maturityDate') else datetime.now(timezone.utc)
                    
                    # Skip past games
                    if maturity < datetime.now(timezone.utc):
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_real",
                        sport="Soccer",  # Default, could be determined from game ID
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
                    logger.info(f"✅ Added REAL market: {details['homeTeam']} vs {details['awayTeam']}")
                    logger.info(f"   Contract: {market_address}")
                    
            except Exception as e:
                logger.debug(f"Error processing market {market_address}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        
    return markets_added

def get_recent_market_transactions(network: str):
    """Alternative: Get markets from recent transactions to the AMM contract."""
    config = CONFIGS[network]
    logger.info(f"\n📝 Checking recent transactions on {network}...")
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            return
            
        # Get recent transactions to the AMM contract
        current_block = w3.eth.block_number
        
        # Check last 1000 blocks
        for block_num in range(current_block - 1000, current_block, 10):
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
                
                for tx in block['transactions']:
                    # Check if transaction is to our AMM contract
                    if tx['to'] and tx['to'].lower() == config['sports_amm_v2'].lower():
                        # This is a transaction to the AMM
                        # Could decode input data to find market creation calls
                        logger.debug(f"Found AMM transaction: {tx['hash'].hex()}")
                        
            except Exception as e:
                continue
                
    except Exception as e:
        logger.error(f"Error checking transactions: {e}")

def main():
    """Main function to clean and fetch real markets."""
    logger.info("🚀 Real Blockchain Market Fetcher")
    logger.info("=" * 60)
    
    # Step 1: Clean placeholder data
    clean_placeholder_data()
    
    # Step 2: Fetch real markets from each network
    total_added = 0
    
    for network in ['optimism', 'arbitrum']:
        added = fetch_real_blockchain_markets(network)
        total_added += added
        logger.info(f"Added {added} real markets from {network}")
        
        # Also try transaction analysis
        get_recent_market_transactions(network)
    
    # Show final stats
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        real_markets = db.query(Market).filter(
            Market.source.like('%_real')
        ).count()
        contract_markets = db.query(Market).filter(
            Market.source_id.like('%contract_%')
        ).count()
        
        logger.info(f"\n✨ REAL MARKET FETCH COMPLETE ✨")
        logger.info(f"Total markets: {total}")
        logger.info(f"Real blockchain markets: {real_markets}")
        logger.info(f"Contract-verified markets: {contract_markets}")
        logger.info(f"New markets added: {total_added}")
        
        # Show samples
        samples = db.query(Market).filter(
            Market.source.like('%_real')
        ).limit(5).all()
        
        if samples:
            logger.info("\n📊 Sample real markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    ID: {m.source_id}")
                logger.info(f"    Date: {m.maturity_date}")

if __name__ == "__main__":
    main()