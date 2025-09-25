#!/usr/bin/env python3
"""
Fetch Overtime markets using factory contract pattern
Based on the documentation about dynamic market creation
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known Overtime/Thales factory contracts from blockchain analysis
FACTORY_CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_factory': '0x3e80fA93428c8a2a74E02bcaB94DC1201eE70E6E',  # SportsAMMFactory
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'explorer': 'https://optimistic.etherscan.io',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_factory': '0xfDaB6B7623Ba7e7Cb8A52f95cD5Cc0F62B1e8D40',  # SportsAMMFactory
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'explorer': 'https://arbiscan.io',
        'name': 'Arbitrum'
    }
}

# Event signatures for market creation
MARKET_CREATION_EVENTS = {
    # SportPositionalMarketCreated event
    'SportPositionalMarketCreated': 'SportPositionalMarketCreated(address,bytes32,string,uint256,uint256,address,address,address,address)',
    # MarketCreated (simplified)
    'MarketCreated': 'MarketCreated(address,bytes32,string,uint256)',
    # NewSportsMarket
    'NewSportsMarket': 'NewSportsMarket(address,string,string,uint256)'
}

# Basic market ABI for reading data
MARKET_ABI = [
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getGameId",
        "outputs": [{"internalType": "bytes32", "name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "times",
        "outputs": [
            {"internalType": "uint256", "name": "maturity", "type": "uint256"},
            {"internalType": "uint256", "name": "destruction", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getGameDetails",
        "outputs": [
            {"internalType": "bytes32", "name": "gameId", "type": "bytes32"},
            {"internalType": "string", "name": "gameLabel", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "teams",
        "outputs": [
            {"internalType": "string", "name": "", "type": "string"},
            {"internalType": "string", "name": "", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "resolved",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "cancelled",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def get_verified_abi_from_explorer(network, contract_address):
    """Try to get verified ABI from blockchain explorer."""
    config = FACTORY_CONTRACTS[network]
    
    # Arbiscan/Optimistic Etherscan API
    if network == 'arbitrum':
        api_url = f"https://api.arbiscan.io/api?module=contract&action=getabi&address={contract_address}"
    else:
        api_url = f"https://api-optimistic.etherscan.io/api?module=contract&action=getabi&address={contract_address}"
    
    try:
        response = requests.get(api_url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1':
                return json.loads(data['result'])
    except:
        pass
    
    return None

def scan_factory_events(w3, factory_address, network):
    """Scan factory contract for market creation events."""
    logger.info(f"📡 Scanning factory events on {network}...")
    
    market_addresses = []
    
    # Calculate event signatures
    event_topics = []
    for event_name, sig in MARKET_CREATION_EVENTS.items():
        topic = Web3.keccak(text=sig).hex()
        event_topics.append(topic)
        logger.debug(f"{event_name}: {topic}")
    
    current_block = w3.eth.block_number
    
    # Scan recent blocks (last 24 hours approximately)
    blocks_per_hour = 1800 if network == 'optimism' else 14400  # Rough estimates
    from_block = current_block - (blocks_per_hour * 24)
    
    logger.info(f"Scanning from block {from_block:,} to {current_block:,}")
    
    # Scan in chunks
    chunk_size = 1000
    for start in range(from_block, current_block + 1, chunk_size):
        end = min(start + chunk_size - 1, current_block)
        
        try:
            # Get logs from factory
            logs = w3.eth.get_logs({
                'fromBlock': start,
                'toBlock': end,
                'address': factory_address
            })
            
            for log in logs:
                # Check if it's a market creation event
                if log['topics'] and log['topics'][0].hex() in event_topics:
                    # Extract market address (usually first indexed parameter)
                    if len(log['topics']) > 1:
                        market_addr = '0x' + log['topics'][1].hex()[-40:]
                        try:
                            market_addr = Web3.to_checksum_address(market_addr)
                            market_addresses.append(market_addr)
                            logger.info(f"Found market: {market_addr}")
                        except:
                            pass
                            
        except Exception as e:
            logger.debug(f"Error scanning blocks {start}-{end}: {e}")
            
    return market_addresses

def get_market_details_enhanced(w3, market_address):
    """Get detailed market information."""
    try:
        # First check if it's a contract
        code = w3.eth.get_code(market_address)
        if not code or code == b'':
            return None
            
        contract = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        details = {
            'address': market_address,
            'valid': True
        }
        
        # Try various methods
        try:
            teams = contract.functions.teams().call()
            details['homeTeam'] = teams[0]
            details['awayTeam'] = teams[1]
        except:
            # Try alternative method
            try:
                game_details = contract.functions.getGameDetails().call()
                # Parse teams from game label
                if game_details[1]:
                    parts = game_details[1].split(' vs ')
                    if len(parts) == 2:
                        details['homeTeam'] = parts[0]
                        details['awayTeam'] = parts[1]
            except:
                pass
                
        # Get timing
        try:
            times = contract.functions.times().call()
            details['maturity'] = times[0]
        except:
            pass
            
        # Check status
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        try:
            details['cancelled'] = contract.functions.cancelled().call()
        except:
            details['cancelled'] = False
            
        # Get tags (sport type)
        try:
            tags = contract.functions.tags().call()
            if tags and len(tags) > 0:
                # Tag 9004 = Soccer, 9001 = American Football, etc.
                sport_tag = tags[0] if tags else 9004
                details['sport'] = get_sport_from_tag(sport_tag)
        except:
            details['sport'] = 'Soccer'
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def get_sport_from_tag(tag):
    """Convert sport tag to name."""
    sport_map = {
        9001: "American Football",
        9002: "Baseball",
        9003: "Basketball", 
        9004: "Soccer",
        9005: "Hockey",
        9006: "MMA",
        9007: "Boxing",
        9008: "Tennis",
        9010: "Golf",
        9011: "Cricket",
        9012: "Rugby",
        9014: "Motorsport"
    }
    return sport_map.get(tag, "Soccer")

def fetch_from_network(network):
    """Fetch markets from a specific network."""
    config = FACTORY_CONTRACTS[network]
    logger.info(f"\n🏗️ Fetching from {config['name']} factory...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # Scan factory for market creation events
        market_addresses = scan_factory_events(w3, config['sports_factory'], network)
        
        # Also scan AMM contract for market interactions
        amm_markets = scan_factory_events(w3, config['sports_amm'], network)
        market_addresses.extend(amm_markets)
        
        # Remove duplicates
        market_addresses = list(set(market_addresses))
        logger.info(f"Found {len(market_addresses)} unique market addresses")
        
        # Process markets
        for market_addr in market_addresses[:30]:  # Limit to 30
            try:
                details = get_market_details_enhanced(w3, market_addr)
                
                if not details or not details.get('homeTeam'):
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                # Skip past games
                maturity = datetime.fromtimestamp(
                    details.get('maturity', 0),
                    tz=timezone.utc
                )
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_factory_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_factory",
                        sport=details.get('sport', 'Soccer'),
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
                    logger.info(f"✅ Added: {details['homeTeam']} vs {details['awayTeam']} ({details.get('sport')})")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🏭 Overtime Factory Markets Fetcher")
    logger.info("=" * 60)
    logger.info("Using factory pattern to find dynamically created markets")
    
    total_added = 0
    
    for network in ['optimism', 'arbitrum']:
        added = fetch_from_network(network)
        total_added += added
        logger.info(f"Added {added} markets from {network}")
    
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ FACTORY FETCH COMPLETE ✨")
        logger.info(f"Total markets: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"New markets added: {total_added}")
        
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")
                logger.info(f"    ID: {m.source_id}")

if __name__ == "__main__":
    main()