#!/usr/bin/env python3
"""
Fetch markets using the correct verified addresses from Overtime documentation
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

# From Overtime docs - these are the correct contracts
OVERTIME_CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'rundown_consumer': '0x56c85448A9d0f0CcE0c31C6a50FC93D7a6bFe39F',
        'manager': '0x8a8ab60E1A0797F8bbfc892B21ddC3F8C4C60Ad3',  # SportPositionalMarketManager
        'sports_amm': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'rundown_consumer': '0xE3D0c5dBBb604fDF6a48604e2dc5fA9B9383A8Df',
        'manager': '0x74c5a7Db1Cc993EB27Ef59cB6f482aD159101ad9',  # SportPositionalMarketManager
        'sports_amm': '0xd375572a9d6f6f464dd315d53053cf8183fb392e',
        'name': 'Arbitrum'
    }
}

def check_contract_on_explorer(address, network):
    """Check if contract is verified on explorer."""
    if network == 'arbitrum':
        url = f"https://api.arbiscan.io/api?module=contract&action=getsourcecode&address={address}"
    else:
        url = f"https://api-optimistic.etherscan.io/api?module=contract&action=getsourcecode&address={address}"
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result'][0]['ContractName']:
                return data['result'][0]['ContractName']
    except:
        pass
    return None

def scan_for_market_creation_events(w3, consumer_address, network):
    """Scan for MarketCreated events from TheRundownConsumer."""
    logger.info("🔍 Scanning for MarketCreated events...")
    
    markets = []
    
    try:
        # MarketCreated(address _marketAddress, bytes32 _id, string _gameLabel, uint256 _maturityDate)
        event_sig = Web3.keccak(text="MarketCreated(address,bytes32,string,uint256)").hex()
        
        current_block = w3.eth.block_number
        
        # Scan in chunks to avoid rate limits
        chunk_size = 1000
        for i in range(5):  # Last 5000 blocks
            from_block = current_block - ((i + 1) * chunk_size)
            to_block = current_block - (i * chunk_size)
            
            try:
                logs = w3.eth.get_logs({
                    'fromBlock': from_block,
                    'toBlock': to_block,
                    'address': consumer_address,
                    'topics': [event_sig]
                })
                
                for log in logs:
                    # Market address is first indexed parameter
                    if len(log['topics']) > 1:
                        market_addr = '0x' + log['topics'][1].hex()[-40:]
                        markets.append(Web3.to_checksum_address(market_addr))
                        
                if logs:
                    logger.info(f"  Found {len(logs)} events in blocks {from_block}-{to_block}")
                    
            except Exception as e:
                logger.debug(f"Error scanning chunk: {e}")
                
    except Exception as e:
        logger.error(f"Error scanning events: {e}")
        
    return list(set(markets))

def get_market_info(w3, market_address):
    """Get info from a market contract."""
    # Basic ABI for market info
    abi = [
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
        }
    ]
    
    try:
        contract = w3.eth.contract(address=market_address, abi=abi)
        
        home = contract.functions.homeTeam().call()
        away = contract.functions.awayTeam().call()
        times = contract.functions.times().call()
        resolved = contract.functions.resolved().call()
        
        tags = []
        try:
            tags = contract.functions.tags().call()
        except:
            pass
            
        return {
            'homeTeam': home,
            'awayTeam': away,
            'maturity': times[0],
            'resolved': resolved,
            'sportId': tags[0] if tags else 9004
        }
    except:
        return None

def fetch_from_network(network):
    """Fetch markets from a network."""
    config = OVERTIME_CONTRACTS[network]
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # Check if RundownConsumer is verified
        contract_name = check_contract_on_explorer(config['rundown_consumer'], network)
        if contract_name:
            logger.info(f"✅ RundownConsumer verified as: {contract_name}")
        else:
            logger.warning("RundownConsumer not verified")
            
        # Scan for market creation events
        market_addresses = scan_for_market_creation_events(w3, config['rundown_consumer'], network)
        
        logger.info(f"Found {len(market_addresses)} markets from events")
        
        # Process markets
        for i, market_addr in enumerate(market_addresses):
            if i >= 20:  # Limit for testing
                break
                
            try:
                info = get_market_info(w3, market_addr)
                if not info or info['resolved']:
                    continue
                    
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_overtime_{market_addr.lower()}"
                
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
                        9006: "MMA"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_overtime",
                        sport=sport_map.get(info['sportId'], 'Soccer'),
                        league_name="Overtime Markets",
                        market_type="winner",
                        home_team=info['homeTeam'],
                        away_team=info['awayTeam'],
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"✅ Added: {info['homeTeam']} vs {info['awayTeam']}")
                    logger.info(f"   Contract: {market_addr}")
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Overtime Market Fetcher - Correct Addresses")
    logger.info("=" * 60)
    
    total = 0
    for network in ['optimism', 'arbitrum']:
        added = fetch_from_network(network)
        total += added
        
    # Summary
    with db_manager.get_db_session() as db:
        count = db.query(Market).count()
        logger.info(f"\n✨ Total markets in database: {count}")
        logger.info(f"Markets added this run: {total}")

if __name__ == "__main__":
    main()