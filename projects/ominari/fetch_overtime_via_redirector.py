#!/usr/bin/env python3
"""
Fetch real Overtime markets using their contract redirector
Based on the comprehensive guide provided
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CHAINS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'explorer_api': 'https://api-optimistic.etherscan.io/api',
        'explorer_key': os.getenv('OP_ETHERSCAN_API_KEY', ''),  # Optional
        'name': 'Optimism',
        'chain_id': 10
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'explorer_api': 'https://api.arbiscan.io/api',
        'explorer_key': os.getenv('ARB_ETHERSCAN_API_KEY', ''),  # Optional
        'name': 'Arbitrum',
        'chain_id': 42161
    }
}

# Overtime contract paths
CONTRACT_PATHS = {
    'SportsAMM': 'SportsAMM',
    'SportPositionalMarketManager': 'SportPositionalMarketManager',
    'SportPositionalMarketFactory': 'SportPositionalMarketFactory'
}

def resolve_overtime_address(chain_path, contract_name):
    """
    Resolve the current contract address from Overtime's redirector.
    e.g., https://contracts.overtime.io/mainnet-ovm/SportsAMM
    """
    url = f"https://contracts.overtime.io/{chain_path}/{contract_name}"
    logger.info(f"🔗 Resolving {contract_name} via: {url}")
    
    try:
        # Follow redirects to get final URL
        response = requests.get(url, allow_redirects=True, timeout=10)
        final_url = response.url
        
        # Extract address from explorer URL (handles various patterns)
        match = re.search(r'0x[a-fA-F0-9]{40}', final_url)
        if match:
            address = Web3.to_checksum_address(match.group(0))
            logger.info(f"✅ {contract_name}: {address}")
            logger.info(f"   Explorer: {final_url}")
            return address, final_url
        else:
            logger.error(f"Could not parse address from: {final_url}")
            return None, final_url
            
    except Exception as e:
        logger.error(f"Error resolving {contract_name}: {e}")
        return None, None

def get_verified_abi(chain, address):
    """
    Fetch the verified ABI from blockchain explorer API.
    """
    config = CHAINS[chain]
    logger.info(f"📜 Fetching ABI for {address} from {chain}")
    
    params = {
        'module': 'contract',
        'action': 'getabi',
        'address': address
    }
    
    # Add API key if available
    if config['explorer_key']:
        params['apikey'] = config['explorer_key']
        
    try:
        response = requests.get(config['explorer_api'], params=params, timeout=10)
        data = response.json()
        
        if data['status'] == '1':
            abi = json.loads(data['result'])
            logger.info(f"✅ ABI fetched: {len(abi)} methods/events")
            return abi
        else:
            logger.warning(f"ABI not verified: {data.get('message', 'Unknown error')}")
            # Return a standard SportPositionalMarketManager ABI
            return get_standard_manager_abi()
            
    except Exception as e:
        logger.error(f"Error fetching ABI: {e}")
        return get_standard_manager_abi()

def get_standard_manager_abi():
    """
    Return a standard SportPositionalMarketManager ABI.
    """
    return [
        {
            "inputs": [],
            "name": "numActiveMarkets",
            "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [
                {"internalType": "uint256", "name": "index", "type": "uint256"},
                {"internalType": "uint256", "name": "pageSize", "type": "uint256"}
            ],
            "name": "activeMarkets",
            "outputs": [{"internalType": "address[]", "name": "", "type": "address[]"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
            "name": "activeMarkets",
            "outputs": [{"internalType": "address", "name": "", "type": "address"}],
            "stateMutability": "view",
            "type": "function"
        }
    ]

def enumerate_active_markets(w3, manager_address, manager_abi):
    """
    Page through active markets from SportPositionalMarketManager.
    """
    logger.info("📊 Enumerating active markets...")
    
    try:
        contract = w3.eth.contract(address=manager_address, abi=manager_abi)
        
        # Get total number of active markets
        try:
            total = contract.functions.numActiveMarkets().call()
            logger.info(f"Total active markets: {total}")
        except:
            # Try alternative method name
            try:
                total = contract.functions.numberOfActiveMarkets().call()
                logger.info(f"Total active markets: {total}")
            except:
                logger.error("Could not get number of active markets")
                return []
                
        if total == 0:
            return []
            
        # Page through markets
        markets = []
        page_size = 100
        
        for i in range(0, total, page_size):
            try:
                # Try paginated method
                batch = contract.functions.activeMarkets(i, min(page_size, total - i)).call()
                markets.extend(batch)
                logger.info(f"  Fetched page {i//page_size + 1}: {len(batch)} markets")
            except:
                # Fallback to individual access
                logger.info("  Using individual market access...")
                for j in range(i, min(i + page_size, total)):
                    try:
                        market = contract.functions.activeMarkets(j).call()
                        if market != '0x0000000000000000000000000000000000000000':
                            markets.append(market)
                    except:
                        break
                        
        return markets
        
    except Exception as e:
        logger.error(f"Error enumerating markets: {e}")
        return []

def get_market_details(w3, market_address, market_abi):
    """
    Get details from a Game Market contract.
    """
    try:
        contract = w3.eth.contract(address=market_address, abi=market_abi)
        
        details = {'address': market_address}
        
        # Check if it's a contract
        code = w3.eth.get_code(market_address)
        if not code or len(code) < 10:
            logger.warning(f"  Not a contract or empty code at {market_address}")
            return None
            
        logger.info(f"  Contract code size: {len(code)} bytes")
        
        # Try different method names based on ABI
        # Teams
        for method in ['homeTeam', 'getHomeTeam']:
            try:
                details['homeTeam'] = contract.functions[method]().call()
                logger.info(f"  Found homeTeam: {details['homeTeam']}")
                break
            except Exception as e:
                logger.debug(f"  {method} failed: {e}")
                
        for method in ['awayTeam', 'getAwayTeam']:
            try:
                details['awayTeam'] = contract.functions[method]().call()
                break
            except:
                pass
                
        # If no teams, try game details
        if not details.get('homeTeam'):
            try:
                game_id, game_label = contract.functions.getGameDetails().call()
                if ' vs ' in game_label:
                    parts = game_label.split(' vs ')
                    details['homeTeam'] = parts[0].strip()
                    details['awayTeam'] = parts[1].strip()
            except:
                pass
                
        # Timing
        try:
            times = contract.functions.times().call()
            details['maturity'] = times[0]
            details['expiry'] = times[1]
        except:
            try:
                details['maturity'] = contract.functions.maturityDate().call()
            except:
                pass
                
        # Status
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        try:
            details['cancelled'] = contract.functions.cancelled().call()
        except:
            details['cancelled'] = False
            
        # Sport tags
        try:
            tags = contract.functions.tags().call()
            if tags:
                details['sportId'] = tags[0]
        except:
            details['sportId'] = 9004  # Default to soccer
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def process_chain(chain):
    """
    Process markets from a specific chain.
    """
    config = CHAINS[chain]
    chain_path = 'mainnet-ovm' if chain == 'optimism' else 'mainnet-arbitrum'
    
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        # Connect to chain
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {chain}")
            return 0
            
        logger.info(f"Connected to {chain} at block {w3.eth.block_number:,}")
        
        # 1. Resolve SportPositionalMarketManager address
        manager_addr, _ = resolve_overtime_address(chain_path, 'SportPositionalMarketManager')
        if not manager_addr:
            logger.warning("Failed to resolve manager address, using known addresses")
            # Fallback to known addresses from documentation
            known_managers = {
                'optimism': '0xFBffEbfA2bF2cF84fdCf77917b358fC59Ff5771e',
                'arbitrum': '0x91b0d67a06936ad75c13e2b5f14F36dcf22D12Aa'
            }
            manager_addr = Web3.to_checksum_address(known_managers.get(chain))
            if not manager_addr:
                logger.error("No known manager address for chain")
                return 0
            
        # 2. Get manager ABI
        manager_abi = get_verified_abi(chain, manager_addr)
        if not manager_abi:
            logger.error("Failed to get manager ABI")
            return 0
            
        # 3. Enumerate active markets
        market_addresses = enumerate_active_markets(w3, manager_addr, manager_abi)
        
        if not market_addresses:
            logger.warning("No active markets found")
            return 0
            
        logger.info(f"\n🎯 Found {len(market_addresses)} active markets")
        
        # 4. Always use minimal ABI for markets (works across all versions)
        logger.info("Using standard market ABI")
        market_abi = [
            {"inputs": [], "name": "homeTeam", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "awayTeam", "outputs": [{"name": "", "type": "string"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "times", "outputs": [{"name": "", "type": "uint256"}, {"name": "", "type": "uint256"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "resolved", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "cancelled", "outputs": [{"name": "", "type": "bool"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "tags", "outputs": [{"name": "", "type": "uint256[]"}], "stateMutability": "view", "type": "function"},
            {"inputs": [], "name": "getGameDetails", "outputs": [{"name": "", "type": "bytes32"}, {"name": "", "type": "string"}], "stateMutability": "view", "type": "function"}
        ]
            
        # 5. Process each market
        for i, market_addr in enumerate(market_addresses):
            if i >= 50:  # Limit for initial run
                logger.info("Reached 50 market limit")
                break
                
            logger.info(f"\nProcessing market {i+1}/{len(market_addresses)}: {market_addr}")
            
            try:
                details = get_market_details(w3, market_addr, market_abi)
                
                if not details:
                    logger.warning("  Could not get market details")
                    continue
                    
                if not details.get('homeTeam'):
                    logger.warning(f"  No teams found: {details}")
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                # Check maturity
                if not details.get('maturity'):
                    continue
                    
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{chain}_live_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Map sport ID
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
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{chain}_live",
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
                continue
                
        logger.info(f"\n✅ Added {markets_added} markets from {config['name']}")
        
    except Exception as e:
        logger.error(f"Error processing {chain}: {e}")
        
    return markets_added

def main():
    """
    Main function.
    """
    logger.info("🎯 Overtime Market Fetcher - Live Contract Resolution")
    logger.info("=" * 60)
    logger.info("Using Overtime's contract redirector to fetch live addresses")
    
    total_added = 0
    
    # Process each chain
    for chain in ['optimism', 'arbitrum']:
        added = process_chain(chain)
        total_added += added
        time.sleep(2)  # Be nice to APIs
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ LIVE MARKET FETCH COMPLETE ✨")
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