#!/usr/bin/env python3
"""
Fetch Overtime markets using direct contract relationships
No event scanning - using known contract methods and relationships
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

# Known contract addresses and relationships
OVERTIME_CONTRACTS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'sports_manager': '0x2dA220f7C6AC27a575DcF3abD0C87C548f756759',
        'market_data': '0x5A17bD71b14236cF91B38342fF45a49Dd3b53fD0',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'sports_manager': '0x8b0518cBdfaC874fbEe6F5dfC5DDF8f71b5D6A28', 
        'market_data': '0x6ACA3f965A8f5A52e6eB088c32e7e9e03cC6A3D6',
        'name': 'Arbitrum'
    }
}

# ABI for SportsAMM methods that return market lists
SPORTS_AMM_ABI = [
    # Get active markets between dates
    {
        "inputs": [
            {"name": "index", "type": "uint256"},
            {"name": "pageSize", "type": "uint256"}
        ],
        "name": "getActiveMarketsPaginated",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Get market addresses by sport
    {
        "inputs": [
            {"name": "sportId", "type": "uint256"},
            {"name": "offset", "type": "uint256"},
            {"name": "numberOfMarkets", "type": "uint256"}
        ],
        "name": "getMarketsBySport",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Get total number of markets
    {
        "inputs": [],
        "name": "numActiveMarkets",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Alternative method names
    {
        "inputs": [],
        "name": "activeMarketsPerSport",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Manager contract ABI for market data
MANAGER_ABI = [
    {
        "inputs": [{"name": "_sportId", "type": "uint256"}],
        "name": "getActiveMarketsForSport",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getAllActiveMarkets",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Market contract ABI
POSITIONAL_MARKET_ABI = [
    {
        "inputs": [],
        "name": "creator",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "options",
        "outputs": [
            {"name": "", "type": "address"},
            {"name": "", "type": "address"},
            {"name": "", "type": "address"}
        ],
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
        "name": "initialMint",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "isChild",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def get_markets_from_amm(w3, amm_address):
    """Get market addresses using AMM contract methods."""
    logger.info("📊 Querying AMM for active markets...")
    
    markets = []
    
    try:
        contract = w3.eth.contract(address=Web3.to_checksum_address(amm_address), abi=SPORTS_AMM_ABI)
        
        # Try different methods
        
        # Method 1: Get total number and paginate
        try:
            num_markets = contract.functions.numActiveMarkets().call()
            logger.info(f"AMM reports {num_markets} active markets")
            
            if num_markets > 0:
                # Get markets in pages
                page_size = 20
                for i in range(0, min(num_markets, 100), page_size):
                    try:
                        page_markets = contract.functions.getActiveMarketsPaginated(i, page_size).call()
                        markets.extend(page_markets)
                    except:
                        pass
                        
        except Exception as e:
            logger.debug(f"numActiveMarkets failed: {e}")
            
        # Method 2: Get markets by sport
        if not markets:
            # Try common sport IDs
            sport_ids = [9004, 9001, 9003, 9005]  # Soccer, Football, Basketball, Hockey
            for sport_id in sport_ids:
                try:
                    sport_markets = contract.functions.getMarketsBySport(sport_id, 0, 20).call()
                    if sport_markets:
                        logger.info(f"Found {len(sport_markets)} markets for sport {sport_id}")
                        markets.extend(sport_markets)
                except:
                    pass
                    
        # Method 3: Try manager contract
        if not markets:
            logger.info("Trying manager contract...")
            
    except Exception as e:
        logger.error(f"Error querying AMM: {e}")
        
    return markets

def get_markets_from_manager(w3, manager_address):
    """Get markets from the sports manager contract."""
    logger.info("📋 Querying manager contract...")
    
    markets = []
    
    try:
        contract = w3.eth.contract(address=Web3.to_checksum_address(manager_address), abi=MANAGER_ABI)
        
        # Try to get all active markets
        try:
            all_markets = contract.functions.getAllActiveMarkets().call()
            logger.info(f"Manager returned {len(all_markets)} markets")
            markets.extend(all_markets)
        except:
            # Try by sport
            for sport_id in [9004, 9001, 9003]:
                try:
                    sport_markets = contract.functions.getActiveMarketsForSport(sport_id).call()
                    markets.extend(sport_markets)
                except:
                    pass
                    
    except Exception as e:
        logger.debug(f"Manager query failed: {e}")
        
    return markets

def validate_market_contract(w3, market_address):
    """Check if address is a valid market contract."""
    try:
        # Check if it's a contract
        code = w3.eth.get_code(market_address)
        if not code or len(code) < 100:
            return False
            
        # Try to read basic properties
        contract = w3.eth.contract(address=market_address, abi=POSITIONAL_MARKET_ABI)
        
        # Check if it has expected methods
        try:
            # Valid market should have options
            options = contract.functions.options().call()
            if options[0] == '0x0000000000000000000000000000000000000000':
                return False
                
            # Should have times
            times = contract.functions.times().call()
            if times[0] == 0:
                return False
                
            return True
            
        except:
            return False
            
    except:
        return False

def get_market_info(w3, market_address):
    """Extract market information using storage reads."""
    try:
        # Read storage slots directly for efficiency
        # Slot layout varies but common patterns exist
        
        info = {
            'address': market_address,
            'valid': True
        }
        
        # Slot 0-2: Often contains game ID, creator
        # Slot 3-5: Teams or game label
        # Slot 6-8: Timing info
        # Slot 9-11: Options/positions
        
        # Try to decode team names from storage
        for slot in range(3, 8):
            try:
                data = w3.eth.get_storage_at(market_address, slot)
                if data != b'\x00' * 32:
                    # Try to decode as string
                    try:
                        text = data.decode('utf-8').strip('\x00')
                        if len(text) > 2 and text.isprintable():
                            if 'homeTeam' not in info:
                                info['homeTeam'] = text
                            elif 'awayTeam' not in info:
                                info['awayTeam'] = text
                    except:
                        pass
            except:
                pass
                
        # Get timing from contract call
        try:
            contract = w3.eth.contract(address=market_address, abi=POSITIONAL_MARKET_ABI)
            times = contract.functions.times().call()
            info['maturity'] = times[0]
            
            # Get sport from tags
            tags = contract.functions.tags().call()
            if tags:
                info['sport_id'] = tags[0]
                
        except:
            pass
            
        # Fallback team names
        if 'homeTeam' not in info:
            info['homeTeam'] = f"Team {market_address[2:8]}"
        if 'awayTeam' not in info:
            info['awayTeam'] = f"Team {market_address[-8:]}"
            
        return info
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def fetch_network_markets(network):
    """Fetch markets from a specific network."""
    config = OVERTIME_CONTRACTS[network]
    logger.info(f"\n🌐 Fetching from {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # Get markets from AMM
        amm_markets = get_markets_from_amm(w3, config['sports_amm'])
        
        # Get markets from manager
        manager_markets = get_markets_from_manager(w3, config['sports_manager'])
        
        # Combine and deduplicate
        all_markets = list(set(amm_markets + manager_markets))
        
        # Validate markets
        valid_markets = []
        for addr in all_markets:
            if validate_market_contract(w3, addr):
                valid_markets.append(addr)
                
        logger.info(f"Found {len(valid_markets)} valid market contracts")
        
        # Process markets
        for market_addr in valid_markets[:20]:  # Limit to 20
            try:
                info = get_market_info(w3, market_addr)
                
                if not info:
                    continue
                    
                # Skip past games
                maturity = datetime.fromtimestamp(
                    info.get('maturity', 0),
                    tz=timezone.utc
                )
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_direct_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    sport_map = {
                        9001: "American Football",
                        9003: "Basketball",
                        9004: "Soccer",
                        9005: "Hockey"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_direct",
                        sport=sport_map.get(info.get('sport_id', 9004), 'Soccer'),
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
                    logger.info(f"   Address: {market_addr}")
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Direct Contract Relationship Fetcher")
    logger.info("=" * 60)
    logger.info("Using known contract methods - no event scanning")
    
    total_added = 0
    
    for network in ['optimism', 'arbitrum']:
        added = fetch_network_markets(network)
        total_added += added
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        
        logger.info(f"\n✨ DIRECT FETCH COMPLETE ✨")
        logger.info(f"Total markets: {total}")
        logger.info(f"Markets added: {total_added}")
        
        if total > 0:
            samples = db.query(Market).limit(5).all()
            logger.info("\n📊 Sample markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")

if __name__ == "__main__":
    main()