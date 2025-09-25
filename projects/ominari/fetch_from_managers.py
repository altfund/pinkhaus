#!/usr/bin/env python3
"""
Fetch markets directly from the discovered manager contracts
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manager addresses found from AMM
MANAGERS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'manager': '0xB155685132eEd3cD848d220e25a9607DD8871D38',
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'manager': '0x2367FB44C4C2c4E5aAC62d78A55876E01F251605',
        'name': 'Optimism'
    }
}

# Try different manager ABI methods
MANAGER_ABI = [
    # Standard active markets array
    {
        "inputs": [{"name": "", "type": "uint256"}],
        "name": "activeMarkets",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Get all active markets
    {
        "inputs": [],
        "name": "getAllActiveMarkets",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Number of active markets
    {
        "inputs": [],
        "name": "numActiveMarkets",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Alternative naming
    {
        "inputs": [],
        "name": "numberOfActiveMarkets",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Paginated method
    {
        "inputs": [
            {"name": "index", "type": "uint256"},
            {"name": "pageSize", "type": "uint256"}
        ],
        "name": "activeMarkets",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # All markets (not just active)
    {
        "inputs": [{"name": "", "type": "uint256"}],
        "name": "allMarkets",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Markets by sport
    {
        "inputs": [{"name": "_sportId", "type": "uint256"}],
        "name": "getActiveMarketsForSport",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Market ABI
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
    }
]

def try_fetch_markets(w3, manager_address):
    """Try various methods to get markets from manager."""
    logger.info("🧪 Trying different manager methods...")
    
    contract = w3.eth.contract(address=Web3.to_checksum_address(manager_address), abi=MANAGER_ABI)
    markets = []
    
    # Try getAllActiveMarkets
    try:
        all_active = contract.functions.getAllActiveMarkets().call()
        if all_active:
            logger.info(f"✅ getAllActiveMarkets returned {len(all_active)} markets")
            markets.extend(all_active)
    except Exception as e:
        logger.debug(f"getAllActiveMarkets failed: {e}")
        
    # Try numActiveMarkets + individual access
    if not markets:
        try:
            num = contract.functions.numActiveMarkets().call()
            logger.info(f"numActiveMarkets: {num}")
            
            for i in range(min(10, num)):
                try:
                    market = contract.functions.activeMarkets(i).call()
                    if market != '0x0000000000000000000000000000000000000000':
                        markets.append(market)
                except:
                    pass
                    
            if markets:
                logger.info(f"✅ Got {len(markets)} markets via individual access")
        except:
            pass
            
    # Try numberOfActiveMarkets
    if not markets:
        try:
            num = contract.functions.numberOfActiveMarkets().call()
            logger.info(f"numberOfActiveMarkets: {num}")
        except:
            pass
            
    # Try paginated activeMarkets
    if not markets:
        try:
            page = contract.functions.activeMarkets(0, 50).call()
            if page:
                markets.extend(page)
                logger.info(f"✅ Got {len(page)} markets via paginated method")
        except:
            pass
            
    # Try markets by sport (soccer = 9004)
    if not markets:
        try:
            soccer_markets = contract.functions.getActiveMarketsForSport(9004).call()
            if soccer_markets:
                markets.extend(soccer_markets)
                logger.info(f"✅ Got {len(soccer_markets)} soccer markets")
        except:
            pass
            
    # Try allMarkets array
    if not markets:
        try:
            for i in range(10):
                market = contract.functions.allMarkets(i).call()
                if market != '0x0000000000000000000000000000000000000000':
                    markets.append(market)
        except:
            pass
            
    return [m for m in markets if m != '0x0000000000000000000000000000000000000000']

def get_market_details(w3, market_address):
    """Get basic details from a market."""
    try:
        market = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        home = market.functions.homeTeam().call()
        away = market.functions.awayTeam().call()
        times = market.functions.times().call()
        
        return {
            'homeTeam': home,
            'awayTeam': away,
            'maturity': times[0]
        }
    except:
        return None

def fetch_from_network(network):
    """Fetch markets from a network."""
    config = MANAGERS[network]
    logger.info(f"\n🌐 Processing {config['name']}...")
    logger.info(f"Manager: {config['manager']}")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        # Check manager contract
        code = w3.eth.get_code(config['manager'])
        logger.info(f"Manager code size: {len(code)} bytes")
        
        # Try to fetch markets
        markets = try_fetch_markets(w3, config['manager'])
        
        if not markets:
            logger.warning("No markets found from manager")
            return 0
            
        logger.info(f"Processing {len(markets)} market addresses...")
        
        for market_addr in markets[:20]:
            try:
                details = get_market_details(w3, market_addr)
                if not details:
                    continue
                    
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_manager_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_manager",
                        sport="Soccer",
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
                    
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                
    except Exception as e:
        logger.error(f"Error: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Fetch Markets from Discovered Managers")
    logger.info("=" * 60)
    
    total = 0
    for network in ['arbitrum', 'optimism']:
        added = fetch_from_network(network)
        total += added
        
    logger.info(f"\n✨ Total markets added: {total}")

if __name__ == "__main__":
    main()