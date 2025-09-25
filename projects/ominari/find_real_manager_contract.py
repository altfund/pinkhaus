#!/usr/bin/env python3
"""
Find the real SportPositionalMarketManager by examining relationships
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

NETWORKS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'amm_implementation': '0xd375572a9d6f6f464dd315d53053cf8183fb392e',
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io', 
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'amm_implementation': '0x8d1fdf6da13f1dd76597dee6fd9a1a16dff4e147',
        'name': 'Optimism'
    }
}

# SportsAMM ABI methods that might reveal the manager
AMM_ABI = [
    {
        "inputs": [],
        "name": "manager",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "sportsAMMUtils",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Try to get a market to test
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
    }
]

# Once we find a market, use these methods
MARKET_ABI = [
    {
        "inputs": [],
        "name": "creator",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getGameDetails",
        "outputs": [
            {"name": "", "type": "bytes32"},
            {"name": "", "type": "string"}
        ],
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
        "name": "times",
        "outputs": [
            {"name": "", "type": "uint256"},
            {"name": "", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

def find_manager_from_amm(w3, amm_address):
    """Try to find the manager address from the AMM."""
    logger.info("🔍 Looking for manager in AMM...")
    
    try:
        amm = w3.eth.contract(address=Web3.to_checksum_address(amm_address), abi=AMM_ABI)
        
        # Try manager() method
        try:
            manager = amm.functions.manager().call()
            if manager != '0x0000000000000000000000000000000000000000':
                logger.info(f"✅ Found manager: {manager}")
                return manager
        except:
            pass
            
        # Check storage slots
        logger.info("Checking AMM storage slots...")
        for slot in range(10):
            try:
                data = w3.eth.get_storage_at(amm_address, slot)
                if data != b'\x00' * 32:
                    # Check if it looks like an address
                    potential_addr = '0x' + data.hex()[-40:]
                    try:
                        # Verify it's a contract
                        code = w3.eth.get_code(potential_addr)
                        if code and len(code) > 1000:  # Manager should be substantial
                            logger.info(f"  Slot {slot}: {potential_addr} (code size: {len(code)})")
                    except:
                        pass
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error checking AMM: {e}")
        
    return None

def find_sample_market(w3, amm_address):
    """Try to find a sample market."""
    logger.info("🎯 Looking for sample markets...")
    
    try:
        # Look for recent MarketCreated-like events
        current_block = w3.eth.block_number
        
        # Common event signatures
        event_sigs = [
            Web3.keccak(text="BuyFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex(),
            Web3.keccak(text="SellToAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
        ]
        
        # Scan recent blocks for AMM activity
        logs = w3.eth.get_logs({
            'fromBlock': current_block - 1000,
            'toBlock': 'latest',
            'address': amm_address
        })
        
        markets = set()
        
        for log in logs[:50]:  # Check first 50 logs
            # Market address is often in topics[1] or topics[2]
            for i in range(1, min(4, len(log['topics']))):
                try:
                    potential_market = '0x' + log['topics'][i].hex()[-40:]
                    addr = Web3.to_checksum_address(potential_market)
                    
                    # Verify it's a contract
                    code = w3.eth.get_code(addr)
                    if code and len(code) > 100:
                        markets.add(addr)
                except:
                    pass
                    
        logger.info(f"Found {len(markets)} potential markets from AMM activity")
        return list(markets)
        
    except Exception as e:
        logger.debug(f"Error finding markets: {e}")
        return []

def analyze_market(w3, market_address):
    """Analyze a market to find its creator (manager)."""
    try:
        market = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        # Get creator
        creator = market.functions.creator().call()
        logger.info(f"  Market {market_address}")
        logger.info(f"    Creator: {creator}")
        
        # Get details
        try:
            home = market.functions.homeTeam().call()
            away = market.functions.awayTeam().call()
            logger.info(f"    Teams: {home} vs {away}")
            
            # Add to database if valid
            times = market.functions.times().call()
            maturity = datetime.fromtimestamp(times[0], tz=timezone.utc)
            
            if maturity > datetime.now(timezone.utc):
                market_id = f"blockchain_arbitrum_discovered_{market_address.lower()}"
                
                with db_manager.get_db_session() as db:
                    if not db.query(Market).filter(Market.source_id == market_id).first():
                        market_obj = Market(
                            source_id=market_id,
                            source="blockchain_arbitrum_discovered",
                            sport="Soccer",
                            league_name="Overtime Markets",
                            market_type="winner",
                            home_team=home,
                            away_team=away,
                            maturity_date=maturity,
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market_obj)
                        db.commit()
                        logger.info(f"    ✅ Added to database!")
        except:
            pass
            
        return creator
        
    except Exception as e:
        logger.debug(f"Error analyzing market: {e}")
        return None

def main():
    """Main function."""
    logger.info("🔍 Finding Real SportPositionalMarketManager")
    logger.info("=" * 60)
    
    # Focus on Arbitrum first
    network = 'arbitrum'
    config = NETWORKS[network]
    
    logger.info(f"\n📡 Analyzing {config['name']}...")
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return
            
        logger.info(f"Connected to {network}")
        
        # Try to find manager from AMM
        manager = find_manager_from_amm(w3, config['sports_amm'])
        
        # Find sample markets
        sample_markets = find_sample_market(w3, config['sports_amm'])
        
        if sample_markets:
            logger.info(f"\n📊 Analyzing {len(sample_markets)} markets...")
            
            creators = set()
            for market in sample_markets[:10]:
                creator = analyze_market(w3, market)
                if creator and creator != '0x0000000000000000000000000000000000000000':
                    creators.add(creator)
                    
            if creators:
                logger.info(f"\n🎯 Found {len(creators)} unique creators (potential managers):")
                for creator in creators:
                    code = w3.eth.get_code(creator)
                    logger.info(f"  {creator} (code size: {len(code)})")
                    
    except Exception as e:
        logger.error(f"Error: {e}")
        
    # Show database status
    with db_manager.get_db_session() as db:
        count = db.query(Market).count()
        logger.info(f"\n📊 Total markets in database: {count}")

if __name__ == "__main__":
    main()