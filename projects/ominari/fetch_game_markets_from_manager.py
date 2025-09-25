#!/usr/bin/env python3
"""
Fetch real Game Market contracts using the documented architecture:
1. TheRundownConsumerWrapper -> RequestGames
2. TheRundownConsumer -> CreateMarketForGame  
3. SportPositionalMarketManager -> Creates individual Game Market contracts
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

# Known Overtime/Thales contract addresses based on architecture
OVERTIME_INFRASTRUCTURE = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'rundown_consumer': '0x82F80c58d06b3874013f0aFe334303e8A91C2a6f',
        'market_manager': '0x56c85448A9d0f0CcE0c31C6a50FC93D7a6bFe39F',
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'wrapper': '0x0F0e3379Fa20f8669b1F38D3Bb24E889B1f7F9d4',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'rundown_consumer': '0xE3D0c5dBBb604fDF6a48604e2dc5fA9B9383A8Df',
        'market_manager': '0x91b0d67a06936ad75c13e2b5f14F36dcf22D12Aa',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'wrapper': '0x3e80fA93428c8a2a74E02bcaB94DC1201eE70E6E',
        'name': 'Arbitrum'
    }
}

# ABI for SportPositionalMarketManager
MARKET_MANAGER_ABI = [
    {
        "inputs": [],
        "name": "getAllActiveMarkets",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "index", "type": "uint256"}],
        "name": "activeMarkets",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "numberOfActiveMarkets",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# ABI for Game Market contracts based on documentation
GAME_MARKET_ABI = [
    # Game details
    {
        "inputs": [],
        "name": "gameDetails",
        "outputs": [
            {"name": "gameId", "type": "bytes32"},
            {"name": "startTime", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Teams
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
    # Outcome tokens
    {
        "inputs": [],
        "name": "getOptions",
        "outputs": [
            {"name": "home", "type": "address"},
            {"name": "away", "type": "address"},
            {"name": "draw", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Market status
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
    },
    # Tags for sport type
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Rundown Consumer ABI for created markets
RUNDOWN_CONSUMER_ABI = [
    {
        "inputs": [{"name": "_gameId", "type": "bytes32"}],
        "name": "marketPerGameId",
        "outputs": [{"name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "_market", "type": "address"}],
        "name": "gameIdPerMarket", 
        "outputs": [{"name": "", "type": "bytes32"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def get_active_markets_from_manager(w3, manager_address):
    """Get active Game Market contracts from the SportPositionalMarketManager."""
    logger.info("📋 Querying SportPositionalMarketManager...")
    
    markets = []
    
    try:
        contract = w3.eth.contract(address=Web3.to_checksum_address(manager_address), abi=MARKET_MANAGER_ABI)
        
        # Try to get number of active markets
        try:
            num_markets = contract.functions.numberOfActiveMarkets().call()
            logger.info(f"Manager reports {num_markets} active markets")
            
            # Get individual markets
            for i in range(min(num_markets, 50)):  # Limit to 50
                try:
                    market_addr = contract.functions.activeMarkets(i).call()
                    if market_addr != '0x0000000000000000000000000000000000000000':
                        markets.append(market_addr)
                except:
                    pass
                    
        except:
            # Try getAllActiveMarkets
            try:
                all_markets = contract.functions.getAllActiveMarkets().call()
                markets = [m for m in all_markets if m != '0x0000000000000000000000000000000000000000']
                logger.info(f"Got {len(markets)} markets from getAllActiveMarkets")
            except:
                pass
                
    except Exception as e:
        logger.error(f"Error querying manager: {e}")
        
    return markets

def get_game_market_details(w3, market_address):
    """Get detailed information from a Game Market contract."""
    try:
        contract = w3.eth.contract(address=market_address, abi=GAME_MARKET_ABI)
        
        details = {
            'address': market_address,
            'valid': True
        }
        
        # Get teams
        try:
            details['homeTeam'] = contract.functions.homeTeam().call()
            details['awayTeam'] = contract.functions.awayTeam().call()
        except:
            # Alternative structure
            try:
                game_details = contract.functions.gameDetails().call()
                details['gameId'] = game_details[0]
                details['startTime'] = game_details[1]
            except:
                pass
                
        # Get status
        try:
            details['resolved'] = contract.functions.resolved().call()
            details['cancelled'] = contract.functions.cancelled().call()
        except:
            details['resolved'] = False
            details['cancelled'] = False
            
        # Get outcome tokens (to verify it's a valid market)
        try:
            options = contract.functions.getOptions().call()
            details['hasOptions'] = options[0] != '0x0000000000000000000000000000000000000000'
        except:
            details['hasOptions'] = False
            
        # Get sport from tags
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

def scan_recent_market_creations(w3, consumer_address, network):
    """Scan for recently created markets via RundownConsumer events."""
    logger.info("🔍 Scanning for recent market creations...")
    
    markets = []
    
    try:
        # MarketCreated event signature
        market_created_topic = Web3.keccak(text="MarketCreated(address,bytes32,string,uint256)").hex()
        
        current_block = w3.eth.block_number
        from_block = current_block - 5000  # Last ~5000 blocks
        
        # Get logs
        logs = w3.eth.get_logs({
            'fromBlock': from_block,
            'toBlock': 'latest',
            'address': consumer_address,
            'topics': [market_created_topic]
        })
        
        logger.info(f"Found {len(logs)} MarketCreated events")
        
        for log in logs:
            try:
                # Market address is usually first indexed parameter
                if len(log['topics']) > 1:
                    market_addr = '0x' + log['topics'][1].hex()[-40:]
                    markets.append(Web3.to_checksum_address(market_addr))
            except:
                pass
                
    except Exception as e:
        logger.debug(f"Event scanning error: {e}")
        
    return markets

def fetch_from_network(network):
    """Fetch Game Markets from a specific network."""
    config = OVERTIME_INFRASTRUCTURE[network]
    logger.info(f"\n🎮 Fetching Game Markets from {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # Get markets from SportPositionalMarketManager
        manager_markets = get_active_markets_from_manager(w3, config['market_manager'])
        
        # Also scan for recent creations
        recent_markets = scan_recent_market_creations(w3, config['rundown_consumer'], network)
        
        # Combine and deduplicate
        all_markets = list(set(manager_markets + recent_markets))
        logger.info(f"Found {len(all_markets)} unique market addresses")
        
        # Process each market
        for market_addr in all_markets[:30]:  # Limit to 30
            try:
                details = get_game_market_details(w3, market_addr)
                
                if not details or not details.get('homeTeam') or not details.get('awayTeam'):
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                # Skip if no valid options
                if not details.get('hasOptions'):
                    continue
                    
                market_id = f"blockchain_{network}_game_{market_addr.lower()}"
                
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Map sport ID to name
                    sport_map = {
                        9001: "American Football",
                        9002: "Baseball",
                        9003: "Basketball",
                        9004: "Soccer",
                        9005: "Hockey",
                        9006: "MMA",
                        9008: "Tennis"
                    }
                    
                    # Use start time or default to future
                    if details.get('startTime'):
                        maturity = datetime.fromtimestamp(details['startTime'], tz=timezone.utc)
                    else:
                        maturity = datetime.now(timezone.utc) + timedelta(days=2)
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_game",
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
                    logger.info(f"✅ Added Game Market: {details['homeTeam']} vs {details['awayTeam']}")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Sport ID: {details.get('sportId')}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎮 Game Market Contract Fetcher")
    logger.info("=" * 60)
    logger.info("Using documented Overtime architecture")
    
    total_added = 0
    
    for network in ['optimism', 'arbitrum']:
        added = fetch_from_network(network)
        total_added += added
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        
        logger.info(f"\n✨ GAME MARKET FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Game Markets added: {total_added}")
        
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest Game Markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    ID: {m.source_id}")

if __name__ == "__main__":
    main()