#!/usr/bin/env python3
"""
Fetch active Overtime markets using direct contract calls
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

# Configurations
CONFIGS = {
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc', 
        'sports_amm_v2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

# ABI for getting active markets
SPORTS_AMM_ABI = [
    # Get all active markets
    {
        "inputs": [],
        "name": "activeMarkets",
        "outputs": [
            {
                "internalType": "address[]",
                "name": "",
                "type": "address[]"
            }
        ],
        "stateMutability": "view",
        "type": "function"
    },
    # Alternative: get markets for specific date range
    {
        "inputs": [
            {"internalType": "uint256", "name": "_dateFrom", "type": "uint256"},
            {"internalType": "uint256", "name": "_dateTo", "type": "uint256"}
        ],
        "name": "getMarketsForDateRange",
        "outputs": [
            {"internalType": "address[]", "name": "", "type": "address[]"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Market contract ABI
MARKET_ABI = [
    {
        "inputs": [],
        "name": "gameDetails",
        "outputs": [
            {"name": "gameId", "type": "bytes32"},
            {"name": "gameLabel", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "times",
        "outputs": [
            {"name": "maturity", "type": "uint256"},
            {"name": "expiry", "type": "uint256"}
        ],
        "stateMutability": "view", 
        "type": "function"
    },
    {
        "inputs": [],
        "name": "teams",
        "outputs": [
            {"name": "home", "type": "string"},
            {"name": "away", "type": "string"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
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
    {
        "inputs": [],
        "name": "resolved",
        "outputs": [{"name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def get_active_markets_from_amm(w3, amm_address):
    """Get list of active market addresses from the AMM contract."""
    try:
        contract = w3.eth.contract(address=Web3.to_checksum_address(amm_address), abi=SPORTS_AMM_ABI)
        
        # Try different methods
        markets = []
        
        # Method 1: activeMarkets()
        try:
            markets = contract.functions.activeMarkets().call()
            logger.info(f"Found {len(markets)} active markets via activeMarkets()")
            return markets
        except Exception as e:
            logger.debug(f"activeMarkets() failed: {e}")
            
        # Method 2: getMarketsForDateRange
        try:
            now = int(time.time())
            future = now + (30 * 24 * 60 * 60)  # 30 days ahead
            markets = contract.functions.getMarketsForDateRange(now, future).call()
            logger.info(f"Found {len(markets)} markets via date range")
            return markets
        except Exception as e:
            logger.debug(f"getMarketsForDateRange() failed: {e}")
            
        return []
        
    except Exception as e:
        logger.error(f"Error getting active markets: {e}")
        return []

def get_market_details(w3, market_address):
    """Get details of a specific market contract."""
    try:
        contract = w3.eth.contract(address=Web3.to_checksum_address(market_address), abi=MARKET_ABI)
        
        details = {
            'address': market_address
        }
        
        # Get game details
        try:
            game_info = contract.functions.gameDetails().call()
            details['gameId'] = game_info[0]
            details['gameLabel'] = game_info[1]
        except:
            pass
            
        # Get teams
        try:
            teams = contract.functions.teams().call()
            details['homeTeam'] = teams[0]
            details['awayTeam'] = teams[1]
        except:
            pass
            
        # Get times
        try:
            times = contract.functions.times().call()
            details['maturity'] = times[0]
        except:
            pass
            
        # Check if resolved
        try:
            details['resolved'] = contract.functions.resolved().call()
        except:
            details['resolved'] = False
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def fetch_markets_from_network(network):
    """Fetch all active markets from a network."""
    config = CONFIGS[network]
    logger.info(f"\n🌐 Fetching from {config['name']}...")
    
    markets_added = 0
    
    try:
        # Connect
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # Get active markets
        market_addresses = get_active_markets_from_amm(w3, config['sports_amm_v2'])
        
        if not market_addresses:
            logger.warning("No active markets found via contract calls")
            # Try alternative: scan recent blocks for events
            return fetch_via_events(w3, config, network)
            
        # Process each market
        for i, market_addr in enumerate(market_addresses[:30]):  # Limit to 30
            try:
                details = get_market_details(w3, market_addr)
                
                if not details or not details.get('homeTeam'):
                    continue
                    
                # Skip resolved markets
                if details.get('resolved'):
                    continue
                    
                # Skip past games
                maturity = datetime.fromtimestamp(
                    details.get('maturity', 0), 
                    tz=timezone.utc
                )
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_active_{market_addr.lower()}"
                
                # Add to database
                with db_manager.get_db_session() as db:
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_active",
                        sport="Soccer",  # Could parse from gameLabel
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
                    logger.info(f"   Address: {market_addr}")
                    logger.info(f"   Date: {maturity}")
                    
            except Exception as e:
                logger.debug(f"Error processing market {i}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        
    return markets_added

def fetch_via_events(w3, config, network):
    """Fallback: fetch via recent events."""
    logger.info("Trying event-based approach...")
    
    markets_added = 0
    current_block = w3.eth.block_number
    
    # MarketCreated event signature
    market_created_sig = Web3.keccak(text="MarketCreated(address,bytes32,string,uint256)").hex()
    
    # Scan last 1000 blocks
    try:
        logs = w3.eth.get_logs({
            'fromBlock': current_block - 1000,
            'toBlock': 'latest',
            'address': config['sports_amm_v2']
        })
        
        logger.info(f"Found {len(logs)} events")
        
        for log in logs[:20]:  # Process first 20
            try:
                # Extract market address from topics
                if len(log['topics']) > 1:
                    market_addr = '0x' + log['topics'][1].hex()[-40:]
                    
                    # Get details
                    details = get_market_details(w3, market_addr)
                    if details and details.get('homeTeam'):
                        # Add market (similar to above)
                        market_id = f"blockchain_{network}_event_{market_addr.lower()}"
                        
                        with db_manager.get_db_session() as db:
                            if not db.query(Market).filter(Market.source_id == market_id).first():
                                market = Market(
                                    source_id=market_id,
                                    source=f"blockchain_{network}_event",
                                    sport="Soccer",
                                    league_name="Overtime Markets",
                                    market_type="winner",
                                    home_team=details['homeTeam'],
                                    away_team=details['awayTeam'],
                                    maturity_date=datetime.fromtimestamp(
                                        details.get('maturity', 0),
                                        tz=timezone.utc
                                    ),
                                    is_finished=False,
                                    updated_at=datetime.now(timezone.utc)
                                )
                                db.add(market)
                                db.commit()
                                markets_added += 1
                                logger.info(f"✅ Added via event: {details['homeTeam']} vs {details['awayTeam']}")
                                
            except Exception as e:
                logger.debug(f"Error processing event: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Event scan failed: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🏈 Overtime Active Markets Fetcher")
    logger.info("=" * 60)
    
    # Show current status
    with db_manager.get_db_session() as db:
        before_count = db.query(Market).count()
        logger.info(f"Starting with {before_count} markets")
    
    total_added = 0
    
    # Fetch from each network
    for network in ['optimism', 'arbitrum']:
        added = fetch_markets_from_network(network)
        total_added += added
        time.sleep(2)  # Rate limiting
    
    # Final summary
    with db_manager.get_db_session() as db:
        after_count = db.query(Market).count()
        active_count = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ FETCH COMPLETE ✨")
        logger.info(f"Markets: {before_count} → {after_count} (+{total_added})")
        logger.info(f"Active future markets: {active_count}")
        
        # Show samples
        samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
        if samples:
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team}")
                logger.info(f"    {m.maturity_date}")

if __name__ == "__main__":
    main()