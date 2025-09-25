#!/usr/bin/env python3
"""
Fetch real Overtime markets using correct approach:
1. Use SportsAMM to find markets (it's the market registry)
2. Get market details from individual contracts
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Verified SportsAMM contracts
NETWORKS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    }
}

# SportsAMM ABI - methods that reveal markets
AMM_ABI = [
    # Get market address for specific game
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
    },
    # Get all markets for a date
    {
        "inputs": [
            {"name": "_date", "type": "uint256"}
        ],
        "name": "getAllMarketsForDate",
        "outputs": [{"name": "", "type": "address[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    # Manager address (might be the SportPositionalMarketManager)
    {
        "inputs": [],
        "name": "manager",
        "outputs": [{"name": "", "type": "address"}],
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
    },
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

def scan_recent_amm_activity(w3, amm_address):
    """Find markets from recent AMM buy/sell activity."""
    logger.info("🔍 Scanning recent AMM activity for markets...")
    
    markets = set()
    
    try:
        current_block = w3.eth.block_number
        
        # BuyFromAMM event
        buy_topic = Web3.keccak(text="BuyFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
        # SellToAMM event  
        sell_topic = Web3.keccak(text="SellToAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
        
        # Get recent logs - scan for each event type separately
        all_logs = []
        
        # Get buy events
        try:
            buy_logs = w3.eth.get_logs({
                'fromBlock': current_block - 1000,
                'toBlock': 'latest',
                'address': amm_address,
                'topics': [buy_topic]
            })
            all_logs.extend(buy_logs)
        except:
            pass
            
        # Get sell events
        try:
            sell_logs = w3.eth.get_logs({
                'fromBlock': current_block - 1000,
                'toBlock': 'latest',
                'address': amm_address,
                'topics': [sell_topic]
            })
            all_logs.extend(sell_logs)
        except:
            pass
            
        logs = all_logs
        
        logger.info(f"Found {len(logs)} buy/sell events")
        
        for log in logs:
            # Market address is second topic
            if len(log['topics']) > 2:
                market_addr = '0x' + log['topics'][2].hex()[-40:]
                markets.add(Web3.to_checksum_address(market_addr))
                
    except Exception as e:
        logger.error(f"Error scanning events: {e}")
        
    return list(markets)

def get_market_details(w3, market_address):
    """Get details from a market contract."""
    try:
        market = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        details = {'address': market_address}
        
        # Get teams
        details['homeTeam'] = market.functions.homeTeam().call()
        details['awayTeam'] = market.functions.awayTeam().call()
        
        # Get timing
        times = market.functions.times().call()
        details['maturity'] = times[0]
        
        # Get status
        details['resolved'] = market.functions.resolved().call()
        details['cancelled'] = market.functions.cancelled().call()
        
        # Get sport
        try:
            tags = market.functions.tags().call()
            if tags:
                details['sportId'] = tags[0]
        except:
            details['sportId'] = 9004  # Default to soccer
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def fetch_from_network(network):
    """Fetch markets from a specific network."""
    config = NETWORKS[network]
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network}")
        
        # First, check AMM for manager
        amm = w3.eth.contract(address=Web3.to_checksum_address(config['sports_amm']), abi=AMM_ABI)
        try:
            manager = amm.functions.manager().call()
            logger.info(f"Manager from AMM: {manager}")
        except:
            pass
            
        # Find markets from recent activity
        market_addresses = scan_recent_amm_activity(w3, config['sports_amm'])
        
        logger.info(f"Found {len(market_addresses)} unique markets from activity")
        
        # Process each market
        for i, market_addr in enumerate(market_addresses):
            if i >= 30:  # Limit for testing
                break
                
            try:
                details = get_market_details(w3, market_addr)
                
                if not details or not details.get('homeTeam'):
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    continue
                    
                # Skip past games
                maturity = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"blockchain_{network}_real_{market_addr.lower()}"
                
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
                        9008: "Tennis"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_real",
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
                    logger.info(f"   Maturity: {maturity}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error processing {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Overtime Real Market Fetcher")
    logger.info("=" * 60)
    logger.info("Using SportsAMM to find active markets")
    
    total_added = 0
    
    # Process each network
    for network in ['arbitrum', 'optimism']:
        added = fetch_from_network(network)
        total_added += added
        time.sleep(2)
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ REAL MARKET FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(5).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")

if __name__ == "__main__":
    main()