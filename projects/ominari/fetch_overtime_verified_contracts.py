#!/usr/bin/env python3
"""
Fetch Overtime markets using VERIFIED contract addresses and ABIs
Following the architecture guide provided
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# VERIFIED CONTRACT ADDRESSES from the guide
VERIFIED_CONTRACTS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        # SportPositionalMarketManager - verified on Arbiscan
        'manager': '0x268BB40F4993f6234D924ba70D20BD59d781F7F6',
        # SportsAMM
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum',
        'chain_id': 42161
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        # SportPositionalMarketManager for Optimism
        'manager': '0x81DD7B07eb4bc9ffF0274d5C7F326b96B6557e53',
        # SportsAMM verified on Optimistic Etherscan
        'sports_amm': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'name': 'Optimism',
        'chain_id': 10
    }
}

# SportPositionalMarketManager ABI (key methods from verified contract)
MANAGER_ABI = [
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
        "inputs": [{"internalType": "uint256", "name": "index", "type": "uint256"}],
        "name": "getActiveMarketAddress",
        "outputs": [{"internalType": "address", "name": "", "type": "address"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"internalType": "address", "name": "candidate", "type": "address"}],
        "name": "isKnownMarket",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function"
    }
]

# Game Market ABI (from verified market contracts)
MARKET_ABI = [
    {
        "inputs": [],
        "name": "getGameDetails",
        "outputs": [
            {"internalType": "bytes32", "name": "", "type": "bytes32"},
            {"internalType": "string", "name": "", "type": "string"}
        ],
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
        "name": "homeTeam",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "awayTeam",
        "outputs": [{"internalType": "string", "name": "", "type": "string"}],
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
    },
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"internalType": "uint256[]", "name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [],
        "name": "getParentMarketPositions",
        "outputs": [
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "address", "name": "", "type": "address"},
            {"internalType": "address", "name": "", "type": "address"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

def fetch_active_markets_from_manager(w3, manager_address, network):
    """Fetch active markets using the verified SportPositionalMarketManager."""
    logger.info(f"📋 Querying SportPositionalMarketManager on {network}...")
    
    markets = []
    
    try:
        manager = w3.eth.contract(address=Web3.to_checksum_address(manager_address), abi=MANAGER_ABI)
        
        # Get number of active markets
        num_markets = manager.functions.numActiveMarkets().call()
        logger.info(f"✅ Manager reports {num_markets} active markets")
        
        if num_markets > 0:
            # Page through markets
            page_size = 100
            for i in range(0, num_markets, page_size):
                try:
                    # Use activeMarkets(index, pageSize)
                    page = manager.functions.activeMarkets(i, min(page_size, num_markets - i)).call()
                    markets.extend(page)
                    logger.info(f"  Fetched page starting at {i}, got {len(page)} markets")
                except Exception as e:
                    logger.error(f"Error fetching page at {i}: {e}")
                    # Try individual fetching as fallback
                    for j in range(i, min(i + page_size, num_markets)):
                        try:
                            addr = manager.functions.getActiveMarketAddress(j).call()
                            if addr != '0x0000000000000000000000000000000000000000':
                                markets.append(addr)
                        except:
                            pass
                            
    except Exception as e:
        logger.error(f"Error querying manager: {e}")
        
    return markets

def get_market_details(w3, market_address):
    """Get details from a verified Game Market contract."""
    try:
        market = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        details = {
            'address': market_address,
            'valid': True
        }
        
        # Get teams
        try:
            details['homeTeam'] = market.functions.homeTeam().call()
            details['awayTeam'] = market.functions.awayTeam().call()
        except:
            # Try via game details
            try:
                game_id, game_label = market.functions.getGameDetails().call()
                details['gameId'] = game_id
                details['gameLabel'] = game_label
                # Parse teams from label
                if ' vs ' in game_label:
                    parts = game_label.split(' vs ')
                    details['homeTeam'] = parts[0].strip()
                    details['awayTeam'] = parts[1].strip()
            except:
                pass
                
        # Get timing
        try:
            maturity, destruction = market.functions.times().call()
            details['maturity'] = maturity
            details['expiry'] = destruction
        except:
            pass
            
        # Get status
        try:
            details['resolved'] = market.functions.resolved().call()
        except:
            details['resolved'] = False
            
        try:
            details['cancelled'] = market.functions.cancelled().call()
        except:
            details['cancelled'] = False
            
        # Get sport tags
        try:
            tags = market.functions.tags().call()
            if tags:
                details['sportId'] = tags[0]
        except:
            details['sportId'] = 9004  # Default to soccer
            
        # Get outcome token addresses
        try:
            positions = market.functions.getParentMarketPositions().call()
            details['hasPositions'] = positions[0] != '0x0000000000000000000000000000000000000000'
        except:
            details['hasPositions'] = True
            
        return details
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def process_network_markets(network):
    """Process markets from a specific network."""
    config = VERIFIED_CONTRACTS[network]
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network} at block {w3.eth.block_number:,}")
        
        # Fetch active markets from manager
        market_addresses = fetch_active_markets_from_manager(w3, config['manager'], network)
        
        logger.info(f"Found {len(market_addresses)} market addresses")
        
        # Process each market
        for i, market_addr in enumerate(market_addresses):
            if i >= 50:  # Limit for initial testing
                logger.info("Reached 50 market limit for testing")
                break
                
            try:
                details = get_market_details(w3, market_addr)
                
                if not details:
                    continue
                    
                # Skip if no teams found
                if not details.get('homeTeam') or not details.get('awayTeam'):
                    logger.debug(f"Skipping market {market_addr} - no teams")
                    continue
                    
                # Skip resolved/cancelled
                if details.get('resolved') or details.get('cancelled'):
                    logger.debug(f"Skipping market {market_addr} - resolved/cancelled")
                    continue
                    
                # Skip past games
                maturity = datetime.fromtimestamp(
                    details.get('maturity', 0),
                    tz=timezone.utc
                )
                if maturity < datetime.now(timezone.utc):
                    logger.debug(f"Skipping market {market_addr} - past game")
                    continue
                    
                market_id = f"blockchain_{network}_verified_{market_addr.lower()}"
                
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
                        9008: "Tennis",
                        9010: "Golf",
                        9011: "Cricket",
                        9012: "Rugby",
                        9014: "Motorsport"
                    }
                    
                    market = Market(
                        source_id=market_id,
                        source=f"blockchain_{network}_verified",
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
                    logger.info(f"   Sport: {sport_map.get(details.get('sportId', 9004), 'Soccer')}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
        logger.info(f"Added {markets_added} markets from {network}")
        
    except Exception as e:
        logger.error(f"Error processing {network}: {e}")
        
    return markets_added

def main():
    """Main function."""
    logger.info("🎯 Overtime Verified Contract Fetcher")
    logger.info("=" * 60)
    logger.info("Using verified SportPositionalMarketManager contracts")
    
    total_added = 0
    
    # Process each network
    for network in ['arbitrum', 'optimism']:
        added = process_network_markets(network)
        total_added += added
        time.sleep(2)  # Be nice to RPCs
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n✨ VERIFIED FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Active future markets: {active}")
        logger.info(f"Markets added this run: {total_added}")
        
        if total > 0:
            samples = db.query(Market).order_by(Market.updated_at.desc()).limit(10).all()
            logger.info("\n📊 Latest markets:")
            for m in samples:
                logger.info(f"  • {m.home_team} vs {m.away_team} ({m.sport})")
                logger.info(f"    {m.maturity_date}")

if __name__ == "__main__":
    main()