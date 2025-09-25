#!/usr/bin/env python3
"""
Fetch Overtime markets using event logs from verified contracts
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

# VERIFIED ADDRESSES FROM USER
CONTRACTS = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'SportsAMM': '0x410AfcF3Abe7A72DD1B74942918d110Ef4a3eDB4',
        'Manager': '0x72ca0765d4bE0529377d656c9645600606214610',
        'Factory': '0x85b827d133FEDC36B844b20f4a198dA583B25BAA',
        'name': 'Arbitrum'
    }
}

# Minimal ABIs
AMM_ABI = [
    {
        "inputs": [
            {"name": "market", "type": "address"},
            {"name": "position", "type": "uint8"}
        ],
        "name": "obtainOdds",
        "outputs": [{"name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "market", "type": "address"}],
        "name": "getMarketDefaultOdds",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "stateMutability": "view",
        "type": "function"
    }
]

MARKET_ABI = [
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

def find_markets_from_amm_events(w3, amm_address):
    """Find markets from AMM buy/sell events."""
    logger.info(f"🔍 Scanning AMM events for market addresses...")
    
    markets = set()
    
    # Event signatures
    buy_sig = Web3.keccak(text="BuyFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
    sell_sig = Web3.keccak(text="SellToAMM(address,address,uint8,uint256,uint256,uint256,uint256)").hex()
    
    current_block = w3.eth.block_number
    
    # Scan in chunks
    chunk_size = 5000
    blocks_to_scan = 50000  # Last ~50k blocks
    
    for i in range(0, blocks_to_scan, chunk_size):
        from_block = current_block - blocks_to_scan + i
        to_block = min(from_block + chunk_size - 1, current_block)
        
        try:
            # Get buy events
            buy_logs = w3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': amm_address,
                'topics': [buy_sig]
            })
            
            for log in buy_logs:
                if len(log['topics']) > 2:
                    market_addr = '0x' + log['topics'][2].hex()[-40:]
                    markets.add(Web3.to_checksum_address(market_addr))
                    
            # Get sell events
            sell_logs = w3.eth.get_logs({
                'fromBlock': from_block,
                'toBlock': to_block,
                'address': amm_address,
                'topics': [sell_sig]
            })
            
            for log in sell_logs:
                if len(log['topics']) > 2:
                    market_addr = '0x' + log['topics'][2].hex()[-40:]
                    markets.add(Web3.to_checksum_address(market_addr))
                    
            if len(buy_logs) + len(sell_logs) > 0:
                logger.info(f"  Block {from_block}-{to_block}: Found {len(buy_logs)} buys, {len(sell_logs)} sells")
                
        except Exception as e:
            logger.debug(f"Error scanning blocks {from_block}-{to_block}: {e}")
            
        if len(markets) >= 50:  # Enough markets
            break
            
    logger.info(f"  Total unique markets found: {len(markets)}")
    return list(markets)

def get_market_info(w3, market_address):
    """Get market information."""
    try:
        contract = w3.eth.contract(address=market_address, abi=MARKET_ABI)
        
        # Get game details
        game_id, game_label = contract.functions.getGameDetails().call()
        
        # Parse teams
        home_team = None
        away_team = None
        if ' vs ' in game_label:
            parts = game_label.split(' vs ')
            home_team = parts[0].strip()
            away_team = parts[1].strip()
        elif ' @ ' in game_label:
            parts = game_label.split(' @ ')
            away_team = parts[0].strip()
            home_team = parts[1].strip()
            
        if not home_team or not away_team:
            return None
            
        # Get times
        times = contract.functions.times().call()
        maturity = times[0]
        
        # Get status
        resolved = contract.functions.resolved().call()
        
        # Get sport
        sport_id = 9004  # Default to soccer
        try:
            tags = contract.functions.tags().call()
            if tags:
                sport_id = tags[0]
        except:
            pass
            
        return {
            'gameLabel': game_label,
            'homeTeam': home_team,
            'awayTeam': away_team,
            'maturity': maturity,
            'resolved': resolved,
            'sportId': sport_id
        }
        
    except Exception as e:
        logger.debug(f"Error reading market {market_address}: {e}")
        return None

def get_odds(w3, amm_address, market_address):
    """Get odds from AMM."""
    try:
        amm = w3.eth.contract(address=amm_address, abi=AMM_ABI)
        
        # Try getMarketDefaultOdds first
        try:
            default_odds = amm.functions.getMarketDefaultOdds(market_address).call()
            if default_odds and len(default_odds) >= 2:
                odds = {}
                if default_odds[0] > 0:
                    odds[0] = default_odds[0] / 1e18  # HOME
                if default_odds[1] > 0:
                    odds[1] = default_odds[1] / 1e18  # AWAY
                if len(default_odds) > 2 and default_odds[2] > 0:
                    odds[2] = default_odds[2] / 1e18  # DRAW
                return odds
        except:
            pass
            
        # Try individual positions
        odds = {}
        for position in range(3):
            try:
                odd = amm.functions.obtainOdds(market_address, position).call()
                if odd > 0:
                    odds[position] = odd / 1e18
            except:
                pass
                
        return odds
        
    except:
        return {}

def main():
    """Main function."""
    logger.info("🎯 Overtime Market Fetcher - Event Based")
    logger.info("=" * 60)
    
    config = CONTRACTS['arbitrum']
    logger.info(f"\n🌐 Processing {config['name']}...")
    
    markets_added = 0
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error("Failed to connect")
            return
            
        logger.info(f"Connected at block {w3.eth.block_number:,}")
        
        # Find markets from AMM events
        market_addresses = find_markets_from_amm_events(w3, config['SportsAMM'])
        
        if not market_addresses:
            logger.warning("No markets found")
            return
            
        # Process markets
        logger.info(f"\n🏆 Processing {len(market_addresses)} markets...")
        
        for i, market_addr in enumerate(market_addresses):
            if i % 10 == 0 and i > 0:
                logger.info(f"  Progress: {i}/{len(market_addresses)}")
                
            try:
                info = get_market_info(w3, market_addr)
                
                if not info:
                    continue
                    
                # Skip resolved
                if info['resolved']:
                    continue
                    
                # Check maturity
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"arbitrum_event_{market_addr.lower()}"
                
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
                        source="arbitrum_event",
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
                    db.flush()
                    
                    # Get odds
                    odds = get_odds(w3, config['SportsAMM'], market_addr)
                    
                    for position, decimal_odds in odds.items():
                        outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                        outcome = outcome_map.get(position)
                        if outcome and decimal_odds > 1.0:
                            # Convert to American
                            if decimal_odds >= 2.0:
                                american = int((decimal_odds - 1) * 100)
                            else:
                                american = int(-100 / (decimal_odds - 1))
                                
                            odd = Odd(
                                source_id=market.source_id,
                                market_type="winner",
                                outcome=outcome,
                                source="arbitrum_amm",
                                bookmaker="Overtime",
                                decimal_odds=decimal_odds,
                                american_odds=american,
                                normalized_implied=1.0 / decimal_odds,
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                            
                    db.commit()
                    
                    markets_added += 1
                    logger.info(f"✅ Added: {info['gameLabel']}")
                    logger.info(f"   Contract: {market_addr}")
                    logger.info(f"   Maturity: {maturity}")
                    if odds:
                        logger.info(f"   Odds: {odds}")
                        
            except Exception as e:
                logger.error(f"Error processing market: {e}")
                continue
                
    except Exception as e:
        logger.error(f"Error: {e}")
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        
        logger.info(f"\n✨ EVENT FETCH COMPLETE ✨")
        logger.info(f"Total markets in database: {total}")
        logger.info(f"Markets added this run: {markets_added}")
        
        if markets_added > 0:
            logger.info("\n🎆 SUCCESS! Found real Overtime markets!")

if __name__ == "__main__":
    main()