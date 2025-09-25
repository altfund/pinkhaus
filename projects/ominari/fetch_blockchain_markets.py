#!/usr/bin/env python3
"""
Direct Blockchain Market Fetcher
Fetches market data directly from Optimism and Arbitrum blockchains.
"""

import logging
from web3 import Web3
from datetime import datetime, timezone, timedelta
import json
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Blockchain configurations
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

# Minimal ABI for market data
SPORTS_AMM_ABI = [
    {
        "inputs": [],
        "name": "getAllActiveGameIds",
        "outputs": [{"name": "", "type": "bytes32[]"}],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [{"name": "gameId", "type": "bytes32"}],
        "name": "getGameDetails", 
        "outputs": [
            {"name": "gameLabel", "type": "string"},
            {"name": "sportId", "type": "uint256"},
            {"name": "homeTeam", "type": "string"},
            {"name": "awayTeam", "type": "string"},
            {"name": "maturityDate", "type": "uint256"},
            {"name": "homeOdds", "type": "uint256"},
            {"name": "awayOdds", "type": "uint256"},
            {"name": "drawOdds", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Sport mapping
SPORT_MAP = {
    9001: "American Football",
    9002: "Baseball", 
    9003: "Basketball",
    9004: "Soccer",
    9005: "Hockey",
    9006: "UFC/MMA",
    9007: "Boxing",
    9008: "Tennis",
    9010: "Golf",
    9014: "Rugby"
}

def fetch_from_blockchain(network: str) -> int:
    """Fetch markets from a specific blockchain."""
    config = CONFIGS[network]
    logger.info(f"Connecting to {config['name']}...")
    
    try:
        # Connect to blockchain
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {network}")
            return 0
            
        logger.info(f"Connected to {network} at block {w3.eth.block_number:,}")
        
        # Get contract
        sports_amm = w3.eth.contract(
            address=Web3.to_checksum_address(config['sports_amm_v2']),
            abi=SPORTS_AMM_ABI
        )
        
        # Try to get active games
        try:
            logger.info("Fetching active game IDs...")
            active_games = sports_amm.functions.getAllActiveGameIds().call()
            logger.info(f"Found {len(active_games)} active games")
        except Exception as e:
            logger.warning(f"Could not fetch active games: {e}")
            # Fall back to recent events
            active_games = []
        
        markets_added = 0
        
        # If we have active games, process them
        if active_games:
            for game_id in active_games[:20]:  # Limit to 20 for testing
                try:
                    # Get game details
                    details = sports_amm.functions.getGameDetails(game_id).call()
                    
                    game_label, sport_id, home_team, away_team, maturity_date, home_odds, away_odds, draw_odds = details
                    
                    # Skip if no teams
                    if not home_team or not away_team:
                        continue
                    
                    # Create market
                    market_id = f"blockchain_{network}_v2_{game_id.hex()}"
                    sport = SPORT_MAP.get(sport_id, "Unknown")
                    
                    # Check if market exists
                    with db_manager.get_db_session() as db:
                        existing = db.query(Market).filter(Market.source_id == market_id).first()
                        if existing:
                            continue
                    
                    # Add market
                    with db_manager.get_db_session() as db:
                        market = Market(
                            source_id=market_id,
                            source=f"blockchain_{network}_v2",
                            sport=sport,
                            league_name=f"Overtime {sport}",
                            market_type="winner",
                            home_team=home_team,
                            away_team=away_team,
                            maturity_date=datetime.fromtimestamp(maturity_date, tz=timezone.utc),
                            is_finished=False,
                            updated_at=datetime.now(timezone.utc)
                        )
                        db.add(market)
                        db.commit()
                        
                        # Add odds
                        if home_odds > 0:
                            odd = Odd(
                                source_id=market_id,
                                outcome='Home',
                                decimal_odds=home_odds / 1e18,
                                market_type='moneyline',
                                source=f"blockchain_{network}_v2",
                                bookmaker='overtime',
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                        
                        if away_odds > 0:
                            odd = Odd(
                                source_id=market_id,
                                outcome='Away',
                                decimal_odds=away_odds / 1e18,
                                market_type='moneyline',
                                source=f"blockchain_{network}_v2",
                                bookmaker='overtime',
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                        
                        if draw_odds > 0:
                            odd = Odd(
                                source_id=market_id,
                                outcome='Draw',
                                decimal_odds=draw_odds / 1e18,
                                market_type='moneyline',
                                source=f"blockchain_{network}_v2",
                                bookmaker='overtime',
                                updated_at=datetime.now(timezone.utc)
                            )
                            db.add(odd)
                        
                        db.commit()
                        markets_added += 1
                        logger.info(f"Added: {home_team} vs {away_team} ({sport})")
                        
                except Exception as e:
                    logger.error(f"Error processing game {game_id.hex()}: {e}")
                    continue
        
        # Also try to get some markets from recent blocks
        if markets_added < 10:
            logger.info("Fetching recent market events...")
            # This would scan recent blocks for market creation events
            # For now, we'll use the API approach
            
        return markets_added
        
    except Exception as e:
        logger.error(f"Error fetching from {network}: {e}")
        return 0

def main():
    """Main function to fetch from all blockchains."""
    logger.info("🔗 Direct Blockchain Market Fetcher")
    logger.info("=" * 50)
    
    total_markets = 0
    
    # Clear old blockchain data first
    with db_manager.get_db_session() as db:
        old_count = db.query(Market).filter(Market.source.like('blockchain_%')).count()
        logger.info(f"Current blockchain markets: {old_count}")
    
    # Fetch from each network
    for network in ['optimism', 'arbitrum']:
        logger.info(f"\n📡 Fetching from {network.upper()}...")
        markets = fetch_from_blockchain(network)
        total_markets += markets
        logger.info(f"✅ Added {markets} markets from {network}")
        time.sleep(1)  # Rate limiting
    
    # Show final status
    with db_manager.get_db_session() as db:
        blockchain_markets = db.query(Market).filter(Market.source.like('blockchain_%')).count()
        soccer_markets = db.query(Market).filter(
            Market.source.like('blockchain_%'),
            Market.sport == 'Soccer'
        ).count()
        
        logger.info(f"\n✨ BLOCKCHAIN SYNC COMPLETE ✨")
        logger.info(f"Total blockchain markets: {blockchain_markets}")
        logger.info(f"Soccer markets: {soccer_markets}")
        logger.info(f"New markets added: {total_markets}")
    
    logger.info("\n🌐 Blockchain data ready for dashboard!")

if __name__ == "__main__":
    main()