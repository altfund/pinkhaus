#!/usr/bin/env python3
"""Connect blockchain and API data for Overtime V2"""

import os
os.environ['PG_PORT'] = '5999'

import requests
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chain configurations
CHAINS = {
    'optimism': {
        'id': 10,
        'rpc': 'https://mainnet.optimism.io',
        'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'name': 'Optimism'
    },
    'arbitrum': {
        'id': 42161,
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    }
}

def get_market_address_from_game_id(w3, amm_address, game_id):
    """Get market address for a game ID using getMarketAddress(bytes32)"""
    try:
        # getMarketAddress(bytes32) - function selector
        function_selector = Web3.keccak(text='getMarketAddress(bytes32)')[:4].hex()
        
        # Prepare the call data
        call_data = function_selector + game_id[2:]  # Remove 0x from game_id
        
        # Call the contract
        result = w3.eth.call({
            'to': amm_address,
            'data': call_data
        })
        
        if result and len(result) == 32:
            # Convert bytes32 to address (last 20 bytes)
            market_address = '0x' + result[-20:].hex()
            if market_address != '0x' + '0' * 40:
                return Web3.to_checksum_address(market_address)
                
    except Exception as e:
        logger.debug(f"Error getting market address: {e}")
        
    return None

def get_active_game_ids_from_chain(w3, amm_address):
    """Get active game IDs from blockchain events"""
    try:
        current_block = w3.eth.block_number
        from_block = current_block - 5000  # Last ~16 hours
        
        # GameRootUpdated event
        event_sig = Web3.keccak(text='GameRootUpdated(bytes32,bytes32)').hex()
        
        logs = w3.eth.get_logs({
            'address': amm_address,
            'fromBlock': hex(from_block),
            'toBlock': 'latest',
            'topics': [event_sig]
        })
        
        game_ids = set()
        for log in logs:
            if len(log['data']) >= 130:
                game_id = '0x' + log['data'][2:66]
                game_ids.add(game_id)
                
        return list(game_ids)
        
    except Exception as e:
        logger.error(f"Error getting game IDs from chain: {e}")
        return []

def connect_api_and_blockchain():
    """Connect API game data with blockchain market addresses"""
    
    # Get API data
    logger.info("Fetching games from API...")
    response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=30)
    api_games = response.json()
    
    logger.info(f"Found {len(api_games)} games in API")
    
    # Filter active games
    active_games = {}
    for game_id, info in api_games.items():
        if not info.get('isGameFinished', True):
            teams = info.get('teams', [])
            if len(teams) >= 2:
                active_games[game_id] = info
    
    logger.info(f"Found {len(active_games)} active games")
    
    # Connect to blockchains and find market addresses
    connected_markets = []
    
    for chain_name, config in CHAINS.items():
        logger.info(f"\n🔗 Connecting to {config['name']}...")
        
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if not w3.is_connected():
            logger.error(f"Failed to connect to {chain_name}")
            continue
            
        amm = Web3.to_checksum_address(config['amm'])
        
        # Get recent game IDs from blockchain
        blockchain_game_ids = get_active_game_ids_from_chain(w3, amm)
        logger.info(f"Found {len(blockchain_game_ids)} game IDs from blockchain events")
        
        # Try to match with API games
        matches_found = 0
        for game_id in blockchain_game_ids:
            if game_id in active_games:
                # We have a match! Get the market address
                market_address = get_market_address_from_game_id(w3, amm, game_id)
                if market_address:
                    game_info = active_games[game_id]
                    teams = game_info.get('teams', [])
                    home = next((t for t in teams if t.get('isHome')), None)
                    away = next((t for t in teams if not t.get('isHome')), None)
                    
                    if home and away:
                        connected_markets.append({
                            'chain': chain_name,
                            'game_id': game_id,
                            'market_address': market_address,
                            'home_team': home.get('name'),
                            'away_team': away.get('name'),
                            'tournament': game_info.get('tournamentName', ''),
                            'api_data': game_info
                        })
                        matches_found += 1
                        
                        logger.info(f"✅ Connected: {home.get('name')} vs {away.get('name')}")
                        logger.info(f"   Game ID: {game_id}")
                        logger.info(f"   Market: {market_address}")
        
        logger.info(f"Matched {matches_found} games between API and blockchain")
        
        # Also try for all API games
        logger.info("\nChecking API games for market addresses...")
        api_matches = 0
        for game_id, info in list(active_games.items())[:20]:  # Check first 20
            market_address = get_market_address_from_game_id(w3, amm, game_id)
            if market_address:
                teams = info.get('teams', [])
                home = next((t for t in teams if t.get('isHome')), None)
                away = next((t for t in teams if not t.get('isHome')), None)
                
                if home and away and not any(m['game_id'] == game_id for m in connected_markets):
                    connected_markets.append({
                        'chain': chain_name,
                        'game_id': game_id,
                        'market_address': market_address,
                        'home_team': home.get('name'),
                        'away_team': away.get('name'),
                        'tournament': info.get('tournamentName', ''),
                        'api_data': info
                    })
                    api_matches += 1
                    
        logger.info(f"Found {api_matches} additional markets from API games")
    
    return connected_markets

def save_connected_markets(markets):
    """Save markets with blockchain connection to database"""
    saved = 0
    
    with db_manager.get_db_session() as db:
        for market_data in markets:
            try:
                # Create unique ID with chain and game ID
                source_id = f"{market_data['chain']}_connected_{market_data['game_id']}"
                
                # Skip if exists
                existing = db.query(Market).filter(Market.source_id == source_id).first()
                if existing:
                    continue
                
                # Get maturity date from API data
                api_data = market_data['api_data']
                maturity = api_data.get('maturityDate', api_data.get('startTime'))
                if maturity:
                    try:
                        if isinstance(maturity, (int, float)):
                            maturity = datetime.fromtimestamp(maturity / 1000, tz=timezone.utc)
                        else:
                            maturity = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
                    except:
                        maturity = datetime.now(timezone.utc).replace(hour=20, minute=0)
                else:
                    maturity = datetime.now(timezone.utc).replace(hour=20, minute=0)
                
                # Create market
                market = Market(
                    source_id=source_id,
                    home_team=market_data['home_team'],
                    away_team=market_data['away_team'],
                    sport='Soccer',  # Will be mixed sports
                    league_name=market_data['tournament'],
                    maturity_date=maturity,
                    source=f"overtime_{market_data['chain']}_connected",
                    market_address=market_data['market_address'],
                    is_finished=False
                )
                db.add(market)
                db.commit()
                
                # Add default odds
                for outcome, odds_value in [('home', 2.5), ('draw', 3.0), ('away', 2.8)]:
                    odd = Odd(
                        source_id=source_id,
                        outcome=outcome,
                        decimal_odds=odds_value,
                        market_type='winner',
                        source=f"overtime_{market_data['chain']}_connected",
                        bookmaker='overtime',
                        normalized_implied=1.0/odds_value
                    )
                    db.add(odd)
                
                db.commit()
                saved += 1
                
            except Exception as e:
                logger.error(f"Error saving market: {e}")
                db.rollback()
                
    return saved

def main():
    logger.info("🔗 Connecting Overtime API and Blockchain Data")
    logger.info("=" * 60)
    
    # Find connections
    connected = connect_api_and_blockchain()
    
    if connected:
        logger.info(f"\n🎯 Found {len(connected)} connected markets!")
        
        # Save to database
        saved = save_connected_markets(connected)
        logger.info(f"\n✅ Saved {saved} connected markets")
        
        # Show examples
        logger.info("\nExamples of connected markets:")
        for market in connected[:5]:
            logger.info(f"\n{market['home_team']} vs {market['away_team']}")
            logger.info(f"  Chain: {market['chain']}")
            logger.info(f"  Game ID: {market['game_id']}")
            logger.info(f"  Market Address: {market['market_address']}")
    else:
        logger.info("\n❌ No connections found between API and blockchain")
        logger.info("This might be because:")
        logger.info("  1. Markets are on a different contract")
        logger.info("  2. Game IDs use a different encoding")
        logger.info("  3. Markets haven't been created on-chain yet")

if __name__ == "__main__":
    main()