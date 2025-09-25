#!/usr/bin/env python3
"""
Fetch real odds from Overtime V2 contracts on blockchain
Connects API game IDs to blockchain market addresses
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
import json
from web3 import Web3
from datetime import datetime, timezone
import sqlite3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Public RPC endpoints (no API key needed)
RPC_ENDPOINTS = {
    'arbitrum': 'https://arb1.arbitrum.io/rpc',
    'optimism': 'https://mainnet.optimism.io',
    'base': 'https://mainnet.base.org'
}

# Overtime V2 contract addresses
OVERTIME_V2_CONTRACTS = {
    'arbitrum': {
        'SportsAMMV2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'Manager': '0xB155685132eEd3cD848d220e25a9607DD8871D38',
        'ResultManager': '0x0f73602224F01669372639E3638c08D61EA8895E'
    },
    'optimism': {
        'SportsAMMV2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'Manager': '0xFb0Dc7c4e8F184e1cbE37c70d90fE616Cd23F123',
        'ResultManager': '0x5E1b40E4249644f4253dD17aFdD33f10B064b058'
    }
}

# Contract ABIs (minimal, just what we need)
SPORTS_AMM_ABI = [
    {
        "inputs": [{"name": "_market", "type": "address"}],
        "name": "getMarketDefaultOdds",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "type": "function"
    },
    {
        "inputs": [{"name": "_market", "type": "address"}, {"name": "_position", "type": "uint8"}],
        "name": "buyFromAMM",
        "outputs": [],
        "type": "function"
    }
]

MARKET_ABI = [
    {
        "inputs": [],
        "name": "gameDetails",
        "outputs": [
            {"name": "gameId", "type": "bytes32"},
            {"name": "gameLabel", "type": "string"}
        ],
        "type": "function"
    },
    {
        "inputs": [],
        "name": "times",
        "outputs": [
            {"name": "maturity", "type": "uint256"},
            {"name": "destruction", "type": "uint256"}
        ],
        "type": "function"
    },
    {
        "inputs": [],
        "name": "resolved",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function"
    },
    {
        "inputs": [],
        "name": "tags",
        "outputs": [{"name": "", "type": "uint256[]"}],
        "type": "function"
    },
    {
        "inputs": [],
        "name": "homeTeam",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    },
    {
        "inputs": [],
        "name": "awayTeam",
        "outputs": [{"name": "", "type": "string"}],
        "type": "function"
    }
]

def connect_to_chain(chain_name):
    """Connect to blockchain via RPC"""
    rpc_url = RPC_ENDPOINTS.get(chain_name)
    if not rpc_url:
        return None
    
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
        if w3.is_connected():
            logger.info(f"✅ Connected to {chain_name} at block {w3.eth.block_number:,}")
            return w3
        else:
            logger.error(f"Failed to connect to {chain_name}")
            return None
    except Exception as e:
        logger.error(f"Error connecting to {chain_name}: {e}")
        return None

def get_market_details(w3, market_address):
    """Get market details from contract"""
    try:
        market = w3.eth.contract(address=Web3.to_checksum_address(market_address), abi=MARKET_ABI)
        
        # Get game details
        try:
            game_id, game_label = market.functions.gameDetails().call()
            game_id_hex = game_id.hex()
        except:
            game_id_hex = None
            game_label = None
        
        # Get teams
        try:
            home_team = market.functions.homeTeam().call()
            away_team = market.functions.awayTeam().call()
        except:
            # Try parsing from game label
            if game_label and ' vs ' in game_label:
                parts = game_label.split(' vs ')
                home_team = parts[0].strip()
                away_team = parts[1].strip()
            else:
                return None
        
        # Get times
        try:
            maturity, _ = market.functions.times().call()
        except:
            maturity = 0
        
        # Get resolved status
        try:
            resolved = market.functions.resolved().call()
        except:
            resolved = False
        
        # Get sport tag
        try:
            tags = market.functions.tags().call()
            sport_id = tags[0] if tags else None
        except:
            sport_id = None
        
        return {
            'game_id': game_id_hex,
            'home_team': home_team,
            'away_team': away_team,
            'maturity': maturity,
            'resolved': resolved,
            'sport_id': sport_id
        }
    except Exception as e:
        logger.debug(f"Error getting market details: {e}")
        return None

def get_odds_from_amm(w3, amm_address, market_address):
    """Get odds from SportsAMM contract"""
    try:
        amm = w3.eth.contract(address=Web3.to_checksum_address(amm_address), abi=SPORTS_AMM_ABI)
        
        # Get default odds for all positions
        odds = amm.functions.getMarketDefaultOdds(Web3.to_checksum_address(market_address)).call()
        
        # Convert from wei to decimal odds
        decimal_odds = []
        for odd in odds:
            if odd > 0:
                # Odds are in 18 decimals
                decimal = odd / 1e18
                if decimal >= 1.0:
                    decimal_odds.append(decimal)
                else:
                    decimal_odds.append(None)
            else:
                decimal_odds.append(None)
        
        return decimal_odds
    except Exception as e:
        logger.debug(f"Error getting odds: {e}")
        return None

def find_markets_from_logs(w3, amm_address, from_block, to_block):
    """Find market addresses from AMM events"""
    markets = set()
    
    try:
        # Look for NewMarket or MarketCreated events
        # Using raw logs since we don't have full ABI
        logs = w3.eth.get_logs({
            'address': amm_address,
            'fromBlock': from_block,
            'toBlock': to_block
        })
        
        for log in logs:
            # Market addresses often appear as first topic or in data
            if len(log['topics']) > 1:
                # Try second topic (often the market address)
                try:
                    market_addr = '0x' + log['topics'][1].hex()[26:]  # Remove padding
                    if len(market_addr) == 42:  # Valid address length
                        markets.add(market_addr)
                except:
                    pass
        
        return list(markets)
    except Exception as e:
        logger.error(f"Error finding markets from logs: {e}")
        return []

def sync_blockchain_odds():
    """Main function to sync odds from blockchain"""
    logger.info("🎯 Fetching real odds from Overtime V2 blockchain contracts...")
    
    # First, get game mappings from API
    logger.info("📊 Getting game data from API...")
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/games-info', timeout=10)
        api_games = response.json()
        logger.info(f"✅ Found {len(api_games)} games from API")
    except Exception as e:
        logger.error(f"Failed to get API data: {e}")
        return
    
    # Get sport mappings
    try:
        response = requests.get('https://api.overtime.io/overtime-v2/sports', timeout=10)
        sport_mappings = response.json()
        logger.info(f"✅ Loaded {len(sport_mappings)} sport definitions")
    except:
        sport_mappings = {}
    
    # Create sport ID to name mapping
    sport_id_map = {}
    for sid, info in sport_mappings.items():
        sport_id_map[int(sid)] = info.get('sport', 'Unknown')
    
    # Add legacy mappings (9000+ IDs from blockchain)
    sport_id_map.update({
        9001: "Football",
        9002: "Baseball", 
        9003: "Basketball",
        9004: "Soccer",
        9005: "Hockey",
        9006: "MMA",
        9007: "Boxing",
        9008: "Tennis"
    })
    
    conn = sqlite3.connect("sport_odds.db")
    cursor = conn.cursor()
    
    markets_with_odds = 0
    total_markets_checked = 0
    
    # Process each chain
    for chain_name, contracts in OVERTIME_V2_CONTRACTS.items():
        logger.info(f"\n📡 Processing {chain_name.upper()} chain...")
        
        w3 = connect_to_chain(chain_name)
        if not w3:
            continue
        
        amm_address = contracts['SportsAMMV2']
        current_block = w3.eth.block_number
        
        # Look back ~1 day of blocks (varies by chain)
        blocks_per_day = 7200 if chain_name == 'arbitrum' else 43200
        from_block = current_block - blocks_per_day
        
        logger.info(f"🔍 Searching for markets from block {from_block:,} to {current_block:,}")
        
        # Find markets from logs
        market_addresses = find_markets_from_logs(w3, amm_address, from_block, current_block)
        logger.info(f"📊 Found {len(market_addresses)} potential markets")
        
        # Process each market
        for market_addr in market_addresses:
            total_markets_checked += 1
            
            # Get market details
            details = get_market_details(w3, market_addr)
            if not details or details['resolved']:
                continue
            
            # Check if it's a future game
            if details['maturity'] > 0:
                maturity_date = datetime.fromtimestamp(details['maturity'], tz=timezone.utc)
                if maturity_date < datetime.now(timezone.utc):
                    continue
            else:
                continue
            
            # Get odds
            odds = get_odds_from_amm(w3, amm_address, market_addr)
            if not odds or len(odds) < 2:
                continue
            
            # Map to sport
            sport = sport_id_map.get(details.get('sport_id'), 'Unknown')
            
            # Create unique market ID
            market_id = f"{chain_name}_blockchain_{market_addr.lower()}"
            
            logger.info(f"\n✅ Found active market with odds:")
            logger.info(f"   Address: {market_addr}")
            logger.info(f"   Teams: {details['home_team']} vs {details['away_team']}")
            logger.info(f"   Sport: {sport}")
            logger.info(f"   Maturity: {maturity_date}")
            logger.info(f"   Odds: Home={odds[0]:.3f}, Away={odds[1]:.3f}, Draw={odds[2] if len(odds) > 2 else 'N/A'}")
            
            # Check if market exists
            cursor.execute("SELECT source_id FROM market WHERE source_id = ?", (market_id,))
            if not cursor.fetchone():
                # Insert market
                cursor.execute("""
                    INSERT INTO market (source_id, source, sport, league_name, market_type,
                                      home_team, away_team, maturity_date, is_finished, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, f"{chain_name}_blockchain", sport, f"Overtime V2 {chain_name.title()}",
                    "winner", details['home_team'], details['away_team'], 
                    maturity_date.isoformat(), 0, datetime.now().isoformat()
                ))
            
            # Update odds
            cursor.execute("DELETE FROM odd WHERE source_id = ?", (market_id,))
            
            for i, (outcome, decimal_odds) in enumerate([('home', odds[0]), ('away', odds[1])]):
                if decimal_odds and decimal_odds >= 1.0:
                    # Convert to American odds
                    if decimal_odds >= 2.0:
                        american = int((decimal_odds - 1) * 100)
                    else:
                        american = int(-100 / (decimal_odds - 1))
                    
                    cursor.execute("""
                        INSERT INTO odd (source_id, position, market_type, outcome, source, bookmaker,
                                       decimal_odds, american_odds, normalized_implied, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, i, "winner", outcome, f"{chain_name}_blockchain", f"Overtime V2 {chain_name.title()}",
                        decimal_odds, american, 1.0 / decimal_odds, datetime.now().isoformat()
                    ))
            
            # Add draw odds if available
            if len(odds) > 2 and odds[2] and odds[2] >= 1.0:
                decimal_odds = odds[2]
                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1))
                
                cursor.execute("""
                    INSERT INTO odd (source_id, position, market_type, outcome, source, bookmaker,
                                   decimal_odds, american_odds, normalized_implied, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, 2, "winner", "draw", f"{chain_name}_blockchain", f"Overtime V2 {chain_name.title()}",
                    decimal_odds, american, 1.0 / decimal_odds, datetime.now().isoformat()
                ))
            
            markets_with_odds += 1
            
            if markets_with_odds % 10 == 0:
                conn.commit()
    
    conn.commit()
    
    # Summary
    cursor.execute("""
        SELECT source, COUNT(DISTINCT source_id) as count
        FROM market 
        WHERE source LIKE '%blockchain%'
        GROUP BY source
    """)
    
    logger.info(f"\n✨ BLOCKCHAIN SYNC COMPLETE")
    logger.info(f"Total markets checked: {total_markets_checked}")
    logger.info(f"Markets with odds found: {markets_with_odds}")
    logger.info(f"\n📊 Blockchain markets in database:")
    for source, count in cursor.fetchall():
        logger.info(f"  {source}: {count} markets")
    
    conn.close()

if __name__ == "__main__":
    sync_blockchain_odds()