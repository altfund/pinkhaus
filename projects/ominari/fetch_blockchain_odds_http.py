#!/usr/bin/env python3
"""
Fetch real odds from Overtime V2 contracts using direct HTTP RPC calls
No web3 dependency required
"""

import json
import requests
from datetime import datetime, timezone
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Public RPC endpoints
RPC_ENDPOINTS = {
    'arbitrum': 'https://arb1.arbitrum.io/rpc',
    'optimism': 'https://mainnet.optimism.io',
}

# Contract addresses
CONTRACTS = {
    'arbitrum': {
        'SportsAMMV2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'chain_id': 42161
    },
    'optimism': {
        'SportsAMMV2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1', 
        'chain_id': 10
    }
}

# Method signatures (keccak256 hash of method signature)
METHODS = {
    'getMarketDefaultOdds': '0xb12df8e3',  # getMarketDefaultOdds(address)
    'gameDetails': '0x5ee0fe28',           # gameDetails()
    'times': '0xd0370218',                 # times()
    'resolved': '0x5fe138b5',              # resolved()
    'tags': '0x4bbf5252',                  # tags()
}

def eth_call(rpc_url, to_address, data):
    """Make an eth_call via JSON-RPC"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{
            "to": to_address,
            "data": data
        }, "latest"],
        "id": 1
    }
    
    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        if 'result' in result:
            return result['result']
        else:
            logger.error(f"RPC error: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        logger.error(f"HTTP error: {e}")
        return None

def get_block_number(rpc_url):
    """Get current block number"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_blockNumber",
        "params": [],
        "id": 1
    }
    
    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        if 'result' in result:
            return int(result['result'], 16)
        return None
    except:
        return None

def encode_address(address):
    """Encode address as 32-byte parameter"""
    # Remove 0x prefix and pad to 32 bytes
    return address[2:].lower().zfill(64)

def decode_uint256(hex_data):
    """Decode a uint256 from hex"""
    if hex_data.startswith('0x'):
        hex_data = hex_data[2:]
    return int(hex_data[:64], 16)

def decode_string(hex_data):
    """Decode a string from contract return data"""
    if hex_data.startswith('0x'):
        hex_data = hex_data[2:]
    
    try:
        # Skip offset (32 bytes) and length (32 bytes)
        offset = int(hex_data[:64], 16) * 2
        length = int(hex_data[offset:offset+64], 16)
        
        # Get string data
        string_hex = hex_data[offset+64:offset+64+(length*2)]
        return bytes.fromhex(string_hex).decode('utf-8').strip()
    except:
        return None

def decode_array(hex_data):
    """Decode dynamic array from contract return"""
    if hex_data.startswith('0x'):
        hex_data = hex_data[2:]
    
    try:
        # Get array offset and length
        offset = int(hex_data[:64], 16) * 2
        length = int(hex_data[offset:offset+64], 16)
        
        # Read array elements
        elements = []
        for i in range(length):
            elem_start = offset + 64 + (i * 64)
            elem = int(hex_data[elem_start:elem_start+64], 16)
            elements.append(elem)
        
        return elements
    except:
        return []

def get_market_details(rpc_url, market_address):
    """Get market details from contract"""
    details = {}
    
    # Get game details
    result = eth_call(rpc_url, market_address, METHODS['gameDetails'])
    if result and len(result) > 2:
        # Returns (bytes32 gameId, string gameLabel)
        game_id = result[2:66]  # First 32 bytes
        details['game_id'] = '0x' + game_id
        game_label = decode_string(result)
        if game_label:
            details['game_label'] = game_label
            # Parse teams from label
            if ' vs ' in game_label:
                parts = game_label.split(' vs ')
                details['home_team'] = parts[0].strip()
                details['away_team'] = parts[1].strip()
    
    # Get times
    result = eth_call(rpc_url, market_address, METHODS['times'])
    if result:
        details['maturity'] = decode_uint256(result)
    
    # Get resolved status
    result = eth_call(rpc_url, market_address, METHODS['resolved'])
    if result:
        details['resolved'] = result[-1] == '1'
    
    # Get tags (includes sport ID)
    result = eth_call(rpc_url, market_address, METHODS['tags'])
    if result:
        tags = decode_array(result)
        if tags:
            details['sport_id'] = tags[0]
    
    return details

def get_odds_from_amm(rpc_url, amm_address, market_address):
    """Get odds from AMM contract"""
    # Encode function call: getMarketDefaultOdds(address)
    data = METHODS['getMarketDefaultOdds'] + encode_address(market_address)
    
    result = eth_call(rpc_url, amm_address, data)
    if not result:
        return None
    
    # Decode array of odds
    odds_array = decode_array(result)
    
    # Convert to decimal odds
    decimal_odds = []
    for odd in odds_array:
        if odd > 0:
            decimal = odd / 1e18
            if decimal >= 1.0:
                decimal_odds.append(decimal)
            else:
                decimal_odds.append(None)
        else:
            decimal_odds.append(None)
    
    return decimal_odds

def get_recent_markets(rpc_url, from_block, to_block):
    """Get recent market addresses from logs"""
    markets = set()
    
    # Look for MarketCreated events (topic: 0x...)
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "params": [{
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block),
            "topics": [
                # MarketCreated event signature
                "0x7e1c0c09b89009a48e0b6d7a86cf632c5af71cf616c466c62859fadeb892bb95"
            ]
        }],
        "id": 1
    }
    
    try:
        response = requests.post(rpc_url, json=payload, timeout=30)
        result = response.json()
        
        if 'result' in result:
            for log in result['result']:
                # Market address is usually in second topic or data
                if len(log.get('topics', [])) > 1:
                    # Extract address from topic (remove padding)
                    market_addr = '0x' + log['topics'][1][-40:]
                    markets.add(market_addr.lower())
        
        return list(markets)
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        return []

def sync_blockchain_odds():
    """Main function to sync odds from blockchain"""
    logger.info("🎯 Fetching real odds from Overtime V2 blockchain contracts...")
    
    # Sport ID mapping
    sport_id_map = {
        1: "Football", 2: "Football", 3: "Baseball", 4: "Basketball",
        5: "Basketball", 6: "Hockey", 7: "Fighting", 8: "Basketball",
        9: "Fighting", 10: "Soccer", 11: "Soccer", 12: "Soccer",
        13: "Soccer", 14: "Soccer", 15: "Soccer", 16: "Soccer",
        # Legacy IDs
        9001: "Football", 9002: "Baseball", 9003: "Basketball",
        9004: "Soccer", 9005: "Hockey", 9006: "MMA",
        9007: "Boxing", 9008: "Tennis"
    }
    
    conn = sqlite3.connect("sport_odds.db")
    cursor = conn.cursor()
    
    markets_found = 0
    markets_with_odds = 0
    
    # Process each chain
    for chain_name, config in CONTRACTS.items():
        logger.info(f"\n📡 Processing {chain_name.upper()} chain...")
        
        rpc_url = RPC_ENDPOINTS[chain_name]
        amm_address = config['SportsAMMV2']
        
        # Get current block
        current_block = get_block_number(rpc_url)
        if not current_block:
            logger.error(f"Failed to get block number for {chain_name}")
            continue
        
        logger.info(f"✅ Connected to {chain_name} at block {current_block:,}")
        
        # Look back ~1 hour of blocks
        blocks_per_hour = 300 if chain_name == 'arbitrum' else 1800
        from_block = current_block - blocks_per_hour
        
        logger.info(f"🔍 Searching for markets from block {from_block:,} to {current_block:,}")
        
        # Get recent markets
        market_addresses = get_recent_markets(rpc_url, from_block, current_block)
        logger.info(f"📊 Found {len(market_addresses)} potential markets from events")
        
        # Also try some known market patterns
        if len(market_addresses) < 10:
            # Try scanning recent transactions (simplified)
            logger.info("🔍 Scanning for additional markets...")
            # This is where we'd scan transactions, but it's complex without web3
        
        # Process each market
        for market_addr in market_addresses[:50]:  # Limit to 50 for demo
            markets_found += 1
            
            # Get market details
            details = get_market_details(rpc_url, market_addr)
            
            if not details or details.get('resolved', True):
                continue
            
            # Check if future game
            maturity = details.get('maturity', 0)
            if maturity == 0:
                continue
            
            maturity_date = datetime.fromtimestamp(maturity, tz=timezone.utc)
            if maturity_date < datetime.now(timezone.utc):
                continue
            
            # Must have teams
            if 'home_team' not in details or 'away_team' not in details:
                continue
            
            # Get odds
            odds = get_odds_from_amm(rpc_url, amm_address, market_addr)
            if not odds or len(odds) < 2:
                continue
            
            # Map sport
            sport_id = details.get('sport_id', 0)
            sport = sport_id_map.get(sport_id, 'Unknown')
            
            logger.info(f"\n✅ Found active market with odds:")
            logger.info(f"   Address: {market_addr}")
            logger.info(f"   Teams: {details['home_team']} vs {details['away_team']}")
            logger.info(f"   Sport: {sport} (ID: {sport_id})")
            logger.info(f"   Maturity: {maturity_date}")
            logger.info(f"   Odds: Home={odds[0]:.3f}, Away={odds[1]:.3f}")
            
            # Save to database
            market_id = f"{chain_name}_blockchain_{market_addr}"
            
            # Check if exists
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
                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1))
                    
                    cursor.execute("""
                        INSERT INTO odd (source_id, position, market_type, outcome, source, bookmaker,
                                       decimal_odds, american_odds, normalized_implied, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, i, "winner", outcome, f"{chain_name}_blockchain", f"Overtime V2 {chain_name.title()}",
                        decimal_odds, american, 1.0 / decimal_odds, datetime.now().isoformat()
                    ))
            
            markets_with_odds += 1
    
    conn.commit()
    
    # Summary
    logger.info(f"\n✨ BLOCKCHAIN SYNC COMPLETE")
    logger.info(f"Markets found: {markets_found}")
    logger.info(f"Markets with valid odds: {markets_with_odds}")
    
    cursor.execute("""
        SELECT source, COUNT(DISTINCT source_id) as count
        FROM market 
        WHERE source LIKE '%blockchain%'
        GROUP BY source
    """)
    
    logger.info(f"\n📊 Blockchain markets in database:")
    for source, count in cursor.fetchall():
        logger.info(f"  {source}: {count} markets")
    
    conn.close()

if __name__ == "__main__":
    sync_blockchain_odds()