#!/usr/bin/env python3
"""
Find active Overtime V2 markets on blockchain by scanning recent transactions
"""

import json
import requests
from datetime import datetime, timezone
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# RPC endpoints
RPC_ENDPOINTS = {
    'arbitrum': 'https://arb1.arbitrum.io/rpc',
    'optimism': 'https://mainnet.optimism.io',
}

# Known Overtime V2 contracts
CONTRACTS = {
    'arbitrum': {
        'SportsAMMV2': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'SportsAMMV2Manager': '0x7465c5d60d3d095443D26d0C2eD3F8Dc5c609C54',
        'Manager': '0xB155685132eEd3cD848d220e25a9607DD8871D38',
    },
    'optimism': {
        'SportsAMMV2': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'SportsAMMV2Manager': '0x278B5A44397c9D8E52743fEdec263c4760dc1A1A',
        'Manager': '0xFb0Dc7c4e8F184e1cbE37c70d90fE616Cd23F123',
    }
}

def eth_call(rpc_url, to_address, data):
    """Make eth_call"""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": to_address, "data": data}, "latest"],
        "id": 1
    }
    
    try:
        response = requests.post(rpc_url, json=payload, timeout=10)
        result = response.json()
        return result.get('result')
    except:
        return None

def get_recent_transactions(rpc_url, address, blocks_back=100):
    """Get recent transactions to a contract"""
    markets = set()
    
    # Get current block
    payload = {"jsonrpc": "2.0", "method": "eth_blockNumber", "params": [], "id": 1}
    response = requests.post(rpc_url, json=payload, timeout=10)
    current_block = int(response.json()['result'], 16)
    
    logger.info(f"Scanning last {blocks_back} blocks for transactions...")
    
    # Scan recent blocks
    for i in range(blocks_back):
        if i % 10 == 0:
            print(f"  Progress: {i}/{blocks_back} blocks", end='\r')
        
        block_num = current_block - i
        
        # Get block with transactions
        payload = {
            "jsonrpc": "2.0",
            "method": "eth_getBlockByNumber",
            "params": [hex(block_num), True],
            "id": 1
        }
        
        try:
            response = requests.post(rpc_url, json=payload, timeout=5)
            block = response.json().get('result', {})
            
            for tx in block.get('transactions', []):
                # Check if transaction is to our AMM
                if tx.get('to', '').lower() == address.lower():
                    input_data = tx.get('input', '')
                    
                    # Look for market addresses in input data
                    # buyFromAMM and similar methods have market address as first parameter
                    if len(input_data) >= 74:  # 0x + 8 chars method + 64 chars address
                        # Extract potential address
                        potential_addr = '0x' + input_data[34:74]
                        if len(potential_addr) == 42:
                            markets.add(potential_addr.lower())
        except:
            pass
    
    print()  # Clear progress line
    return list(markets)

def decode_string(hex_data):
    """Decode string from hex"""
    if hex_data.startswith('0x'):
        hex_data = hex_data[2:]
    
    try:
        offset = int(hex_data[:64], 16) * 2
        length = int(hex_data[offset:offset+64], 16)
        string_hex = hex_data[offset+64:offset+64+(length*2)]
        return bytes.fromhex(string_hex).decode('utf-8').strip()
    except:
        return None

def get_market_info(rpc_url, market_address):
    """Get basic market info"""
    # gameDetails() - 0x5ee0fe28
    result = eth_call(rpc_url, market_address, '0x5ee0fe28')
    if result:
        game_label = decode_string(result)
        if game_label and ' vs ' in game_label:
            parts = game_label.split(' vs ')
            return {
                'home_team': parts[0].strip(),
                'away_team': parts[1].strip(),
                'address': market_address
            }
    return None

def get_odds_from_amm(rpc_url, amm_address, market_address):
    """Get odds for market"""
    # getMarketDefaultOdds(address) - 0xb12df8e3
    data = '0xb12df8e3' + market_address[2:].zfill(64)
    
    result = eth_call(rpc_url, amm_address, data)
    if not result or result == '0x':
        return None
    
    # Decode array
    try:
        hex_data = result[2:]  # Remove 0x
        offset = int(hex_data[:64], 16) * 2
        length = int(hex_data[offset:offset+64], 16)
        
        odds = []
        for i in range(min(length, 3)):  # Get up to 3 odds
            elem_start = offset + 64 + (i * 64)
            odd_wei = int(hex_data[elem_start:elem_start+64], 16)
            if odd_wei > 0:
                decimal_odd = odd_wei / 1e18
                if decimal_odd >= 1.0:
                    odds.append(decimal_odd)
        
        return odds if len(odds) >= 2 else None
    except:
        return None

def main():
    """Find and display active markets with odds"""
    logger.info("🎯 Finding active Overtime V2 markets on blockchain...")
    
    conn = sqlite3.connect("sport_odds.db")
    cursor = conn.cursor()
    
    total_markets = 0
    markets_with_odds = 0
    
    for chain_name, contracts in CONTRACTS.items():
        logger.info(f"\n📡 Processing {chain_name.upper()}...")
        
        rpc_url = RPC_ENDPOINTS[chain_name]
        amm_address = contracts['SportsAMMV2']
        
        # Find market addresses from recent transactions
        markets = get_recent_transactions(rpc_url, amm_address, blocks_back=50)
        logger.info(f"Found {len(markets)} potential market addresses")
        
        # Check each market
        for market_addr in markets[:20]:  # Limit to 20 for testing
            total_markets += 1
            
            # Get market info
            info = get_market_info(rpc_url, market_addr)
            if not info:
                continue
            
            # Get odds
            odds = get_odds_from_amm(rpc_url, amm_address, market_addr)
            if odds:
                markets_with_odds += 1
                
                logger.info(f"\n✅ Active market found:")
                logger.info(f"   Chain: {chain_name}")
                logger.info(f"   Address: {market_addr}")
                logger.info(f"   Game: {info['home_team']} vs {info['away_team']}")
                logger.info(f"   Odds: Home={odds[0]:.3f}, Away={odds[1]:.3f}")
                
                # Save to database
                market_id = f"{chain_name}_blockchain_live_{market_addr}"
                
                # Update or insert market
                cursor.execute("""
                    INSERT OR REPLACE INTO market (
                        source_id, source, sport, league_name, market_type,
                        home_team, away_team, is_finished, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    market_id, f"{chain_name}_blockchain_live", "Unknown",
                    f"Overtime V2 {chain_name.title()}", "winner",
                    info['home_team'], info['away_team'], 0,
                    datetime.now().isoformat()
                ))
                
                # Update odds
                cursor.execute("DELETE FROM odd WHERE source_id = ?", (market_id,))
                
                for i, (outcome, decimal_odds) in enumerate([('home', odds[0]), ('away', odds[1])]):
                    american = int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1))
                    
                    cursor.execute("""
                        INSERT INTO odd (source_id, position, market_type, outcome, source, 
                                       bookmaker, decimal_odds, american_odds, normalized_implied, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        market_id, i, "winner", outcome, f"{chain_name}_blockchain_live",
                        f"Overtime V2", decimal_odds, american, 1.0 / decimal_odds,
                        datetime.now().isoformat()
                    ))
    
    conn.commit()
    
    logger.info(f"\n✨ SUMMARY:")
    logger.info(f"Total markets checked: {total_markets}")
    logger.info(f"Markets with valid odds: {markets_with_odds}")
    
    # Show blockchain markets
    cursor.execute("""
        SELECT source, sport, COUNT(*) as count 
        FROM market 
        WHERE source LIKE '%blockchain%'
        GROUP BY source, sport
        ORDER BY count DESC
        LIMIT 10
    """)
    
    logger.info(f"\n📊 Top blockchain markets in database:")
    for source, sport, count in cursor.fetchall():
        logger.info(f"  {source} ({sport}): {count} markets")
    
    conn.close()

if __name__ == "__main__":
    main()