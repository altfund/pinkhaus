#!/usr/bin/env python3
"""
Update odds for existing blockchain markets
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

# AMM contracts
AMM_CONTRACTS = {
    'arbitrum': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
    'optimism': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
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
    """Update odds for existing blockchain markets"""
    logger.info("🎯 Updating odds for existing blockchain markets...")
    
    conn = sqlite3.connect("sport_odds.db")
    cursor = conn.cursor()
    
    # Get existing blockchain markets
    cursor.execute("""
        SELECT source_id, source, home_team, away_team
        FROM market 
        WHERE source LIKE '%blockchain%' 
        AND is_finished = 0
        ORDER BY updated_at DESC
        LIMIT 100
    """)
    
    markets = cursor.fetchall()
    logger.info(f"📊 Found {len(markets)} active blockchain markets to update")
    
    odds_updated = 0
    
    for source_id, source, home_team, away_team in markets:
        # Extract chain and address from source_id
        parts = source_id.split('_')
        
        if len(parts) < 3:
            continue
        
        chain = parts[0]
        address = parts[-1]
        
        if chain not in RPC_ENDPOINTS:
            continue
        
        if not address.startswith('0x') or len(address) != 42:
            continue
        
        # Get odds
        rpc_url = RPC_ENDPOINTS[chain]
        amm_address = AMM_CONTRACTS[chain]
        
        odds = get_odds_from_amm(rpc_url, amm_address, address)
        
        if odds and len(odds) >= 2:
            odds_updated += 1
            
            logger.info(f"\n✅ Updated odds for: {home_team} vs {away_team}")
            logger.info(f"   Chain: {chain}")
            logger.info(f"   Odds: Home={odds[0]:.3f}, Away={odds[1]:.3f}")
            
            # Update odds
            cursor.execute("DELETE FROM odd WHERE source_id = ?", (source_id,))
            
            for i, (outcome, decimal_odds) in enumerate([('home', odds[0]), ('away', odds[1])]):
                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1))
                
                cursor.execute("""
                    INSERT INTO odd (source_id, position, market_type, outcome, source, 
                                   bookmaker, decimal_odds, american_odds, normalized_implied, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source_id, i, "winner", outcome, f"{chain}_blockchain",
                    "Overtime V2", decimal_odds, american, 1.0 / decimal_odds,
                    datetime.now().isoformat()
                ))
            
            # Add draw odds if available
            if len(odds) > 2 and odds[2]:
                decimal_odds = odds[2]
                american = int((decimal_odds - 1) * 100) if decimal_odds >= 2.0 else int(-100 / (decimal_odds - 1))
                
                cursor.execute("""
                    INSERT INTO odd (source_id, position, market_type, outcome, source, 
                                   bookmaker, decimal_odds, american_odds, normalized_implied, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    source_id, 2, "winner", "draw", f"{chain}_blockchain",
                    "Overtime V2", decimal_odds, american, 1.0 / decimal_odds,
                    datetime.now().isoformat()
                ))
            
            # Update market timestamp
            cursor.execute(
                "UPDATE market SET updated_at = ? WHERE source_id = ?",
                (datetime.now().isoformat(), source_id)
            )
    
    conn.commit()
    
    logger.info(f"\n✨ SUMMARY:")
    logger.info(f"Markets checked: {len(markets)}")
    logger.info(f"Odds updated: {odds_updated}")
    
    # Show current odds
    cursor.execute("""
        SELECT m.home_team, m.away_team, m.sport, 
               o1.decimal_odds as home_odds, o2.decimal_odds as away_odds
        FROM market m
        JOIN odd o1 ON m.source_id = o1.source_id AND o1.outcome = 'home'
        JOIN odd o2 ON m.source_id = o2.source_id AND o2.outcome = 'away'
        WHERE m.source LIKE '%blockchain%' 
        AND o1.updated_at > datetime('now', '-1 hour')
        ORDER BY o1.updated_at DESC
        LIMIT 10
    """)
    
    logger.info(f"\n📊 Recent blockchain odds:")
    for home, away, sport, home_odds, away_odds in cursor.fetchall():
        logger.info(f"  {home} vs {away} ({sport}): {home_odds:.3f} / {away_odds:.3f}")
    
    conn.close()

if __name__ == "__main__":
    main()