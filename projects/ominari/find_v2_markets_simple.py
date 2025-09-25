#!/usr/bin/env python3
"""
Simple approach to find V2 markets by checking AMM logs
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import requests
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_recent_amm_markets():
    """Get markets from recent AMM interactions via logs."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    amm_address = Web3.to_checksum_address('0xfb64E79A562F7250131cf528242CEB10fDC82395')
    
    logger.info("🎯 Finding V2 Markets - Simple Approach")
    logger.info("=" * 60)
    
    current_block = w3.eth.block_number
    markets_found = set()
    
    # Check recent blocks for any logs from the AMM
    logger.info("\n📡 Scanning AMM logs for market addresses...")
    
    # Get logs from AMM
    from_block = current_block - 5000
    
    try:
        logs = w3.eth.get_logs({
            'address': amm_address,
            'fromBlock': from_block,
            'toBlock': current_block
        })
        
        logger.info(f"Found {len(logs)} logs from AMM")
        
        # Look through logs for addresses that might be markets
        for log in logs[:100]:  # Check first 100
            # Check topics for addresses (skip first topic which is event signature)
            for topic in log['topics'][1:]:
                addr_hex = topic.hex() if hasattr(topic, 'hex') else topic
                # Extract address from topic (last 40 chars)
                potential_addr = '0x' + addr_hex[-40:]
                
                try:
                    addr = Web3.to_checksum_address(potential_addr)
                    # Check if it's a minimal proxy (market)
                    code = w3.eth.get_code(addr)
                    if code and len(code) == 45:
                        markets_found.add(addr)
                except:
                    pass
                    
            # Also check log data for addresses
            if log['data']:
                data = log['data'].hex() if hasattr(log['data'], 'hex') else log['data']
                # Skip 0x
                data = data[2:] if data.startswith('0x') else data
                
                # Check each 32-byte chunk
                for i in range(0, len(data), 64):
                    chunk = data[i:i+64]
                    if len(chunk) == 64:
                        # Address in last 40 chars
                        potential_addr = '0x' + chunk[-40:]
                        try:
                            addr = Web3.to_checksum_address(potential_addr)
                            if addr != '0x' + '0' * 40:
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:
                                    markets_found.add(addr)
                        except:
                            pass
                            
    except Exception as e:
        logger.error(f"Error getting logs: {e}")
        
    logger.info(f"\n🎯 Found {len(markets_found)} potential markets from logs")
    
    # Alternative: Use Arbiscan API to get AMM transactions
    logger.info("\n🔍 Checking via Arbiscan API (no key required for basic calls)...")
    
    try:
        # Get recent transactions to AMM (limited without API key)
        url = f"https://api.arbiscan.io/api?module=account&action=txlist&address={amm_address}&startblock={current_block-1000}&endblock={current_block}&sort=desc"
        
        response = requests.get(url, timeout=10)
        data = response.json()
        
        if data.get('status') == '1' and data.get('result'):
            txs = data['result'][:20]  # Check first 20
            logger.info(f"Got {len(txs)} recent transactions")
            
            for tx in txs:
                # Decode input to find market addresses
                input_data = tx.get('input', '')
                if len(input_data) > 10:
                    # Skip method sig
                    data = input_data[10:]
                    
                    # Look for addresses in parameters
                    for i in range(0, len(data), 64):
                        param = data[i:i+64]
                        if len(param) == 64:
                            potential_addr = '0x' + param[-40:]
                            try:
                                addr = Web3.to_checksum_address(potential_addr)
                                if addr != '0x' + '0' * 40:
                                    code = w3.eth.get_code(addr)
                                    if code and len(code) == 45:
                                        markets_found.add(addr)
                                        logger.info(f"  Found market in tx: {addr}")
                            except:
                                pass
                                
    except Exception as e:
        logger.info(f"Arbiscan check failed (expected without API key): {e}")
        
    # Process found markets
    if markets_found:
        logger.info(f"\n🏆 Processing {len(markets_found)} markets...")
        
        markets_added = 0
        
        for market_addr in list(markets_found)[:10]:  # Process up to 10
            try:
                # Get market info
                info = get_market_info_raw(w3, market_addr)
                
                if info.get('gameLabel'):
                    logger.info(f"\n🏈 Market: {market_addr}")
                    logger.info(f"  Game: {info['gameLabel']}")
                    
                    if info.get('maturity'):
                        maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                        logger.info(f"  Date: {maturity}")
                        logger.info(f"  Resolved: {info.get('resolved', False)}")
                        
                        # Add to DB if future and not resolved
                        if maturity > datetime.now(timezone.utc) and not info.get('resolved', False):
                            market_id = f"arbitrum_v2_simple_{market_addr.lower()[-8:]}"
                            
                            with db_manager.get_db_session() as db:
                                if not db.query(Market).filter(Market.source_id == market_id).first():
                                    
                                    market = Market(
                                        source_id=market_id,
                                        source="arbitrum_v2_simple",
                                        sport="Soccer",
                                        league_name="Overtime V2",
                                        market_type="winner",
                                        home_team=info.get('homeTeam', 'Unknown'),
                                        away_team=info.get('awayTeam', 'Unknown'),
                                        maturity_date=maturity,
                                        is_finished=False,
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(market)
                                    db.commit()
                                    
                                    markets_added += 1
                                    logger.info("  ✅ Added to database!")
                                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                
        if markets_added > 0:
            # Clear sample data
            with db_manager.get_db_session() as db:
                sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
                for market in sample_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
                
            logger.info(f"\n🎆 SUCCESS! Added {markets_added} real V2 markets!")
            logger.info("Dashboard at http://localhost:8888/unified now shows REAL blockchain data!")
            
        else:
            logger.info("\n⚠️ No active future markets found")
            logger.info("This could be due to:")
            logger.info("  1. Off-season - no current sports events")
            logger.info("  2. All markets already resolved")
            logger.info("  3. Markets created through different mechanism")
            
    else:
        logger.info("\n⚠️ No markets found in AMM interactions")
        
    # Summary
    with db_manager.get_db_session() as db:
        total = db.query(Market).count()
        active = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).count()
        
        logger.info(f"\n📊 Database Summary:")
        logger.info(f"Total markets: {total}")
        logger.info(f"Active markets: {active}")
        
def get_market_info_raw(w3, market_address):
    """Get basic market info using raw calls."""
    info = {'address': market_address}
    
    # getGameDetails
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5ee0fe28'
        })
        
        if result and len(result) > 96:
            try:
                offset = int.from_bytes(result[32:64], 'big')
                length = int.from_bytes(result[offset:offset+32], 'big') 
                if length > 0 and length < 1000:
                    game_label = result[offset+32:offset+32+length].decode('utf-8', errors='ignore').strip()
                    if game_label:
                        info['gameLabel'] = game_label
                        if ' vs ' in game_label:
                            parts = game_label.split(' vs ')
                            info['homeTeam'] = parts[0].strip()
                            info['awayTeam'] = parts[1].strip()
                        elif ' @ ' in game_label:
                            parts = game_label.split(' @ ')
                            info['awayTeam'] = parts[0].strip()
                            info['homeTeam'] = parts[1].strip()
            except:
                pass
    except:
        pass
        
    # times
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0xd0370218'
        })
        if result and len(result) >= 32:
            maturity = int.from_bytes(result[:32], 'big')
            if maturity > 0:
                info['maturity'] = maturity
    except:
        pass
        
    # resolved
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5fe138b5'
        })
        if result:
            info['resolved'] = result[-1] == 1
    except:
        pass
        
    return info

if __name__ == "__main__":
    get_recent_amm_markets()