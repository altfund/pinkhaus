#!/usr/bin/env python3
"""
Analyze the unknown events to understand V2 patterns
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def decode_address_from_data(data, position=0):
    """Decode an address from event data at specified position."""
    try:
        # Each parameter is 32 bytes
        start = 2 + (position * 64)  # Skip 0x
        end = start + 64
        if len(data) >= end:
            param = data[start:end]
            # Address is in the last 40 chars
            addr = '0x' + param[-40:]
            return Web3.to_checksum_address(addr)
    except:
        pass
    return None

def analyze_unknown_events():
    """Deep dive into the unknown events."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    amm_address = Web3.to_checksum_address('0xfb64E79A562F7250131cf528242CEB10fDC82395')
    
    logger.info("🔍 Deep Analysis of V2 AMM Events")
    logger.info("=" * 60)
    
    current_block = w3.eth.block_number
    from_block = current_block - 5000  # Last 5000 blocks
    
    # Get logs
    logs = w3.eth.get_logs({
        'address': amm_address,
        'fromBlock': from_block,
        'toBlock': current_block
    })
    
    logger.info(f"\nFound {len(logs)} total events")
    
    # Group by event signature
    events_by_sig = {}
    for log in logs:
        if log['topics']:
            sig = log['topics'][0].hex()
            if sig not in events_by_sig:
                events_by_sig[sig] = []
            events_by_sig[sig].append(log)
            
    # Analyze the most common event
    most_common_sig = max(events_by_sig.keys(), key=lambda k: len(events_by_sig[k]))
    events = events_by_sig[most_common_sig]
    
    logger.info(f"\n🎯 Most common event: {most_common_sig[:10]}...")
    logger.info(f"Count: {len(events)}")
    
    # Analyze structure
    logger.info("\n📊 Analyzing event structure...")
    
    sample_events = events[:10]
    markets_found = set()
    
    for i, event in enumerate(sample_events):
        logger.info(f"\nEvent #{i+1}:")
        logger.info(f"  Topics: {len(event['topics'])}")
        for j, topic in enumerate(event['topics']):
            topic_hex = topic.hex()
            logger.info(f"    Topic {j}: {topic_hex[:10]}...")
            
            # Try to decode as address (skip event signature)
            if j > 0:
                addr = '0x' + topic_hex[-40:]
                try:
                    addr = Web3.to_checksum_address(addr)
                    code = w3.eth.get_code(addr)
                    if code:
                        if len(code) == 45:
                            logger.info(f"      → Market contract! {addr}")
                            markets_found.add(addr)
                        elif len(code) > 100:
                            logger.info(f"      → Regular contract ({len(code)} bytes)")
                except:
                    pass
                    
        # Analyze data
        if event['data'] and len(event['data']) > 2:
            logger.info(f"  Data length: {len(event['data'])} chars")
            
            # Try to extract addresses from data
            for pos in range(10):  # Check first 10 positions
                addr = decode_address_from_data(event['data'], pos)
                if addr and addr != '0x' + '0' * 40:
                    try:
                        code = w3.eth.get_code(addr)
                        if code and len(code) == 45:
                            logger.info(f"    → Found market in data position {pos}: {addr}")
                            markets_found.add(addr)
                    except:
                        pass
                        
    # Process found markets
    if markets_found:
        logger.info(f"\n🎆 Found {len(markets_found)} potential V2 markets!")
        
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
                            market_id = f"arbitrum_v2_deep_{market_addr.lower()[-8:]}"
                            
                            with db_manager.get_db_session() as db:
                                if not db.query(Market).filter(Market.source_id == market_id).first():
                                    
                                    # Parse teams
                                    home_team = info.get('homeTeam', 'Unknown')
                                    away_team = info.get('awayTeam', 'Unknown')
                                    
                                    market = Market(
                                        source_id=market_id,
                                        source="arbitrum_v2_deep",
                                        sport="Soccer",
                                        league_name="Overtime V2",
                                        market_type="winner",
                                        home_team=home_team,
                                        away_team=away_team,
                                        maturity_date=maturity,
                                        is_finished=False,
                                        updated_at=datetime.now(timezone.utc)
                                    )
                                    db.add(market)
                                    
                                    # Try to get odds
                                    odds = get_odds_from_amm(w3, amm_address, market_addr)
                                    
                                    for position, decimal_odds in odds.items():
                                        if decimal_odds > 1.0:
                                            outcome_map = {0: 'home', 1: 'away', 2: 'draw'}
                                            outcome = outcome_map.get(position)
                                            
                                            if outcome:
                                                # Convert to American
                                                if decimal_odds >= 2.0:
                                                    american = int((decimal_odds - 1) * 100)
                                                else:
                                                    american = int(-100 / (decimal_odds - 1))
                                                    
                                                odd = Odd(
                                                    source_id=market.source_id,
                                                    market_type="winner",
                                                    outcome=outcome,
                                                    source="arbitrum_v2_amm",
                                                    bookmaker="Overtime V2",
                                                    decimal_odds=decimal_odds,
                                                    american_odds=american,
                                                    normalized_implied=1.0 / decimal_odds,
                                                    updated_at=datetime.now(timezone.utc)
                                                )
                                                db.add(odd)
                                                
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
        logger.info("\n⚠️  No markets found in events")
        
    # Try a different approach - check if events might be encoded differently
    logger.info("\n🔍 Checking alternative event encodings...")
    
    # The signature might be BoughtFromAMMWithDifferentCollateral or similar
    # Let's decode the data as if it's a trading event
    sample = events[0] if events else None
    if sample and sample['data']:
        logger.info("\nDecoding first event as potential trade:")
        data = sample['data']
        
        # Common structure: buyer, market, position, amount, sUSDPaid, sUSDAfterFees, timestamp
        for i in range(min(7, len(data) // 64)):
            param = decode_address_from_data(data, i)
            if param:
                logger.info(f"  Param {i}: {param}")
                
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

def get_odds_from_amm(w3, amm_address, market_address):
    """Get odds from AMM."""
    odds = {}
    
    # Try getMarketDefaultOdds(address,bool)
    selector = '0xb12df8e3'
    encoded_market = market_address[2:].lower().rjust(64, '0')
    encoded_bool = '0' * 64  # false
    call_data = selector + encoded_market + encoded_bool
    
    try:
        result = w3.eth.call({
            'to': amm_address,
            'data': call_data
        })
        
        if result and len(result) >= 96:
            offset = int.from_bytes(result[0:32], 'big')
            length = int.from_bytes(result[offset:offset+32], 'big')
            
            if length >= 2:
                home_odds = int.from_bytes(result[offset+64:offset+96], 'big')
                away_odds = int.from_bytes(result[offset+96:offset+128], 'big')
                
                if home_odds > 0:
                    odds[0] = home_odds / 1e18
                if away_odds > 0:
                    odds[1] = away_odds / 1e18
                    
                if length >= 3:
                    draw_odds = int.from_bytes(result[offset+128:offset+160], 'big')
                    if draw_odds > 0:
                        odds[2] = draw_odds / 1e18
    except:
        pass
        
    return odds

if __name__ == "__main__":
    analyze_unknown_events()