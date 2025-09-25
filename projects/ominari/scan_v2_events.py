#!/usr/bin/env python3
"""
Scan V2 contracts for events to find market creation patterns
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common event signatures
EVENT_SIGNATURES = {
    # Market creation
    'MarketCreated': Web3.keccak(text='MarketCreated(address,bytes32,string,uint256)').hex(),
    'NewSportsMarket': Web3.keccak(text='NewSportsMarket(address,bytes32,string,uint256)').hex(),
    'CreateMarket': Web3.keccak(text='CreateMarket(address)').hex(),
    'MarketAdded': Web3.keccak(text='MarketAdded(address)').hex(),
    
    # Trading events (might contain market addresses)
    'BoughtFromAMM': Web3.keccak(text='BoughtFromAMM(address,address,uint8,uint256,uint256,uint256,uint256)').hex(),
    'SoldToAMM': Web3.keccak(text='SoldToAMM(address,address,uint8,uint256,uint256,uint256,uint256)').hex(),
    
    # Generic
    'Transfer': Web3.keccak(text='Transfer(address,address,uint256)').hex(),
}

def scan_for_events(w3, contract_address, from_block, to_block):
    """Scan a contract for all events in a block range."""
    events = []
    
    try:
        logs = w3.eth.get_logs({
            'address': contract_address,
            'fromBlock': from_block,
            'toBlock': to_block
        })
        
        for log in logs:
            events.append({
                'address': log['address'],
                'topics': [topic.hex() for topic in log['topics']],
                'data': log['data'],
                'blockNumber': log['blockNumber'],
                'transactionHash': log['transactionHash'].hex()
            })
            
    except Exception as e:
        logger.error(f"Error scanning events: {e}")
        
    return events

def decode_address_from_topic(topic):
    """Decode an address from a topic (32 bytes with padding)."""
    if len(topic) == 66:  # 0x + 64 chars
        return '0x' + topic[-40:]
    return None

def analyze_v2_events():
    """Analyze V2 contract events to find market creation patterns."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    
    contracts = {
        'AMM': Web3.to_checksum_address('0xfb64E79A562F7250131cf528242CEB10fDC82395'),
        'Manager': Web3.to_checksum_address('0xB155685132eEd3cD848d220e25a9607DD8871D38'),
        'RiskManager': Web3.to_checksum_address('0x10764f2787841e928e53e5be1588a73e3c994ede')
    }
    
    logger.info("🔍 Analyzing Overtime V2 Events")
    logger.info("=" * 60)
    
    current_block = w3.eth.block_number
    scan_blocks = 10000  # Last ~10000 blocks
    from_block = current_block - scan_blocks
    
    all_events = {}
    potential_markets = set()
    
    # Scan each contract
    for name, address in contracts.items():
        logger.info(f"\n📡 Scanning {name} at {address}...")
        
        events = scan_for_events(w3, address, from_block, current_block)
        all_events[name] = events
        
        logger.info(f"  Found {len(events)} events")
        
        # Analyze event patterns
        event_types = {}
        for event in events:
            if event['topics']:
                sig = event['topics'][0]
                event_types[sig] = event_types.get(sig, 0) + 1
                
                # Look for potential market addresses in topics
                for topic in event['topics'][1:]:
                    addr = decode_address_from_topic(topic)
                    if addr and addr != '0x' + '0' * 40:
                        # Check if it's a contract
                        code = w3.eth.get_code(addr)
                        if code and len(code) > 0 and len(code) != 45:  # Not a minimal proxy
                            potential_markets.add(addr)
                            
        # Display event types
        logger.info("\n  Event signatures found:")
        for sig, count in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:10]:
            known = None
            for event_name, known_sig in EVENT_SIGNATURES.items():
                if sig == known_sig:
                    known = event_name
                    break
            
            if known:
                logger.info(f"    {sig[:10]}... = {known} ({count} times)")
            else:
                logger.info(f"    {sig[:10]}... = Unknown ({count} times)")
                
                # For the most common unknown event, check if it has market addresses
                if count > 100 and name == 'AMM':
                    sample_events = [e for e in events if e['topics'] and e['topics'][0] == sig][:5]
                    for e in sample_events:
                        if len(e['topics']) >= 2:
                            # Check second topic (often first indexed parameter)
                            addr = decode_address_from_topic(e['topics'][1])
                            if addr:
                                code = w3.eth.get_code(addr)
                                if code and len(code) == 45:  # Minimal proxy!
                                    logger.info(f"      → Contains market: {addr}")
                
    # Check BoughtFromAMM events specifically
    logger.info("\n💰 Analyzing BoughtFromAMM events for market addresses...")
    
    amm_events = all_events.get('AMM', [])
    bought_events = [e for e in amm_events if e['topics'] and e['topics'][0] == EVENT_SIGNATURES.get('BoughtFromAMM', '')]
    
    verified_markets = []  # Initialize here
    
    if bought_events:
        logger.info(f"  Found {len(bought_events)} BoughtFromAMM events")
        
        # Extract market addresses from these events
        markets_from_trades = set()
        for event in bought_events[:20]:  # Sample first 20
            # First indexed parameter after event signature is buyer
            # Second indexed parameter is the market
            if len(event['topics']) >= 3:
                market_addr = decode_address_from_topic(event['topics'][2])
                if market_addr:
                    markets_from_trades.add(market_addr)
                    
        logger.info(f"  Found {len(markets_from_trades)} unique markets from trades")
        
        # Verify these are actual markets
        verified_markets = []
        for addr in list(markets_from_trades)[:10]:  # Check first 10
            code = w3.eth.get_code(addr)
            if code and len(code) == 45:  # Minimal proxy!
                verified_markets.append(addr)
                logger.info(f"    ✅ Verified market: {addr}")
                
        # Get details for verified markets
        if verified_markets:
            logger.info("\n🏆 Getting details for verified markets...")
            
            markets_added = 0
            
            for market_addr in verified_markets:
                try:
                    # Get basic info using raw calls
                    info = get_market_info_raw(w3, market_addr)
                    
                    if info.get('gameLabel'):
                        logger.info(f"\n  Market: {market_addr}")
                        logger.info(f"  Game: {info['gameLabel']}")
                        if info.get('maturity'):
                            maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                            logger.info(f"  Date: {maturity}")
                            
                            # Add to database if future game
                            if maturity > datetime.now(timezone.utc) and not info.get('resolved', False):
                                market_id = f"arbitrum_v2_event_{market_addr.lower()[-8:]}"
                                
                                with db_manager.get_db_session() as db:
                                    if not db.query(Market).filter(Market.source_id == market_id).first():
                                        market = Market(
                                            source_id=market_id,
                                            source="arbitrum_v2_events",
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
                    
                logger.info(f"\n🎆 SUCCESS! Added {markets_added} real V2 markets from events!")
                logger.info("Dashboard at http://localhost:8888/unified now shows REAL data!")
                
    else:
        logger.info("\n⚠️  No BoughtFromAMM events found")
        
    # Summary
    logger.info("\n📊 Event Analysis Summary:")
    logger.info(f"Scanned last {scan_blocks} blocks")
    logger.info(f"Potential market addresses found: {len(potential_markets)}")
    if verified_markets:
        logger.info(f"Verified V2 markets: {len(verified_markets)}")
        
def get_market_info_raw(w3, market_address):
    """Get basic market info using raw calls."""
    info = {'address': market_address}
    
    # getGameDetails selector
    try:
        result = w3.eth.call({
            'to': market_address,
            'data': '0x5ee0fe28'
        })
        
        if result and len(result) > 96:
            # Decode game label
            try:
                offset = int.from_bytes(result[32:64], 'big')
                length = int.from_bytes(result[offset:offset+32], 'big')
                if length > 0 and length < 1000:
                    game_label = result[offset+32:offset+32+length].decode('utf-8', errors='ignore').strip()
                    if game_label:
                        info['gameLabel'] = game_label
                        # Parse teams
                        if ' vs ' in game_label:
                            parts = game_label.split(' vs ')
                            info['homeTeam'] = parts[0].strip()
                            info['awayTeam'] = parts[1].strip()
            except:
                pass
    except:
        pass
        
    # times() selector
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
        
    # resolved() selector
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
    analyze_v2_events()