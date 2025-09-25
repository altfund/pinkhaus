#!/usr/bin/env python3
"""
Investigate V2 Manager contract to find markets
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from eth_utils import function_signature_to_4byte_selector
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market, Odd
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# V2 Manager methods to try
MANAGER_METHODS = [
    # Market enumeration
    'markets(uint256)',
    'allMarkets(uint256)', 
    'activeMarkets(uint256)',
    'marketCount()',
    'numMarkets()',
    'totalMarkets()',
    'getMarket(uint256)',
    'getMarkets()',
    'getActiveMarkets()',
    'getMarketsBySport(uint256)',
    
    # Market info
    'isActiveMarket(address)',
    'marketInfo(address)',
    'getMarketData(address)',
    
    # Common patterns
    'owner()',
    'ammV2()',
    'sportsAMM()',
]

def try_method_call(w3, address, method_signature, param=None):
    """Try calling a method and return result."""
    try:
        selector = '0x' + function_signature_to_4byte_selector(method_signature).hex()
        
        if param is not None:
            # Encode uint256 parameter
            call_data = selector + str(param).rjust(64, '0')
        else:
            call_data = selector
            
        result = w3.eth.call({
            'to': address,
            'data': call_data
        })
        
        return result
    except:
        return None

def decode_address(data):
    """Decode an address from return data."""
    if data and len(data) == 32 and data[:12] == b'\x00' * 12:
        return '0x' + data[12:].hex()
    return None

def decode_number(data):
    """Decode a number from return data."""
    if data and len(data) == 32:
        return int.from_bytes(data, 'big')
    return None

def get_market_info_raw(w3, market_address):
    """Get basic market info using raw calls."""
    info = {'address': market_address}
    
    # Known selectors from previous work
    selectors = {
        'getGameDetails': '0x5ee0fe28',
        'times': '0xd0370218',
        'resolved': '0x5fe138b5',
        'tags': '0x4bbf5252',
        'homeTeam': '0x8de859d8',
        'awayTeam': '0x36c78516',
    }
    
    # Try getGameDetails
    result = try_method_call(w3, market_address, 'getGameDetails()')
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
    
    # Get times
    result = try_method_call(w3, market_address, 'times()')
    if result and len(result) >= 64:
        maturity = decode_number(result[:32])
        if maturity:
            info['maturity'] = maturity
    
    # Get resolved status
    result = try_method_call(w3, market_address, 'resolved()')
    if result:
        info['resolved'] = result[-1] == 1
        
    return info

def investigate_manager():
    """Investigate the V2 Manager contract."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    manager_address = '0xB155685132eEd3cD848d220e25a9607DD8871D38'
    amm_address = '0xfb64E79A562F7250131cf528242CEB10fDC82395'
    
    logger.info("🎯 Investigating Overtime V2 Manager")
    logger.info("=" * 60)
    logger.info(f"Manager: {manager_address}")
    
    # First, try to find how many markets exist
    logger.info("\n🔢 Looking for market count...")
    
    count_methods = ['marketCount()', 'numMarkets()', 'totalMarkets()']
    market_count = None
    
    for method in count_methods:
        result = try_method_call(w3, manager_address, method)
        if result:
            count = decode_number(result)
            if count is not None and count > 0:
                logger.info(f"  ✅ {method}: {count} markets")
                market_count = count
                break
    
    # Try to enumerate markets
    logger.info("\n🔍 Looking for market enumeration...")
    
    markets_found = []
    enum_methods = ['markets(uint256)', 'allMarkets(uint256)', 'activeMarkets(uint256)', 'getMarket(uint256)']
    
    # Try different methods to get markets
    for method in enum_methods:
        logger.info(f"\nTrying {method}...")
        found_any = False
        
        # Try first 20 indices
        for i in range(20):
            result = try_method_call(w3, manager_address, method, i)
            if result:
                addr = decode_address(result)
                if addr and addr != '0x' + '0' * 40:
                    # Verify it's a contract
                    code = w3.eth.get_code(addr)
                    if code and len(code) > 0:
                        logger.info(f"  [#{i}] Found market: {addr}")
                        markets_found.append(addr)
                        found_any = True
                        
                        # Get basic info
                        info = get_market_info_raw(w3, addr)
                        if info.get('gameLabel'):
                            logger.info(f"       Game: {info['gameLabel']}")
                        if info.get('maturity'):
                            maturity_date = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                            logger.info(f"       Date: {maturity_date}")
                        if 'resolved' in info:
                            logger.info(f"       Resolved: {info['resolved']}")
                            
        if found_any:
            logger.info(f"\n✅ Success with {method}! Found {len(markets_found)} markets")
            break
    
    # If we found markets, add them to database
    if markets_found:
        logger.info(f"\n💾 Processing {len(markets_found)} markets for database...")
        
        markets_added = 0
        
        for market_addr in markets_found:
            try:
                info = get_market_info_raw(w3, market_addr)
                
                # Skip if missing critical info
                if not info.get('homeTeam') or not info.get('maturity'):
                    continue
                    
                # Skip resolved markets
                if info.get('resolved', False):
                    continue
                    
                # Check if future game
                maturity = datetime.fromtimestamp(info['maturity'], tz=timezone.utc)
                if maturity < datetime.now(timezone.utc):
                    continue
                    
                market_id = f"arbitrum_v2_manager_{market_addr.lower()[-8:]}"
                
                with db_manager.get_db_session() as db:
                    # Check if already exists
                    if db.query(Market).filter(Market.source_id == market_id).first():
                        continue
                        
                    # Create market
                    market = Market(
                        source_id=market_id,
                        source="arbitrum_v2_manager",
                        sport="Soccer",  # Default, would need to decode from tags
                        league_name="Overtime V2",
                        market_type="winner",
                        home_team=info.get('homeTeam', 'Unknown'),
                        away_team=info.get('awayTeam', 'Unknown'),
                        maturity_date=maturity,
                        is_finished=False,
                        updated_at=datetime.now(timezone.utc)
                    )
                    db.add(market)
                    db.flush()
                    
                    # Try to get odds from AMM
                    # Using getMarketDefaultOdds(address,bool)
                    selector = '0xb12df8e3'
                    # Encode: address + bool (false)
                    encoded_market = market_addr[2:].lower().rjust(64, '0')
                    encoded_bool = '0' * 64  # false
                    call_data = selector + encoded_market + encoded_bool
                    
                    try:
                        result = w3.eth.call({
                            'to': amm_address,
                            'data': call_data
                        })
                        
                        if result and len(result) >= 96:
                            # Decode array
                            offset = int.from_bytes(result[0:32], 'big')
                            length = int.from_bytes(result[offset:offset+32], 'big')
                            
                            if length >= 2:
                                home_odds = int.from_bytes(result[offset+64:offset+96], 'big')
                                away_odds = int.from_bytes(result[offset+96:offset+128], 'big')
                                
                                odds_map = {}
                                if home_odds > 0:
                                    odds_map[0] = home_odds / 1e18
                                if away_odds > 0:
                                    odds_map[1] = away_odds / 1e18
                                    
                                if length >= 3:
                                    draw_odds = int.from_bytes(result[offset+128:offset+160], 'big')
                                    if draw_odds > 0:
                                        odds_map[2] = draw_odds / 1e18
                                        
                                # Add odds to database
                                for position, decimal_odds in odds_map.items():
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
                    except:
                        pass
                        
                    db.commit()
                    markets_added += 1
                    logger.info(f"\n✅ Added market: {info.get('gameLabel', 'Unknown')}")
                    
            except Exception as e:
                logger.error(f"Error processing market {market_addr}: {e}")
                continue
                
        # Summary
        logger.info(f"\n✨ Manager Investigation Complete ✨")
        logger.info(f"Markets found: {len(markets_found)}")
        logger.info(f"Markets added to DB: {markets_added}")
        
        if markets_added > 0:
            # Clear sample data
            with db_manager.get_db_session() as db:
                logger.info("\n🧹 Clearing sample data...")
                sample_markets = db.query(Market).filter(Market.source.like('%sample%')).all()
                for market in sample_markets:
                    db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                    db.delete(market)
                db.commit()
                logger.info(f"Removed {len(sample_markets)} sample markets")
                
            logger.info("\n🎆 SUCCESS! Found real Overtime V2 markets via Manager!")
            logger.info("Dashboard at http://localhost:8888/unified now shows REAL blockchain data!")
            
    else:
        logger.info("\n⚠️  No markets found via Manager")
        logger.info("Possible reasons:")
        logger.info("  1. Different enumeration method needed")
        logger.info("  2. Markets stored differently in V2")
        logger.info("  3. Need to check events or logs")

if __name__ == "__main__":
    investigate_manager()