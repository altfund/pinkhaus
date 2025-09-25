#!/usr/bin/env python3
"""
Decode V2 AMM method signatures and analyze contract structure
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
import requests
import json
from eth_utils import function_signature_to_4byte_selector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Known V2 AMM methods we should look for
POSSIBLE_METHODS = [
    # Market enumeration
    'activeMarkets(uint256)',
    'activeMarketsPerSport(uint256,uint256)', 
    'numActiveMarkets()',
    'getActiveMarketAddress(uint256)',
    'getActiveMarketsBySport(uint256)',
    
    # Trading
    'buyFromAMM(address,uint256,uint256,uint256,uint256)',
    'buyFromAMMWithDifferentCollateral(address,uint256,uint256,uint256,uint256,address)',
    'sellToAMM(address,uint256,uint256,uint256,uint256)',
    
    # Market info
    'getMarketInfo(address)',
    'getMarketData(address)',
    'getMarketOdds(address)',
    'getMarketDefaultOdds(address,bool)',
    
    # V2 specific
    'tradeData()',
    'liquidityPool()',
    'manager()',
    'riskManager()',
    
    # Common patterns
    'markets(uint256)',
    'marketCount()',
    'allMarkets(uint256)',
    'getMarket(uint256)',
    'getMarkets()',
]

def get_method_signature(selector):
    """Try to decode method signature from 4byte.directory."""
    try:
        url = f"https://www.4byte.directory/api/v1/signatures/?hex_signature={selector}"
        response = requests.get(url, timeout=5)
        data = response.json()
        
        if data['results']:
            return data['results'][0]['text_signature']
    except:
        pass
    return None

def check_storage_patterns(w3, address):
    """Check storage slots for common patterns."""
    logger.info("\n💾 Analyzing storage patterns...")
    
    patterns = {}
    
    # Check first 50 storage slots
    for slot in range(50):
        try:
            data = w3.eth.get_storage_at(address, slot)
            if data != b'\x00' * 32:
                # Check if it's an address
                if data[:12] == b'\x00' * 12:
                    potential_addr = '0x' + data[12:].hex()
                    try:
                        code = w3.eth.get_code(potential_addr)
                        if code and len(code) > 0:
                            patterns[slot] = {
                                'type': 'address',
                                'value': potential_addr,
                                'has_code': True,
                                'code_size': len(code)
                            }
                            logger.info(f"  Slot {slot}: Contract at {potential_addr} ({len(code)} bytes)")
                    except:
                        pass
                else:
                    # Could be a number
                    value = int.from_bytes(data, 'big')
                    if value > 0 and value < 1000000:  # Reasonable range
                        patterns[slot] = {'type': 'number', 'value': value}
                        logger.info(f"  Slot {slot}: Number = {value}")
                    elif value > 1600000000 and value < 2000000000:  # Timestamp range
                        patterns[slot] = {'type': 'timestamp', 'value': value}
                        logger.info(f"  Slot {slot}: Timestamp = {value}")
        except:
            pass
            
    return patterns

def try_common_calls(w3, address):
    """Try common getter methods."""
    logger.info("\n🔍 Trying common getter methods...")
    
    results = {}
    
    # Generate selectors for common methods
    test_methods = [
        ('manager()', '0x481c6a75'),
        ('owner()', '0x8da5cb5b'),
        ('markets(uint256)', None),  # Will need parameter
        ('numActiveMarkets()', None),
        ('activeMarkets(uint256)', None),
        ('marketCount()', None),
        ('allMarkets(uint256)', None),
        ('tradeData()', None),
        ('liquidityPool()', None),
        ('riskManager()', None),
    ]
    
    for method_name, known_selector in test_methods:
        try:
            if known_selector:
                selector = known_selector
            else:
                # Calculate selector
                selector = '0x' + function_signature_to_4byte_selector(method_name).hex()
                
            # Make the call
            if '(uint256)' in method_name:
                # Try with parameter 0
                call_data = selector + '0' * 64
            else:
                call_data = selector
                
            result = w3.eth.call({
                'to': address,
                'data': call_data
            })
            
            if result and len(result) > 0:
                # Try to interpret as address
                if len(result) == 32 and result[:12] == b'\x00' * 12:
                    addr = '0x' + result[12:].hex()
                    logger.info(f"  {method_name}: {addr}")
                    results[method_name] = addr
                else:
                    logger.info(f"  {method_name}: {result.hex()}")
                    results[method_name] = result.hex()
                    
        except Exception as e:
            # Method doesn't exist or reverted
            pass
            
    return results

def analyze_v2_structure():
    """Analyze V2 AMM structure and find market access methods."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    amm_address = '0xfb64E79A562F7250131cf528242CEB10fDC82395'
    
    logger.info("🔍 Analyzing Overtime V2 AMM Structure")
    logger.info("=" * 60)
    
    # First, check the unknown signatures we found
    unknown_sigs = ['0x942b67dc', '0x6dbf6cc7']
    
    logger.info("\n📝 Decoding unknown method signatures...")
    for sig in unknown_sigs:
        decoded = get_method_signature(sig)
        if decoded:
            logger.info(f"  {sig}: {decoded}")
        else:
            # Try matching against our known methods
            for method in POSSIBLE_METHODS:
                try:
                    calc_sig = '0x' + function_signature_to_4byte_selector(method).hex()
                    if calc_sig == sig:
                        logger.info(f"  {sig}: {method} ✅")
                        break
                except:
                    pass
            else:
                logger.info(f"  {sig}: Unknown")
    
    # Check storage patterns
    storage_patterns = check_storage_patterns(w3, amm_address)
    
    # Try common calls
    call_results = try_common_calls(w3, amm_address)
    
    # If we found a manager or similar contract, investigate it
    if 'manager()' in call_results:
        manager_addr = call_results['manager()']
        logger.info(f"\n🎯 Found manager at: {manager_addr}")
        logger.info("Investigating manager contract...")
        
        # Check manager storage
        manager_patterns = check_storage_patterns(w3, manager_addr)
        
    # Look for market enumeration patterns
    logger.info("\n🔍 Looking for market enumeration methods...")
    
    # Try array-style access
    for i in range(10):
        for method_pattern in ['markets(uint256)', 'activeMarkets(uint256)', 'allMarkets(uint256)']:
            try:
                selector = '0x' + function_signature_to_4byte_selector(method_pattern).hex()
                # Encode parameter i
                param = str(i).rjust(64, '0')
                call_data = selector + param
                
                result = w3.eth.call({
                    'to': amm_address,
                    'data': call_data
                })
                
                if result and len(result) == 32:
                    # Check if it's an address
                    if result[:12] == b'\x00' * 12:
                        market_addr = '0x' + result[12:].hex()
                        # Verify it's a contract
                        code = w3.eth.get_code(market_addr)
                        if code and len(code) > 0:
                            logger.info(f"\n✅ Found market via {method_pattern}[{i}]: {market_addr}")
                            # Try to get basic info
                            try:
                                # Try getGameDetails
                                game_selector = '0x5ee0fe28'
                                game_result = w3.eth.call({
                                    'to': market_addr,
                                    'data': game_selector
                                })
                                if game_result:
                                    logger.info(f"   Has getGameDetails(): Yes")
                            except:
                                pass
                            break
            except:
                pass
                
    # Summary
    logger.info("\n📊 V2 Structure Summary:")
    logger.info(f"AMM Address: {amm_address}")
    if storage_patterns:
        logger.info("\nNotable storage slots:")
        for slot, info in storage_patterns.items():
            if info['type'] == 'address':
                logger.info(f"  Slot {slot}: {info['value']} (contract)")
            elif info['type'] == 'number':
                logger.info(f"  Slot {slot}: {info['value']} (number)")
                
    if call_results:
        logger.info("\nSuccessful method calls:")
        for method, result in call_results.items():
            logger.info(f"  {method}: {result}")
            
    logger.info("\n💡 Next steps:")
    logger.info("1. If manager found, enumerate markets through manager")
    logger.info("2. Check transaction logs for market creation events")
    logger.info("3. Analyze internal transactions from AMM")

if __name__ == "__main__":
    analyze_v2_structure()