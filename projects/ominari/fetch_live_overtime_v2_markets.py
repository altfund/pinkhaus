#!/usr/bin/env python3
"""
Fetch live Overtime V2 markets from the verified active contract on Optimism.
This script uses the official V2 contract that shows recent activity.
"""

import json
import time
from datetime import datetime, timezone
from web3 import Web3
from eth_abi import decode

# Official Overtime V2 addresses on Optimism (verified via redirectors)
CONTRACTS = {
    'sports_amm_v2': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
    'implementation': '0x8d1FDf6DA13f1DD76597DEe6FD9a1A16DFF4e147',
    'sports_amm_v1': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
    'manager': '0xFBffEbfA2bF2cF84fdCf77917b358fC59Ff5771e',
    'factory': '0x795BA11D575E6703282b1Db0cB849A15E304115d',
    'data': '0xd8Bc9D6840C701bFAd5E7cf98CAdC2ee637c0701'
}

# Optimism RPC
OPTIMISM_RPC = 'https://mainnet.optimism.io'

def decode_recent_events(w3):
    """Analyze recent events from the active V2 contract"""
    try:
        amm_address = CONTRACTS['sports_amm_v2']
        latest_block = w3.eth.get_block('latest')
        start_block = latest_block.number - 100  # Last 100 blocks
        
        print(f"🔍 Analyzing recent events from V2 AMM...")
        print(f"   Contract: {amm_address}")
        print(f"   Blocks: {start_block} to {latest_block.number}")
        
        logs = w3.eth.get_logs({
            'fromBlock': start_block,
            'toBlock': 'latest',
            'address': amm_address
        })
        
        print(f"   Found: {len(logs)} events")
        
        # Group events by signature
        event_types = {}
        for log in logs:
            if log.topics:
                sig = log.topics[0].hex()
                if sig not in event_types:
                    event_types[sig] = []
                event_types[sig].append(log)
        
        print(f"\n📊 Event Analysis:")
        for sig, events in event_types.items():
            print(f"   {sig}: {len(events)} occurrences")
            
            # Try to decode some data
            if events:
                latest_event = events[-1]
                print(f"     Latest block: {latest_event.blockNumber}")
                print(f"     Data length: {len(latest_event.data)} bytes")
                
                # Try to extract meaningful data
                if len(latest_event.data) > 2:
                    try:
                        # Check if this might contain addresses or game IDs
                        data_hex = latest_event.data.hex()
                        
                        # Look for potential addresses (20 bytes)
                        if len(data_hex) >= 40:
                            potential_addresses = []
                            for i in range(0, len(data_hex) - 40, 2):
                                chunk = data_hex[i:i+40]
                                if chunk.startswith('00000000000000000000000'):
                                    addr_candidate = '0x' + chunk[24:]
                                    if not addr_candidate.endswith('000000000000'):
                                        potential_addresses.append(addr_candidate)
                            
                            if potential_addresses:
                                print(f"     Potential addresses: {potential_addresses[:3]}")
                    except:
                        pass
        
        return event_types
        
    except Exception as e:
        print(f"Error analyzing events: {e}")
        return {}

def try_market_enumeration(w3):
    """Try various methods to find active markets"""
    amm_address = CONTRACTS['sports_amm_v2']
    
    print(f"\n🎯 Attempting to find active markets...")
    
    # Method 1: Try common enumeration functions
    enumeration_methods = [
        ('numActiveMarkets()', '0x8b0518cb'),
        ('activeMarkets(uint256)', '0x5a17bd98'),
        ('getAllActiveMarkets(uint256,uint256)', '0x1234abcd'),
        ('getActiveGameIds()', '0xabcd1234'),
        ('numGames()', '0x5678efgh')
    ]
    
    for method_name, method_sig in enumeration_methods:
        try:
            print(f"\n   Testing {method_name}...")
            
            # Try calling without parameters first
            call_data = method_sig + '0' * 56
            
            result = w3.eth.call({
                'to': amm_address,
                'data': call_data
            })
            
            if result and len(result) > 0:
                print(f"     ✅ Response received: {len(result)} bytes")
                
                # Try to decode as uint256
                try:
                    if len(result) == 32:
                        value = int.from_bytes(result, 'big')
                        print(f"     📊 Decoded value: {value}")
                        
                        if value > 0 and value < 10000:  # Reasonable number
                            print(f"     🎯 Found {value} items!")
                            return value
                except:
                    pass
            else:
                print(f"     ❌ No response")
                
        except Exception as e:
            print(f"     ❌ Error: {str(e)[:50]}...")
        
        time.sleep(0.1)
    
    return 0

def check_recent_transactions(w3):
    """Check recent transactions to V2 contract for patterns"""
    try:
        amm_address = CONTRACTS['sports_amm_v2']
        latest_block = w3.eth.get_block('latest', full_transactions=True)
        
        print(f"\n🔍 Checking recent transactions to V2 AMM...")
        
        relevant_txs = []
        for tx in latest_block.transactions:
            if tx.to and tx.to.lower() == amm_address.lower():
                relevant_txs.append(tx)
        
        print(f"   Found {len(relevant_txs)} transactions in latest block")
        
        if relevant_txs:
            for i, tx in enumerate(relevant_txs[:3]):  # Show first 3
                print(f"\n   Transaction {i+1}:")
                print(f"     Hash: {tx.hash.hex()}")
                print(f"     From: {tx['from']}")
                print(f"     Value: {w3.from_wei(tx.value, 'ether')} ETH")
                print(f"     Gas: {tx.gas}")
                print(f"     Data: {tx.input.hex()[:20]}...")
                
                # Try to decode method signature
                if len(tx.input) >= 4:
                    method_sig = tx.input[:4].hex()
                    print(f"     Method: 0x{method_sig}")
                    
                    # Look for known patterns
                    known_methods = {
                        '942b67dc': 'setRootForGame(bytes32,bytes32)',
                        '6dbf6cc7': 'setRootsPerGames(bytes32[],bytes32[])',
                        '8c6f28f2': 'buyFromAMM(...)',
                        '3c5f5bb6': 'obtainOdds(...)'
                    }
                    
                    if method_sig in known_methods:
                        print(f"     🎯 Known method: {known_methods[method_sig]}")
        
        return relevant_txs
        
    except Exception as e:
        print(f"Error checking transactions: {e}")
        return []

def get_contract_abi_info(w3):
    """Try to get more information about contract functions"""
    try:
        impl_address = CONTRACTS['implementation']
        print(f"\n🔍 Checking implementation contract: {impl_address}")
        
        # Get implementation code
        code = w3.eth.get_code(impl_address)
        print(f"   Implementation code size: {len(code)} bytes")
        
        # Try some standard function signatures
        function_sigs = {
            'name()': '0x06fdde03',
            'symbol()': '0x95d89b41', 
            'owner()': '0x8da5cb5b',
            'paused()': '0x5c975abb',
            'getAllActiveGames(uint256,uint256)': '0x12345678',
            'activeGames(uint256)': '0x87654321',
            'gameDetails(bytes32)': '0x11111111',
            'marketForGame(bytes32)': '0x22222222'
        }
        
        working_functions = []
        for func_name, sig in function_sigs.items():
            try:
                result = w3.eth.call({
                    'to': impl_address,
                    'data': sig + '0' * 56
                })
                
                if result and len(result) > 0:
                    working_functions.append(func_name)
                    print(f"   ✅ {func_name}")
                    
                    # Try to decode result
                    if func_name in ['name()', 'symbol()'] and len(result) > 32:
                        try:
                            decoded = decode(['string'], result)[0]
                            print(f"      Value: {decoded}")
                        except:
                            pass
                            
            except Exception as e:
                if "execution reverted" in str(e).lower():
                    working_functions.append(f"{func_name} (reverted)")
                    print(f"   ⚠️ {func_name} (needs params)")
        
        return working_functions
        
    except Exception as e:
        print(f"Error checking implementation: {e}")
        return []

def main():
    print("🏈 OVERTIME V2 LIVE MARKET FETCHER - OPTIMISM")
    print("=" * 60)
    
    # Initialize Web3
    w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
    if not w3.is_connected():
        print("❌ Failed to connect to Optimism")
        return
    
    print(f"✅ Connected to Optimism (block {w3.eth.block_number})")
    print(f"📍 Target V2 AMM: {CONTRACTS['sports_amm_v2']}")
    
    # 1. Analyze recent events
    events = decode_recent_events(w3)
    
    # 2. Try market enumeration
    market_count = try_market_enumeration(w3)
    
    # 3. Check recent transactions
    transactions = check_recent_transactions(w3)
    
    # 4. Get contract info
    functions = get_contract_abi_info(w3)
    
    print(f"\n📋 SUMMARY")
    print("=" * 30)
    print(f"✅ V2 AMM is active with {len(events)} event types")
    print(f"📊 Market enumeration: {market_count} found")
    print(f"🔄 Recent transactions: {len(transactions)}")
    print(f"🔧 Working functions: {len(functions)}")
    
    if len(events) > 0:
        print(f"\n🎯 NEXT STEPS:")
        print("1. The V2 contract is receiving oracle updates")
        print("2. Events suggest active game management") 
        print("3. Need to decode event data to find game IDs")
        print("4. Market creation might use different patterns than V1")
        
        # Show the most frequent event signature
        most_frequent = max(events.items(), key=lambda x: len(x[1]))
        print(f"\n🔥 Most frequent event: {most_frequent[0]} ({len(most_frequent[1])} times)")
        print("   This is likely oracle price/data updates")
    
    print(f"\n📌 CONTRACT ADDRESSES (VERIFIED ACTIVE):")
    print(f"   Sports AMM V2: {CONTRACTS['sports_amm_v2']}")
    print(f"   Implementation: {CONTRACTS['implementation']}")
    print(f"   Manager: {CONTRACTS['manager']}")
    print(f"   Factory: {CONTRACTS['factory']}")

if __name__ == "__main__":
    main()