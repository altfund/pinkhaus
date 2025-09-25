#!/usr/bin/env python3
"""
Decode Overtime V2 events to extract game IDs and market information.
This analyzes the active events we found to understand the data structure.
"""

import json
from web3 import Web3
from eth_abi import decode
import struct

# Official Overtime V2 addresses on Optimism
SPORTS_AMM_V2 = '0xFb4e4811C7A811E098A556bD79B64c20b479E431'
OPTIMISM_RPC = 'https://mainnet.optimism.io'

def decode_event_data(event_data_hex, event_signature):
    """Try to decode event data based on signature patterns"""
    try:
        # Convert to string if it's bytes
        if isinstance(event_data_hex, bytes):
            event_data_hex = event_data_hex.hex()
        
        # Remove '0x' prefix if present
        if isinstance(event_data_hex, str) and event_data_hex.startswith('0x'):
            event_data_hex = event_data_hex[2:]
        
        data_bytes = bytes.fromhex(event_data_hex)
        
        print(f"\n🔍 Decoding event {event_signature[:10]}...")
        print(f"   Data length: {len(data_bytes)} bytes")
        print(f"   Raw hex: {event_data_hex[:40]}...")
        
        # Try different decoding strategies
        results = {}
        
        # Strategy 1: Look for game IDs (bytes32)
        if len(data_bytes) >= 32:
            potential_game_id = data_bytes[:32]
            results['potential_game_id'] = '0x' + potential_game_id.hex()
            
            # Check if it looks like a meaningful game ID
            if not all(b == 0 for b in potential_game_id):
                print(f"   🎯 Potential Game ID: {results['potential_game_id']}")
        
        # Strategy 2: Try to decode as (bytes32, bytes32) for oracle updates
        if len(data_bytes) == 64:
            try:
                game_id, merkle_root = decode(['bytes32', 'bytes32'], data_bytes)
                results['game_id'] = '0x' + game_id.hex()
                results['merkle_root'] = '0x' + merkle_root.hex()
                print(f"   📊 Game ID: {results['game_id']}")
                print(f"   🌳 Merkle Root: {results['merkle_root']}")
            except:
                pass
        
        # Strategy 3: Look for multiple game IDs (batch updates)
        if len(data_bytes) > 64 and len(data_bytes) % 32 == 0:
            try:
                num_items = len(data_bytes) // 32
                game_ids = []
                
                for i in range(min(num_items, 10)):  # Max 10 to avoid spam
                    start = i * 32
                    end = start + 32
                    item = data_bytes[start:end]
                    if not all(b == 0 for b in item):
                        game_ids.append('0x' + item.hex())
                
                if game_ids:
                    results['batch_game_ids'] = game_ids
                    print(f"   📦 Batch Game IDs: {len(game_ids)} items")
                    for i, gid in enumerate(game_ids[:3]):
                        print(f"      {i+1}: {gid}")
                    if len(game_ids) > 3:
                        print(f"      ... and {len(game_ids) - 3} more")
                        
            except:
                pass
        
        # Strategy 4: Look for addresses and amounts (trading events)
        if len(data_bytes) >= 96:  # Enough for address + 2 uint256
            try:
                # Try parsing as trading event
                chunks = []
                for i in range(0, min(len(data_bytes), 96), 32):
                    chunk = data_bytes[i:i+32]
                    chunks.append(chunk)
                
                # Check if first chunk looks like an address
                if len(chunks) >= 3:
                    potential_addr = chunks[0][-20:]  # Last 20 bytes
                    if not all(b == 0 for b in potential_addr):
                        addr = '0x' + potential_addr.hex()
                        results['potential_address'] = addr
                        print(f"   👤 Potential Address: {addr}")
                        
                        # Try to decode amounts
                        amount1 = int.from_bytes(chunks[1], 'big')
                        amount2 = int.from_bytes(chunks[2], 'big')
                        
                        if amount1 > 0 and amount1 < 10**30:  # Reasonable range
                            results['amount1'] = amount1
                            print(f"   💰 Amount 1: {amount1}")
                        
                        if amount2 > 0 and amount2 < 10**30:
                            results['amount2'] = amount2
                            print(f"   💰 Amount 2: {amount2}")
                            
            except:
                pass
        
        return results
        
    except Exception as e:
        print(f"Error decoding event data: {e}")
        return {}

def analyze_recent_events():
    """Get and analyze recent events from V2 AMM"""
    try:
        w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        if not w3.is_connected():
            print("❌ Failed to connect to Optimism")
            return
        
        print(f"✅ Connected to Optimism (block {w3.eth.block_number})")
        
        latest_block = w3.eth.get_block('latest')
        start_block = latest_block.number - 200  # Last 200 blocks for more data
        
        print(f"\n🔍 Fetching events from blocks {start_block} to {latest_block.number}")
        
        logs = w3.eth.get_logs({
            'fromBlock': start_block,
            'toBlock': 'latest',
            'address': SPORTS_AMM_V2
        })
        
        print(f"📊 Found {len(logs)} events")
        
        # Group by event signature
        events_by_sig = {}
        for log in logs:
            if log.topics:
                sig = log.topics[0].hex()
                if sig not in events_by_sig:
                    events_by_sig[sig] = []
                events_by_sig[sig].append(log)
        
        # Analyze each event type
        decoded_results = {}
        
        for sig, event_logs in events_by_sig.items():
            print(f"\n🎯 ANALYZING EVENT TYPE: {sig}")
            print(f"   Count: {len(event_logs)}")
            
            # Take a few recent examples
            recent_events = event_logs[-3:]  # Last 3 events of this type
            
            for i, log in enumerate(recent_events):
                print(f"\n   📋 Example {i+1} (Block {log.blockNumber}):")
                
                # Decode topics
                print(f"      Topics: {len(log.topics)}")
                if len(log.topics) > 1:
                    for j, topic in enumerate(log.topics[1:], 1):
                        topic_hex = topic.hex()
                        print(f"        Topic {j}: {topic_hex}")
                        
                        # Check if topic might be a game ID
                        if not topic_hex.endswith('000000000000000000000000'):
                            print(f"          🎯 Possible Game ID in topic {j}")
                
                # Decode data
                if log.data and log.data != '0x':
                    result = decode_event_data(log.data, sig)
                    if result:
                        decoded_results[f"{sig}_{i}"] = result
        
        # Summary
        print(f"\n📋 DECODING SUMMARY")
        print("=" * 50)
        
        all_game_ids = set()
        all_addresses = set()
        
        for key, result in decoded_results.items():
            if 'game_id' in result:
                all_game_ids.add(result['game_id'])
            if 'potential_game_id' in result:
                all_game_ids.add(result['potential_game_id'])
            if 'batch_game_ids' in result:
                all_game_ids.update(result['batch_game_ids'])
            if 'potential_address' in result:
                all_addresses.add(result['potential_address'])
        
        print(f"🎯 Unique Game IDs found: {len(all_game_ids)}")
        for gid in list(all_game_ids)[:5]:
            print(f"   {gid}")
        if len(all_game_ids) > 5:
            print(f"   ... and {len(all_game_ids) - 5} more")
        
        print(f"\n👤 Unique Addresses found: {len(all_addresses)}")
        for addr in list(all_addresses)[:3]:
            print(f"   {addr}")
        
        return {
            'game_ids': list(all_game_ids),
            'addresses': list(all_addresses),
            'events_by_signature': {sig: len(logs) for sig, logs in events_by_sig.items()}
        }
        
    except Exception as e:
        print(f"Error analyzing events: {e}")
        return {}

def test_game_id_queries(game_ids):
    """Test if we can get market info using discovered game IDs"""
    if not game_ids:
        print("No game IDs to test")
        return
    
    print(f"\n🔍 TESTING GAME ID QUERIES")
    print("=" * 40)
    
    w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
    
    # Test a few game IDs
    test_ids = game_ids[:3]
    
    for i, game_id in enumerate(test_ids):
        print(f"\n🎯 Testing Game ID {i+1}: {game_id}")
        
        # Try various function calls with this game ID
        test_functions = [
            ('gameDetails(bytes32)', '0x11111111'),
            ('marketForGame(bytes32)', '0x22222222'),
            ('getGameInfo(bytes32)', '0x33333333'),
            ('isGameResolved(bytes32)', '0x44444444')
        ]
        
        for func_name, func_sig in test_functions:
            try:
                # Encode game ID as parameter
                game_id_bytes = bytes.fromhex(game_id[2:])  # Remove 0x
                call_data = func_sig + game_id_bytes.hex()
                
                result = w3.eth.call({
                    'to': SPORTS_AMM_V2,
                    'data': call_data
                })
                
                if result and len(result) > 0:
                    print(f"   ✅ {func_name} -> {len(result)} bytes response")
                    
                    # Try to decode result
                    if len(result) == 32:  # Single value
                        value = int.from_bytes(result, 'big')
                        print(f"      Decoded: {value}")
                    elif len(result) == 20:  # Address
                        addr = '0x' + result.hex()
                        print(f"      Address: {addr}")
                    elif len(result) > 32:
                        print(f"      Complex data: {result.hex()[:40]}...")
                
            except Exception as e:
                if "execution reverted" not in str(e).lower():
                    print(f"   ❌ {func_name} -> {str(e)[:30]}...")

def main():
    print("🔍 OVERTIME V2 EVENT DECODER")
    print("=" * 50)
    
    # Analyze recent events
    results = analyze_recent_events()
    
    # Test game ID queries
    if results and results.get('game_ids'):
        test_game_id_queries(results['game_ids'])
    
    print(f"\n✅ Analysis complete!")
    
    if results:
        print(f"\n📌 FINDINGS:")
        print(f"   - {len(results.get('game_ids', []))} unique game IDs discovered")
        print(f"   - {len(results.get('addresses', []))} unique addresses found")
        print(f"   - {len(results.get('events_by_signature', {}))} event types")
        
        if results.get('game_ids'):
            print(f"\n🎯 NEXT STEPS:")
            print("1. Use discovered game IDs to query market details")
            print("2. Try connecting to V2 API with proper authentication") 
            print("3. Check if these represent active betting markets")

if __name__ == "__main__":
    main()