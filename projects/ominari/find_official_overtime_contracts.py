#!/usr/bin/env python3
"""
Comprehensive script to find and verify official Overtime V2 contract addresses on Optimism.
This script checks both official redirectors and blockchain data to find active contracts.
"""

import requests
import time
from web3 import Web3
from eth_abi import decode

# Optimism RPC endpoint
OPTIMISM_RPC = 'https://mainnet.optimism.io'

def get_redirector_address(contract_name):
    """Get contract address from official Overtime redirector"""
    try:
        # Try both V1 and V2 redirectors
        v1_url = f"https://contracts.overtime.io/mainnet-ovm/{contract_name}"
        v2_url = f"https://v2.contracts.overtime.io/mainnet-ovm/{contract_name}"
        
        for url in [v2_url, v1_url]:
            try:
                response = requests.get(url, allow_redirects=False, timeout=10)
                if response.status_code in [301, 302]:
                    location = response.headers.get('location', '')
                    if 'optimistic.etherscan.io/address/' in location:
                        address = location.split('/address/')[-1]
                        if address.startswith('0x') and len(address) == 42:
                            return address, url
                time.sleep(0.5)
            except Exception as e:
                print(f"Error checking {url}: {e}")
                continue
                
    except Exception as e:
        print(f"Error with redirector for {contract_name}: {e}")
    
    return None, None

def check_contract_activity(w3, address, contract_name):
    """Check if contract has recent activity"""
    try:
        # Get latest block
        latest_block = w3.eth.get_block('latest')
        
        # Check last 1000 blocks for activity
        start_block = max(0, latest_block.number - 1000)
        
        print(f"\nChecking {contract_name} ({address}) for activity...")
        
        # Get basic info
        try:
            code = w3.eth.get_code(address)
            if len(code) <= 2:  # No code
                print(f"  ❌ No contract code found")
                return False
                
            balance = w3.eth.get_balance(address)
            print(f"  💰 Balance: {w3.from_wei(balance, 'ether')} ETH")
            print(f"  📊 Code size: {len(code)} bytes")
            
        except Exception as e:
            print(f"  ❌ Error getting basic info: {e}")
            return False
        
        # Check for recent transactions
        try:
            # Get recent transactions to this address
            filter_params = {
                'fromBlock': start_block,
                'toBlock': 'latest',
                'address': address
            }
            
            logs = w3.eth.get_logs(filter_params)
            print(f"  📝 Recent events: {len(logs)}")
            
            if len(logs) > 0:
                print(f"  ✅ Contract is active (found {len(logs)} events in last 1000 blocks)")
                
                # Show some recent event signatures
                signatures = set()
                for log in logs[-10:]:  # Last 10 events
                    if log.topics:
                        signatures.add(log.topics[0].hex())
                
                print(f"  🔍 Recent event signatures: {list(signatures)[:3]}")
                return True
            else:
                print(f"  ⚠️ No recent activity found")
                return False
                
        except Exception as e:
            print(f"  ❌ Error checking activity: {e}")
            return False
            
    except Exception as e:
        print(f"Error checking contract activity: {e}")
        return False

def get_proxy_implementation(w3, proxy_address):
    """Get implementation address from proxy contract"""
    try:
        # Standard proxy storage slot for implementation
        impl_slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
        
        storage_value = w3.eth.get_storage_at(proxy_address, impl_slot)
        if storage_value != b'\x00' * 32:
            # Extract address from storage (last 20 bytes)
            impl_address = '0x' + storage_value[-20:].hex()
            if impl_address != '0x' + '00' * 20:
                return w3.to_checksum_address(impl_address)
    except:
        pass
    
    return None

def check_function_signatures(w3, address, contract_name):
    """Check what function signatures the contract responds to"""
    common_signatures = {
        'obtainOdds(address,uint)': '0x3c5f5bb6',
        'buyFromAMM(address,uint,uint,uint,uint)': '0x8c6f28f2',
        'getGameDetails(uint)': '0x37e6b69d', 
        'times()': '0x9d4e5dd1',
        'resolved()': '0x2b68bb3c',
        'getAllActiveGames(uint,uint)': '0x8b0518cb',
        'getActiveMarkets(uint,uint)': '0x5a17bD98',
        'getMarketDetails(address)': '0x6ACA3f96',
        'numActiveMarkets()': '0x8b0518cb',
        'activeMarkets(uint)': '0x5a17bD98'
    }
    
    print(f"\n🔍 Testing function signatures for {contract_name}:")
    working_functions = []
    
    for func_name, sig in common_signatures.items():
        try:
            # Try calling the function
            call_data = sig + '0' * 56  # Function signature + padding
            
            result = w3.eth.call({
                'to': address,
                'data': call_data
            })
            
            if result and len(result) > 0:
                working_functions.append(func_name)
                print(f"  ✅ {func_name} -> responds")
            
        except Exception as e:
            if "execution reverted" in str(e).lower():
                # Function exists but reverted (probably needs parameters)
                working_functions.append(f"{func_name} (needs params)")
                print(f"  ⚠️ {func_name} -> exists but needs parameters")
            else:
                print(f"  ❌ {func_name} -> not found")
            
        time.sleep(0.1)  # Rate limiting
    
    return working_functions

def main():
    print("🔍 OVERTIME V2 CONTRACT DISCOVERY ON OPTIMISM")
    print("=" * 60)
    
    # Initialize Web3
    try:
        w3 = Web3(Web3.HTTPProvider(OPTIMISM_RPC))
        if not w3.is_connected():
            print("❌ Failed to connect to Optimism RPC")
            return
        print(f"✅ Connected to Optimism (block {w3.eth.block_number})")
    except Exception as e:
        print(f"❌ Failed to initialize Web3: {e}")
        return
    
    # Contracts to check
    contracts_to_check = [
        'SportsAMM',
        'SportsAMMV2', 
        'SportPositionalMarketManager',
        'SportPositionalMarketFactory',
        'SportPositionalMarketData',
        'ParlayAMM',
        'ParlayMarketManager'
    ]
    
    found_contracts = {}
    
    print("\n📡 CHECKING OFFICIAL REDIRECTORS")
    print("-" * 40)
    
    for contract_name in contracts_to_check:
        print(f"\n🔍 Looking up {contract_name}...")
        address, source_url = get_redirector_address(contract_name)
        
        if address:
            print(f"  ✅ Found: {address}")
            print(f"  📍 Source: {source_url}")
            found_contracts[contract_name] = {
                'address': address,
                'source': source_url
            }
        else:
            print(f"  ❌ Not found in redirectors")
        
        time.sleep(1)  # Rate limiting
    
    print("\n🔬 VERIFYING CONTRACT ACTIVITY")
    print("-" * 40)
    
    active_contracts = {}
    
    for contract_name, info in found_contracts.items():
        address = info['address']
        is_active = check_contract_activity(w3, address, contract_name)
        
        if is_active:
            active_contracts[contract_name] = info
            
            # Check if it's a proxy
            impl_address = get_proxy_implementation(w3, address)
            if impl_address:
                print(f"  🔗 Implementation: {impl_address}")
                info['implementation'] = impl_address
            
            # Check function signatures
            working_functions = check_function_signatures(w3, address, contract_name)
            info['working_functions'] = working_functions
        
        time.sleep(1)
    
    print("\n📋 SUMMARY OF ACTIVE CONTRACTS")
    print("=" * 60)
    
    if active_contracts:
        for contract_name, info in active_contracts.items():
            print(f"\n🏗️ {contract_name}")
            print(f"   Address: {info['address']}")
            print(f"   Source: {info['source']}")
            if 'implementation' in info:
                print(f"   Implementation: {info['implementation']}")
            if info.get('working_functions'):
                print(f"   Working functions: {len(info['working_functions'])}")
                for func in info['working_functions'][:3]:
                    print(f"     - {func}")
                if len(info['working_functions']) > 3:
                    print(f"     - ... and {len(info['working_functions']) - 3} more")
    else:
        print("❌ No active contracts found!")
    
    print(f"\n🎯 RECOMMENDED V2 ADDRESSES FOR OPTIMISM:")
    print("-" * 50)
    
    # Look for V2 or most active contracts
    if 'SportsAMMV2' in active_contracts:
        v2_amm = active_contracts['SportsAMMV2']
        print(f"Sports AMM V2: {v2_amm['address']}")
    elif 'SportsAMM' in active_contracts:
        v1_amm = active_contracts['SportsAMM'] 
        print(f"Sports AMM V1: {v1_amm['address']} (V2 not found)")
    
    if 'SportPositionalMarketManager' in active_contracts:
        manager = active_contracts['SportPositionalMarketManager']
        print(f"Market Manager: {manager['address']}")
    
    if 'SportPositionalMarketFactory' in active_contracts:
        factory = active_contracts['SportPositionalMarketFactory']
        print(f"Market Factory: {factory['address']}")
    
    print("\n✅ Contract discovery complete!")

if __name__ == "__main__":
    main()