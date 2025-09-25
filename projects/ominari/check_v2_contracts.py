#!/usr/bin/env python3
"""
Check V2 contracts from v2.contracts.overtime.io
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from web3 import Web3
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def resolve_v2_address(path):
    """Resolve V2 contract address from redirector."""
    url = f"https://v2.contracts.overtime.io/{path}"
    logger.info(f"🔗 Resolving: {url}")
    
    try:
        response = requests.get(url, allow_redirects=True, timeout=10)
        final_url = response.url
        
        # Extract address
        match = re.search(r'0x[a-fA-F0-9]{40}', final_url)
        if match:
            address = Web3.to_checksum_address(match.group(0))
            logger.info(f"✅ Found: {address}")
            logger.info(f"   Explorer: {final_url}")
            return address, final_url
        else:
            logger.error(f"Could not parse address from: {final_url}")
            return None, final_url
            
    except Exception as e:
        logger.error(f"Error resolving: {e}")
        return None, None

def check_contract_activity(w3, address, name):
    """Check if contract has recent activity."""
    try:
        # Get code
        code = w3.eth.get_code(address)
        logger.info(f"\n🔍 {name} at {address}")
        logger.info(f"   Code size: {len(code)} bytes")
        
        # Check recent transactions
        current_block = w3.eth.block_number
        tx_count = 0
        
        for block_num in range(current_block - 100, current_block):
            try:
                block = w3.eth.get_block(block_num, full_transactions=True)
                for tx in block['transactions']:
                    if tx['to'] and tx['to'].lower() == address.lower():
                        tx_count += 1
            except:
                pass
                
        logger.info(f"   Recent transactions (last 100 blocks): {tx_count}")
        
        return tx_count > 0
        
    except Exception as e:
        logger.error(f"Error checking {name}: {e}")
        return False

def main():
    """Main function."""
    logger.info("🎯 Checking Overtime V2 Contracts")
    logger.info("=" * 60)
    
    # V2 contract paths to check
    v2_contracts = [
        ('arbitrumOne/SportsAMMV2', 'SportsAMMV2'),
        ('arbitrumOne/SportsAMMV2Manager', 'Manager'),
        ('arbitrumOne/SportsAMMV2RiskManager', 'RiskManager'),
        ('arbitrumOne/SportsAMMV2Data', 'Data'),
        ('optimismMainnet/SportsAMMV2', 'SportsAMMV2'),
        ('optimismMainnet/SportsAMMV2Manager', 'Manager')
    ]
    
    active_contracts = {}
    
    # Resolve all V2 addresses
    logger.info("\n🔍 Resolving V2 contract addresses...")
    for path, name in v2_contracts:
        addr, url = resolve_v2_address(path)
        if addr:
            chain = 'arbitrum' if 'arbitrum' in path else 'optimism'
            if chain not in active_contracts:
                active_contracts[chain] = {}
            active_contracts[chain][name] = addr
            
    # Check activity on each chain
    for chain, contracts in active_contracts.items():
        logger.info(f"\n🌐 Checking {chain.upper()} V2 activity...")
        
        if chain == 'arbitrum':
            rpc = 'https://arb1.arbitrum.io/rpc'
        else:
            rpc = 'https://mainnet.optimism.io'
            
        try:
            w3 = Web3(Web3.HTTPProvider(rpc))
            if w3.is_connected():
                logger.info(f"Connected to {chain} at block {w3.eth.block_number:,}")
                
                for name, addr in contracts.items():
                    has_activity = check_contract_activity(w3, addr, name)
                    if has_activity:
                        logger.info(f"   ✅ {name} has recent activity!")
                        
        except Exception as e:
            logger.error(f"Error checking {chain}: {e}")
            
    # Summary
    logger.info("\n📊 V2 Contract Summary:")
    for chain, contracts in active_contracts.items():
        logger.info(f"\n{chain.upper()}:")
        for name, addr in contracts.items():
            logger.info(f"  {name}: {addr}")
            
    logger.info("\n💡 Next Steps:")
    logger.info("1. Use the V2 contracts with recent activity")
    logger.info("2. Fetch ABIs from the explorer pages")
    logger.info("3. Look for market creation events or manager methods")

if __name__ == "__main__":
    main()