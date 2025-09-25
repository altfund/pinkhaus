#!/usr/bin/env python3
"""
Check the actual verified contract addresses
Based on the guide's links
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# From the guide - these are the verified contracts we should check
ADDRESSES_TO_CHECK = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        # From the guide's Arbiscan link
        'manager_from_guide': '0x268BB40F4993f6234D924ba70D20BD59d781F7F6',
        # SportsAMM we know works
        'sports_amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        # Other candidates from various sources
        'candidates': [
            '0x91b0d67a06936ad75c13e2b5f14F36dcf22D12Aa',  # Possible manager
            '0x0089282ac624bbbda7ec472d63ee99643d7edf4a',  # From storage
        ]
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        # SportsAMM implementation from guide
        'sports_amm_impl': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        # Regular AMM
        'sports_amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
        'candidates': [
            '0x81DD7B07eb4bc9ffF0274d5C7F326b96B6557e53',
            '0xef2b0a2ffcdcd3b53d81800d19a543136f26a6b8',  # From storage
        ]
    }
}

def check_contract_info(w3, address, name):
    """Check basic info about a contract."""
    try:
        # Check if contract exists
        code = w3.eth.get_code(address)
        if code and len(code) > 10:
            logger.info(f"✅ {name} at {address}")
            logger.info(f"   Code size: {len(code)} bytes")
            
            # Try to check if it's a proxy
            # Implementation slot for EIP-1967 proxies
            impl_slot = '0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc'
            impl_data = w3.eth.get_storage_at(address, impl_slot)
            if impl_data != b'\x00' * 32:
                impl_addr = '0x' + impl_data.hex()[-40:]
                logger.info(f"   Proxy implementation: {impl_addr}")
                
            return True
        else:
            logger.warning(f"❌ No contract at {address} ({name})")
            return False
    except Exception as e:
        logger.error(f"Error checking {address}: {e}")
        return False

def check_arbiscan(address):
    """Check contract name from Arbiscan."""
    try:
        url = f"https://api.arbiscan.io/api?module=contract&action=getsourcecode&address={address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result'][0]['ContractName']:
                return data['result'][0]['ContractName']
    except:
        pass
    return None

def main():
    """Main function."""
    logger.info("🔍 Checking Verified Contract Addresses")
    logger.info("=" * 60)
    
    # Check Arbitrum
    logger.info("\n📡 Checking Arbitrum contracts...")
    config = ADDRESSES_TO_CHECK['arbitrum']
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if w3.is_connected():
            logger.info(f"Connected to Arbitrum")
            
            # Check manager from guide
            check_contract_info(w3, config['manager_from_guide'], "Manager from guide")
            name = check_arbiscan(config['manager_from_guide'])
            if name:
                logger.info(f"   Verified as: {name}")
                
            # Check SportsAMM
            check_contract_info(w3, config['sports_amm'], "SportsAMM")
            
            # Check candidates
            for addr in config['candidates']:
                if check_contract_info(w3, addr, "Candidate"):
                    name = check_arbiscan(addr)
                    if name:
                        logger.info(f"   Verified as: {name}")
                        
    except Exception as e:
        logger.error(f"Error checking Arbitrum: {e}")
        
    # Check Optimism
    logger.info("\n📡 Checking Optimism contracts...")
    config = ADDRESSES_TO_CHECK['optimism']
    
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if w3.is_connected():
            logger.info(f"Connected to Optimism")
            
            # Check known contracts
            check_contract_info(w3, config['sports_amm_impl'], "SportsAMM Implementation")
            check_contract_info(w3, config['sports_amm'], "SportsAMM")
            
            # Check candidates
            for addr in config['candidates']:
                check_contract_info(w3, addr, "Candidate")
                
    except Exception as e:
        logger.error(f"Error checking Optimism: {e}")

if __name__ == "__main__":
    main()