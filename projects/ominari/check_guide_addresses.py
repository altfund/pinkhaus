#!/usr/bin/env python3
"""
Check the exact addresses from the user's guide
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# From the user's comprehensive guide
FROM_GUIDE = {
    'arbitrum': {
        'rpc': 'https://arb1.arbitrum.io/rpc',
        'SportPositionalMarketManager': '0x268BB40F4993f6234D924ba70D20BD59d781F7F6',
        'SportsAMM': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
        'name': 'Arbitrum'
    },
    'optimism': {
        'rpc': 'https://mainnet.optimism.io',
        'SportPositionalMarketManager': '0x81DD7B07eb4bc9ffF0274d5C7F326b96B6557e53',
        'SportsAMM': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
        'name': 'Optimism'
    }
}

def check_arbiscan(address):
    """Check Arbiscan for contract info."""
    try:
        url = f"https://api.arbiscan.io/api?module=contract&action=getsourcecode&address={address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result']:
                result = data['result'][0]
                return {
                    'name': result.get('ContractName', 'Unknown'),
                    'compiler': result.get('CompilerVersion', 'Unknown'),
                    'proxy': result.get('Proxy', '0'),
                    'implementation': result.get('Implementation', '')
                }
    except Exception as e:
        logger.error(f"Arbiscan error: {e}")
    return None

def check_optimistic(address):
    """Check Optimistic Etherscan for contract info."""
    try:
        url = f"https://api-optimistic.etherscan.io/api?module=contract&action=getsourcecode&address={address}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data['status'] == '1' and data['result']:
                result = data['result'][0]
                return {
                    'name': result.get('ContractName', 'Unknown'),
                    'compiler': result.get('CompilerVersion', 'Unknown'),
                    'proxy': result.get('Proxy', '0'),
                    'implementation': result.get('Implementation', '')
                }
    except Exception as e:
        logger.error(f"Optimistic error: {e}")
    return None

def check_contract_methods(w3, address):
    """Try to call common manager methods."""
    logger.info("\n  Testing contract methods...")
    
    # Try different method signatures
    methods = [
        ('numActiveMarkets', '0x69cf17d0'),  # numActiveMarkets()
        ('numberOfActiveMarkets', '0x6ca77f42'),  # numberOfActiveMarkets()
        ('activeMarkets(uint256)', '0xc6da3ba6'),  # activeMarkets(uint256)
        ('getAllActiveMarkets', '0x6abb1973'),  # getAllActiveMarkets()
    ]
    
    for name, selector in methods:
        try:
            result = w3.eth.call({
                'to': address,
                'data': selector
            })
            if result:
                logger.info(f"    ✅ {name} responded with {len(result)} bytes")
                # Try to decode as uint256
                if len(result) == 32:
                    value = int.from_bytes(result, 'big')
                    if value < 1000000:  # Reasonable number
                        logger.info(f"       Value: {value}")
        except Exception as e:
            logger.debug(f"    ❌ {name} failed: {e}")

def main():
    """Main function."""
    logger.info("🔍 Checking Guide Contract Addresses")
    logger.info("=" * 60)
    
    # Check Arbitrum
    logger.info("\n📡 ARBITRUM")
    config = FROM_GUIDE['arbitrum']
    
    logger.info(f"\nSportPositionalMarketManager: {config['SportPositionalMarketManager']}")
    info = check_arbiscan(config['SportPositionalMarketManager'])
    if info:
        logger.info(f"  Contract Name: {info['name']}")
        logger.info(f"  Is Proxy: {info['proxy']}")
        if info['implementation']:
            logger.info(f"  Implementation: {info['implementation']}")
    else:
        logger.warning("  Not verified on Arbiscan")
        
    # Test methods
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if w3.is_connected():
            code = w3.eth.get_code(config['SportPositionalMarketManager'])
            logger.info(f"  Code size: {len(code)} bytes")
            check_contract_methods(w3, config['SportPositionalMarketManager'])
    except:
        pass
        
    logger.info(f"\nSportsAMM: {config['SportsAMM']}")
    info = check_arbiscan(config['SportsAMM'])
    if info:
        logger.info(f"  Contract Name: {info['name']}")
        logger.info(f"  Is Proxy: {info['proxy']}")
    else:
        logger.warning("  Not verified on Arbiscan")
        
    # Check Optimism
    logger.info("\n\n📡 OPTIMISM")
    config = FROM_GUIDE['optimism']
    
    logger.info(f"\nSportPositionalMarketManager: {config['SportPositionalMarketManager']}")
    info = check_optimistic(config['SportPositionalMarketManager'])
    if info:
        logger.info(f"  Contract Name: {info['name']}")
        logger.info(f"  Is Proxy: {info['proxy']}")
        if info['implementation']:
            logger.info(f"  Implementation: {info['implementation']}")
    else:
        logger.warning("  Not verified on Optimistic Etherscan")
        
    # Test methods
    try:
        w3 = Web3(Web3.HTTPProvider(config['rpc']))
        if w3.is_connected():
            code = w3.eth.get_code(config['SportPositionalMarketManager'])
            logger.info(f"  Code size: {len(code)} bytes")
            check_contract_methods(w3, config['SportPositionalMarketManager'])
    except:
        pass
        
    logger.info(f"\nSportsAMM: {config['SportsAMM']}")
    info = check_optimistic(config['SportsAMM'])
    if info:
        logger.info(f"  Contract Name: {info['name']}")
        logger.info(f"  Is Proxy: {info['proxy']}")
    else:
        logger.warning("  Not verified on Optimistic Etherscan")

if __name__ == "__main__":
    main()