#!/usr/bin/env python3
"""
Analyze V2 AMM transactions to understand the structure
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
from web3 import Web3
from collections import Counter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_recent_transactions():
    """Analyze recent V2 AMM transactions to understand patterns."""
    
    w3 = Web3(Web3.HTTPProvider('https://arb1.arbitrum.io/rpc'))
    amm_address = '0xfb64E79A562F7250131cf528242CEB10fDC82395'
    
    logger.info("🔍 Analyzing V2 AMM transactions...")
    
    current_block = w3.eth.block_number
    method_signatures = Counter()
    sample_transactions = []
    
    # Scan last 500 blocks
    for i in range(500):
        block_num = current_block - i
        
        try:
            block = w3.eth.get_block(block_num, full_transactions=True)
            
            for tx in block['transactions']:
                if tx['to'] and tx['to'].lower() == amm_address.lower():
                    input_data = tx['input']
                    
                    # Get method signature (first 10 chars including 0x)
                    if len(input_data) >= 10:
                        method_sig = input_data[:10]
                        method_signatures[method_sig] += 1
                        
                        # Keep some sample transactions
                        if len(sample_transactions) < 10:
                            sample_transactions.append({
                                'hash': tx['hash'].hex(),
                                'method': method_sig,
                                'input_length': len(input_data),
                                'from': tx['from'],
                                'block': block_num
                            })
                            
        except:
            continue
    
    # Display results
    logger.info(f"\n📊 Found {sum(method_signatures.values())} total transactions")
    logger.info("\n🔧 Method signatures used:")
    for sig, count in method_signatures.most_common():
        logger.info(f"  {sig}: {count} calls")
    
    logger.info("\n📝 Sample transactions:")
    for tx in sample_transactions[:5]:
        logger.info(f"\n  Transaction: {tx['hash']}")
        logger.info(f"  Method: {tx['method']}")
        logger.info(f"  Input length: {tx['input_length']} chars")
        logger.info(f"  From: {tx['from']}")
        logger.info(f"  Block: {tx['block']}")
    
    # Try to decode common methods
    logger.info("\n🔍 Attempting to identify methods:")
    
    # Common Overtime/AMM methods
    known_methods = {
        '0x942b67dc': 'buyFromAMM',
        '0x8cc7bcc5': 'sellToAMM',
        '0x1ed66ad7': 'buyFromAMMWithDifferentCollateral',
        '0x4917dc72': 'sellToAMMWithDifferentCollateral',
        '0xb12df8e3': 'getMarketDefaultOdds',
        '0x3850c7bd': 'getCurrentRoundId',  # Chainlink related
        '0x06fdde03': 'name()',
        '0x95d89b41': 'symbol()',
        '0x313ce567': 'decimals()'
    }
    
    for sig, count in method_signatures.most_common():
        if sig in known_methods:
            logger.info(f"  {sig}: {known_methods[sig]} ({count} calls)")
        else:
            logger.info(f"  {sig}: Unknown method ({count} calls)")
    
    # Check if markets might be in storage
    logger.info("\n💾 Checking AMM storage slots...")
    for slot in range(10):
        try:
            data = w3.eth.get_storage_at(amm_address, slot)
            if data != b'\x00' * 32:
                logger.info(f"  Slot {slot}: {data.hex()}")
                # Try to interpret as address
                potential_addr = '0x' + data.hex()[-40:]
                try:
                    code = w3.eth.get_code(potential_addr)
                    if code and len(code) > 100:
                        logger.info(f"    Possible contract at: {potential_addr}")
                except:
                    pass
        except:
            pass

if __name__ == "__main__":
    analyze_recent_transactions()