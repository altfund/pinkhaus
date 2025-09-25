#!/usr/bin/env python3
"""Scan for recent market activity on blockchain"""

import logging
from datetime import datetime, timedelta
from rpc_config import RPCManager
from web3 import Web3
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def scan_for_markets(hours=24):
    """Scan for market creation events in the last N hours."""
    
    logger.info(f"\n📊 Scanning for markets created in the last {hours} hours...")
    
    results = {}
    
    for network in ['optimism', 'arbitrum']:
        logger.info(f"\n🔗 Scanning {network.upper()}...")
        
        try:
            rpc_manager = RPCManager(network)
            w3, _ = rpc_manager.get_web3()
            
            # Network-specific addresses
            sports_amm_addresses = {
                'optimism': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
                'arbitrum': '0x7465c5d60d3d095443CF9991Da03304A30D42Eae'
            }
            
            sports_amm = sports_amm_addresses[network]
            
            # Estimate blocks for time period
            blocks_per_second = {
                'optimism': 0.5,  # ~2 seconds per block
                'arbitrum': 4.0   # ~0.25 seconds per block
            }
            
            blocks_to_scan = int(hours * 3600 * blocks_per_second[network])
            latest_block = w3.eth.block_number
            from_block = max(1, latest_block - blocks_to_scan)
            
            logger.info(f"  Scanning blocks {from_block:,} to {latest_block:,} ({blocks_to_scan:,} blocks)")
            
            # MarketCreated event signature
            event_signature = '0x' + w3.keccak(text="MarketCreated(address,bytes32,string,uint256,uint256[],uint256[])").hex()
            
            # Scan in chunks to avoid timeouts
            chunk_size = 5000
            all_logs = []
            
            for start in range(from_block, latest_block + 1, chunk_size):
                end = min(start + chunk_size - 1, latest_block)
                
                try:
                    logs = w3.eth.get_logs({
                        'fromBlock': hex(start),
                        'toBlock': hex(end),
                        'address': sports_amm,
                        'topics': [event_signature]
                    })
                    all_logs.extend(logs)
                    
                    if logs:
                        logger.info(f"  Found {len(logs)} events in blocks {start:,}-{end:,}")
                        
                except Exception as e:
                    logger.warning(f"  Error scanning blocks {start:,}-{end:,}: {e}")
                    
            logger.info(f"\n  Total MarketCreated events found: {len(all_logs)}")
            
            results[network] = {
                'count': len(all_logs),
                'events': all_logs
            }
            
            # Show recent markets
            if all_logs:
                logger.info(f"\n  Recent markets on {network}:")
                for i, log in enumerate(all_logs[-5:]):  # Last 5 markets
                    market_address = '0x' + log['topics'][1].hex()[26:]
                    block_number = log['blockNumber']
                    
                    # Get block timestamp
                    try:
                        block = w3.eth.get_block(block_number)
                        timestamp = datetime.fromtimestamp(block['timestamp'])
                        logger.info(f"    Market: {market_address}")
                        logger.info(f"    Block: {block_number:,} at {timestamp}")
                    except:
                        logger.info(f"    Market: {market_address} at block {block_number:,}")
                        
        except Exception as e:
            logger.error(f"Error scanning {network}: {e}")
            results[network] = {'count': 0, 'error': str(e)}
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("SUMMARY:")
    for network, data in results.items():
        if 'error' in data:
            logger.info(f"  {network}: Error - {data['error']}")
        else:
            logger.info(f"  {network}: {data['count']} markets created")
    
    return results

def check_trading_activity():
    """Check for recent trading activity (BoughtFromAmm events)."""
    
    logger.info("\n💰 Checking recent trading activity...")
    
    try:
        rpc_manager = RPCManager('optimism')
        w3, _ = rpc_manager.get_web3()
        
        sports_amm = '0xFb4e4811C7A811E098A556bD79B64c20b479E431'
        
        # BoughtFromAmm event signature
        event_signature = '0x' + w3.keccak(text="BoughtFromAmm(address,address,uint8,uint256,uint256,address,address)").hex()
        
        # Last 1000 blocks
        latest_block = w3.eth.block_number
        from_block = latest_block - 1000
        
        logs = w3.eth.get_logs({
            'fromBlock': hex(from_block),
            'toBlock': hex(latest_block),
            'address': sports_amm,
            'topics': [event_signature]
        })
        
        logger.info(f"  Found {len(logs)} trades in the last 1000 blocks")
        
        if logs:
            logger.info("  Recent trades show active markets exist!")
            
    except Exception as e:
        logger.error(f"Error checking trades: {e}")

if __name__ == "__main__":
    logger.info("🔍 Scanning for Overtime Markets Activity")
    logger.info("=" * 50)
    
    # Scan for markets in last 24 hours
    scan_for_markets(hours=24)
    
    # Check trading activity
    check_trading_activity()
    
    logger.info("\n✅ Scan complete!")