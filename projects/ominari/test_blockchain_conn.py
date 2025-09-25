#!/usr/bin/env python3
"""Test blockchain connection and market discovery"""

import logging
from rpc_config import RPCManager
from web3 import Web3
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test blockchain connection for both networks."""
    
    for network in ['optimism', 'arbitrum']:
        logger.info(f"\n🔗 Testing {network.upper()} connection...")
        
        try:
            # Get RPC connection
            rpc_manager = RPCManager(network)
            w3, endpoint = rpc_manager.get_web3()
            
            # Test connection
            if w3.is_connected():
                block_number = w3.eth.block_number
                logger.info(f"✅ Connected to {network} via {endpoint.name}")
                logger.info(f"   Current block: {block_number:,}")
                
                # Check chain ID
                chain_id = w3.eth.chain_id
                logger.info(f"   Chain ID: {chain_id}")
                
                # Get latest block info
                block = w3.eth.get_block('latest')
                logger.info(f"   Latest block timestamp: {block['timestamp']}")
                
                # Check SportsAMMV2 contract
                sports_amm_addresses = {
                    'optimism': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
                    'arbitrum': '0x7465c5d60d3d095443CF9991Da03304A30D42Eae'
                }
                
                if network in sports_amm_addresses:
                    address = sports_amm_addresses[network]
                    code = w3.eth.get_code(address)
                    if code:
                        logger.info(f"✅ SportsAMMV2 contract found at {address}")
                        logger.info(f"   Contract code length: {len(code)} bytes")
                    else:
                        logger.warning(f"⚠️  No contract found at {address}")
                
            else:
                logger.error(f"❌ Failed to connect to {network}")
                
        except Exception as e:
            logger.error(f"❌ Error connecting to {network}: {e}")

def check_recent_markets():
    """Check for recent market creation events."""
    
    logger.info("\n📊 Checking for recent markets...")
    
    try:
        rpc_manager = RPCManager('optimism')
        w3, _ = rpc_manager.get_web3()
        
        # SportsAMMV2 contract
        sports_amm = '0xFb4e4811C7A811E098A556bD79B64c20b479E431'
        
        # MarketCreated event signature
        # event MarketCreated(address indexed market, bytes32 indexed gameId, string gameLabel, uint256 maturityDate, uint256[] tags, uint256[] normalizedOdds)
        event_signature = w3.keccak(text="MarketCreated(address,bytes32,string,uint256,uint256[],uint256[])").hex()
        
        # Get recent blocks (last 1000 blocks = ~33 minutes on Optimism)
        latest_block = w3.eth.block_number
        from_block = latest_block - 1000
        
        logger.info(f"Scanning blocks {from_block:,} to {latest_block:,}")
        
        # Get logs
        logs = w3.eth.get_logs({
            'fromBlock': from_block,
            'toBlock': latest_block,
            'address': sports_amm,
            'topics': [event_signature]
        })
        
        logger.info(f"Found {len(logs)} MarketCreated events")
        
        if logs:
            # Show first few markets
            for i, log in enumerate(logs[:5]):
                market_address = '0x' + log['topics'][1].hex()[26:]
                game_id = log['topics'][2].hex()
                logger.info(f"\n  Market {i+1}:")
                logger.info(f"    Address: {market_address}")
                logger.info(f"    Game ID: {game_id}")
                logger.info(f"    Block: {log['blockNumber']:,}")
                logger.info(f"    Tx: {log['transactionHash'].hex()}")
                
            if len(logs) > 5:
                logger.info(f"\n  ... and {len(logs) - 5} more markets")
                
    except Exception as e:
        logger.error(f"Error checking markets: {e}")

if __name__ == "__main__":
    logger.info("🧪 Testing Ominari Blockchain Connection")
    logger.info("=" * 50)
    
    # Test connections
    test_connection()
    
    # Check for recent markets
    check_recent_markets()
    
    logger.info("\n✅ Blockchain connection test complete!")