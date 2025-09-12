#!/usr/bin/env python3
"""Test blockchain reader functionality."""

import asyncio
import logging
from blockchain_reader import BlockchainReader, ChainSyncService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


async def test_blockchain_reader():
    """Test blockchain reader functionality."""
    
    print("=" * 60)
    print("BLOCKCHAIN READER TEST")
    print("=" * 60)
    
    # Test Optimism connection
    print("\n1. Testing Optimism connection...")
    try:
        reader = BlockchainReader(network='optimism')
        print(f"✓ Connected to Optimism at block {reader.w3.eth.block_number}")
        print(f"✓ SportsAMMV2 contract: {reader.sports_amm.address}")
    except Exception as e:
        print(f"✗ Failed to connect to Optimism: {e}")
        return
    
    # Test recent market creations
    print("\n2. Scanning for recent market creations...")
    try:
        # Scan last 1000 blocks (about 30 minutes on Optimism)
        from_block = reader.w3.eth.block_number - 1000
        to_block = reader.w3.eth.block_number
        
        print(f"   Scanning blocks {from_block} to {to_block}...")
        markets = reader.scan_market_creations(from_block, to_block)
        
        if markets:
            print(f"✓ Found {len(markets)} markets")
            for i, market in enumerate(markets[:3]):
                print(f"\n   Market {i+1}:")
                print(f"   - Address: {market['market_address']}")
                print(f"   - Game ID: {market['game_id']}")
                print(f"   - Label: {market['game_label']}")
                print(f"   - Block: {market['creation_block']}")
        else:
            print("✓ No markets found in recent blocks (this is normal)")
    except Exception as e:
        print(f"✗ Error scanning markets: {e}")
        import traceback
        traceback.print_exc()
    
    # Test event filter directly
    print("\n3. Testing event filter directly...")
    try:
        if hasattr(reader.sports_amm.events, 'MarketCreated'):
            print("✓ MarketCreated event found in ABI")
            
            # Try to get event signature
            event = reader.sports_amm.events.MarketCreated
            print("✓ Event signature available")
        else:
            print("✗ MarketCreated event not found in contract events")
            print(f"   Available events: {dir(reader.sports_amm.events)}")
    except Exception as e:
        print(f"✗ Error checking events: {e}")
    
    # Test odds fetching (with a known market if available)
    print("\n4. Testing odds fetching...")
    # You would need a real market address here
    # For now, we'll skip this test
    print("   (Skipped - requires active market address)")
    
    # Test Arbitrum connection
    print("\n5. Testing Arbitrum connection...")
    try:
        arb_reader = BlockchainReader(network='arbitrum')
        print(f"✓ Connected to Arbitrum at block {arb_reader.w3.eth.block_number}")
        
        # Quick scan
        from_block = arb_reader.w3.eth.block_number - 100
        markets = arb_reader.scan_market_creations(from_block)
        print(f"✓ Scanned recent blocks, found {len(markets)} markets")
    except Exception as e:
        print(f"✗ Failed Arbitrum test: {e}")
    
    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)


async def test_chain_sync():
    """Test continuous chain sync service."""
    print("\n\nTesting Chain Sync Service...")
    print("(Press Ctrl+C to stop)")
    
    reader = BlockchainReader(network='optimism')
    sync_service = ChainSyncService(reader)
    
    # Run for a short time
    try:
        await asyncio.wait_for(
            sync_service.sync_continuously(batch_size=50),
            timeout=30
        )
    except asyncio.TimeoutError:
        print("Sync test completed (30s timeout)")
    except KeyboardInterrupt:
        print("Sync test interrupted")


if __name__ == "__main__":
    # Run basic tests
    asyncio.run(test_blockchain_reader())
    
    # Optionally test continuous sync
    # asyncio.run(test_chain_sync())