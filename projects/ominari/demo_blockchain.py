#!/usr/bin/env python3
"""
Demonstration of Blockchain Integration for Ominari Trading System

Shows how to:
1. Connect to blockchain (Optimism/Arbitrum)
2. Read market data and odds
3. Sync with database
4. Find trading opportunities
5. Execute trades (paper mode)
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def demo_blockchain_integration():
    """Demonstrate the complete blockchain integration workflow."""
    try:
        from blockchain_reader import BlockchainReader
        from blockchain_trading import BlockchainIntegration, BlockchainPosition
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.info("Make sure all dependencies are installed")
        return
        
    print("\n" + "="*60)
    print("OMINARI BLOCKCHAIN INTEGRATION DEMO")
    print("="*60 + "\n")
    
    # 1. Initialize blockchain reader
    print("1. Initializing Blockchain Reader...")
    try:
        reader = BlockchainReader(network='optimism')
        if reader.check_connection():
            print(f"   ✅ Connected to Optimism")
            print(f"   📊 Current block: {reader.w3.eth.block_number:,}")
            print(f"   🏢 SportsAMM: {reader.sports_amm.address}")
        else:
            print("   ❌ Connection failed")
            return
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
        
    # 2. Fetch recent markets
    print("\n2. Fetching Recent Markets...")
    try:
        markets = reader.fetch_recent_markets(hours_back=24)
        print(f"   📈 Found {len(markets)} markets in last 24 hours")
        
        # Show a few examples
        for i, market in enumerate(markets[:3]):
            print(f"\n   Market #{i+1}:")
            print(f"     🏟️  {market.get('home_team', 'Unknown')} vs {market.get('away_team', 'Unknown')}")
            print(f"     ⚽ Sport: {market.get('sport', 'Unknown')}")
            print(f"     🏆 League: {market.get('league', 'Unknown')}")
            print(f"     📅 Starts: {market.get('maturity_date', 'Unknown')}")
            print(f"     📍 Address: {market.get('address', 'Unknown')[:10]}...")
            
            # Get current odds
            if market.get('address'):
                try:
                    odds = reader.get_current_odds(market['address'])
                    if odds:
                        print("     💰 Current Odds:")
                        for pos, odd in odds.items():
                            print(f"        Position {pos}: {odd['buy']:.2f}")
                except Exception as e:
                    print(f"     ⚠️  Could not fetch odds: {e}")
                    
    except Exception as e:
        print(f"   ❌ Error fetching markets: {e}")
        
    # 3. Initialize integration (paper trading mode)
    print("\n3. Initializing Blockchain Integration...")
    try:
        integration = BlockchainIntegration(network='optimism', paper_trading_mode=True)
        print("   ✅ Integration initialized (Paper Trading Mode)")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
        
    # 4. Sync markets to database
    print("\n4. Syncing Markets to Database...")
    try:
        count = integration.sync_blockchain_markets(hours_back=24)
        print(f"   💾 Synced {count} new markets to database")
    except Exception as e:
        print(f"   ⚠️  Error syncing (database may not be available): {e}")
        
    # 5. Check trader balance (if configured)
    print("\n5. Checking Trader Balance...")
    if not integration.paper_trading_mode and integration.trader:
        try:
            balances = integration.trader.check_balance()
            print(f"   💵 SUSD Balance: ${balances['susd']}")
            print(f"   ⛽ ETH Balance: {balances['eth']} ETH")
        except Exception as e:
            print(f"   ⚠️  Could not check balance: {e}")
    else:
        print("   📝 Paper trading mode - no real balance needed")
        
    # 6. Find trading opportunities
    print("\n6. Finding Trading Opportunities...")
    try:
        opportunities = integration.evaluate_blockchain_opportunities(min_edge=0.02)
        
        if opportunities:
            print(f"   🎯 Found {len(opportunities)} opportunities!")
            
            for i, opp in enumerate(opportunities[:3]):
                market = opp['market']
                odd = opp['odd']
                
                print(f"\n   Opportunity #{i+1}:")
                print(f"     🏟️  {market.home_team} vs {market.away_team}")
                print(f"     📊 Outcome: {odd.outcome}")
                print(f"     💹 Edge: {opp['edge']:.2%}")
                print(f"     🎲 Odds: {odd.decimal_odds:.2f}")
                print(f"     📈 Fair Prob: {opp['fair_prob']:.2%}")
                print(f"     📉 Market Prob: {opp['implied_prob']:.2%}")
                print(f"     💰 Kelly Stake: {opp['recommended_stake']:.2%} of bankroll")
        else:
            print("   ℹ️  No opportunities found with minimum 2% edge")
            
    except Exception as e:
        print(f"   ⚠️  Error evaluating opportunities: {e}")
        
    # 7. Demo trade execution (paper mode)
    if opportunities and integration.paper_trading_mode:
        print("\n7. Demo Trade Execution (Paper Mode)...")
        
        opp = opportunities[0]
        print(f"\n   📋 Would execute trade:")
        print(f"     Market: {opp['market'].home_team} vs {opp['market'].away_team}")
        print(f"     Outcome: {opp['odd'].outcome}")
        print(f"     Stake: $100 (demo amount)")
        print(f"     Expected Odds: {opp['odd'].decimal_odds:.2f}")
        
        # Create position
        position = BlockchainPosition(
            market_address=opp['market'].source_id,
            position=opp['odd'].position,
            amount=Decimal('100'),
            expected_odds=Decimal(str(opp['odd'].decimal_odds))
        )
        
        print(f"\n   ✅ In real mode, this would execute on-chain!")
        
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60 + "\n")


def demo_chain_sync():
    """Demonstrate continuous chain syncing."""
    print("\n" + "="*60)
    print("BLOCKCHAIN SYNC SERVICE DEMO")
    print("="*60 + "\n")
    
    print("This would start a continuous sync service that:")
    print("  1. Monitors new blocks on Optimism/Arbitrum")
    print("  2. Scans for new market creations")
    print("  3. Tracks trading activity")
    print("  4. Updates odds in real-time")
    print("  5. Stores all data in local database")
    print("\nTo run: python -c \"from blockchain_reader import ChainSyncService; ...\"")


if __name__ == "__main__":
    # Run main demo
    demo_blockchain_integration()
    
    # Show sync service info
    demo_chain_sync()
    
    print("\n📚 For more information, see:")
    print("   - blockchain_reader.py: Reading market data from chain")
    print("   - blockchain_trading.py: Executing trades on-chain")
    print("   - test_blockchain_integration.py: Comprehensive test suite")