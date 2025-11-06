#!/usr/bin/env python3
"""Start live blockchain trading with proper filtering and chunked execution"""

import os
import asyncio
import logging
from datetime import datetime, timezone

# Import testnet configuration
from testnet_config_simple import setup_testnet_environment, TESTNET_CONFIG

# Setup environment
setup_testnet_environment()

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from integrated_blockchain_trading import IntegratedBlockchainTrading
from paper_trading_postgres_integrated import PaperTradingSessionManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Start live blockchain trading system"""
    print("🚀 Starting Live Blockchain Trading System")
    print("=" * 80)
    
    # Display configuration
    print(f"📋 Configuration:")
    print(f"  Network: {TESTNET_CONFIG['default_network']}")
    print(f"  Paper Trading: {TESTNET_CONFIG['paper_trading_mode']}")
    print(f"  Simulation: {TESTNET_CONFIG['simulate_blockchain_calls']}")
    print(f"  Supported Sports: {TESTNET_CONFIG['supported_sports']}")
    print(f"  Supported Outcomes: {TESTNET_CONFIG['supported_outcomes']}")
    print(f"  Chunk Size: {TESTNET_CONFIG['chunk_size']}")
    print(f"  Min/Max Odds: {TESTNET_CONFIG['min_odds']}-{TESTNET_CONFIG['max_odds']}")
    
    # Get or create trading session
    session_manager = PaperTradingSessionManager()
    session_id = session_manager.get_current_session()
    
    if not session_id:
        print("\n❌ No active trading session found")
        print("Creating new trading session...")
        session_id = session_manager.create_session(
            initial_bankroll=10000.0,
            description="Live Blockchain Trading Session"
        )
        print(f"✅ Created new session: {session_id}")
    else:
        session = session_manager.get_session(session_id)
        print(f"\n📊 Using existing session: {session_id}")
        print(f"  Current bankroll: ${session['current_bankroll']:,.2f}")
        print(f"  Session created: {session.get('created_at', 'Unknown')}")
    
    # Initialize integrated trading system
    try:
        trading_system = IntegratedBlockchainTrading(session_id)
        print(f"\n✅ Trading system initialized successfully")
        
    except Exception as e:
        print(f"\n❌ Failed to initialize trading system: {e}")
        return False
    
    # Ask user for trading mode
    print(f"\n🎯 Trading Options:")
    print(f"1. Single cycle test")
    print(f"2. Continuous trading (30s intervals)")
    print(f"3. Continuous trading (custom interval)")
    print(f"4. Multiple cycles (specify count)")
    
    try:
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            # Single cycle
            print(f"\n🔄 Running single trading cycle...")
            result = await trading_system.run_trading_cycle(simulate=True)
            
            if result['success']:
                print(f"\n✅ Trading cycle completed successfully!")
                print(f"  Markets analyzed: {result.get('markets_analyzed', 0)}")
                print(f"  Blockchain trades executed: {result.get('trades_executed', 0)}")
                print(f"  Portfolio value: ${result.get('positions_value', 0):,.2f}")
                print(f"  Cash balance: ${result.get('cash_balance', 0):,.2f}")
            else:
                print(f"\n❌ Trading cycle failed: {result.get('error', 'Unknown error')}")
        
        elif choice == '2':
            # Continuous trading - 30s intervals
            print(f"\n🔄 Starting continuous trading (30s intervals)")
            print(f"Press Ctrl+C to stop...")
            
            await trading_system.monitor_and_trade(interval=30)
        
        elif choice == '3':
            # Continuous trading - custom interval
            interval = int(input("Enter interval in seconds: "))
            print(f"\n🔄 Starting continuous trading ({interval}s intervals)")
            print(f"Press Ctrl+C to stop...")
            
            await trading_system.monitor_and_trade(interval=interval)
        
        elif choice == '4':
            # Multiple cycles
            cycles = int(input("Enter number of cycles: "))
            interval = int(input("Enter interval between cycles (seconds): "))
            
            print(f"\n🔄 Running {cycles} trading cycles with {interval}s intervals")
            
            for i in range(cycles):
                print(f"\n📊 Cycle {i+1}/{cycles}")
                result = await trading_system.run_trading_cycle(simulate=True)
                
                if result['success']:
                    print(f"  ✅ Cycle {i+1} complete: {result.get('trades_executed', 0)} trades executed")
                else:
                    print(f"  ❌ Cycle {i+1} failed: {result.get('error', 'Unknown error')}")
                
                if i < cycles - 1:  # Don't wait after last cycle
                    print(f"  ⏳ Waiting {interval}s...")
                    await asyncio.sleep(interval)
        
        else:
            print("Invalid option selected")
            return False
        
        print(f"\n🎉 Trading session completed!")
        
        # Show final session stats
        final_session = session_manager.get_session(session_id)
        print(f"\n📈 Final Session Stats:")
        print(f"  Session ID: {session_id}")
        print(f"  Final bankroll: ${final_session['current_bankroll']:,.2f}")
        print(f"  Session created: {final_session.get('created_at', 'Unknown')}")
        print(f"  Description: {final_session.get('description', 'N/A')}")
        
        return True
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Trading stopped by user")
        return True
    except Exception as e:
        print(f"\n❌ Error during trading: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    if success:
        print(f"\n✅ Live blockchain trading system completed successfully!")
    else:
        print(f"\n❌ Live blockchain trading system encountered errors")