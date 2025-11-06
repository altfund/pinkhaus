#!/usr/bin/env python3
"""Continuous terminal trading with unified blockchain + API data"""

import os
import sys
import asyncio
import time
import logging
from datetime import datetime, timezone
import traceback

# Set up database environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999' 
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

# Import components
from paper_trading_postgres_integrated import PaperTradingSessionManager
from portfolio_trading_engine import PortfolioTradingEngine
from edge_calculator import EdgeCalculator
from continuous_portfolio_optimizer import ContinuousPortfolioOptimizer
from unified_data_fetcher import UnifiedDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


async def run_continuous_trading():
    """Main continuous trading function with unified data"""
    
    # Initialize components
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    unified_fetcher = UnifiedDataFetcher()
    
    # Get or create session
    session_id = session_manager.get_current_session()
    if not session_id:
        print("❌ No active trading session!")
        return
    
    session = session_manager.get_session(session_id)
    current_bankroll = float(session['current_bankroll'])
    
    print(f"💰 Session: {session_id}")
    print(f"   Bankroll: ${current_bankroll:,.2f}")
    
    # Trading configuration
    strategy_config = {
        'bankroll': current_bankroll,
        'kelly_fraction': 0.25,
        'min_edge': 0.01,  # 1% minimum edge
        'cap_per_bet': 0.02,  # 2% of bankroll max
        'cap_per_game': 0.02,
        'min_bet': 10,
        'max_positions': 20
    }
    
    portfolio_engine = PortfolioTradingEngine(
        session_manager, 
        edge_calculator, 
        strategy_config
    )
    
    # Initialize continuous optimizer with 2-hour chunks
    continuous_optimizer = ContinuousPortfolioOptimizer(
        portfolio_engine, 
        chunk_hours=2.0
    )
    
    print(f"\n🚀 Starting continuous trading with unified data sources...")
    print(f"   Chunk size: 2.0 hours")
    print(f"   Max positions: {strategy_config['max_positions']}")
    print(f"   Min edge: {strategy_config['min_edge']*100:.1f}%")
    print(f"   Kelly fraction: {strategy_config['kelly_fraction']*100:.0f}%")
    
    # Trading loop
    cycle = 0
    while True:
        cycle += 1
        print(f"\n{'='*60}")
        print(f"📊 Trading Cycle {cycle} - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"{'='*60}")
        
        try:
            # Fetch unified market data
            print("\n🔄 Fetching unified market data...")
            markets = await unified_fetcher.fetch_all_markets()
            
            if not markets:
                print("❌ No markets available")
                print("⏳ Waiting 60 seconds...")
                await asyncio.sleep(60)
                continue
            
            # Format for trading
            trading_markets = unified_fetcher.format_for_trading(markets)
            print(f"✅ Found {len(trading_markets)} valid markets")
            
            # Show data source breakdown
            blockchain_only = sum(1 for m in trading_markets if 'blockchain' in m['data_sources'] and len(m['data_sources']) == 1)
            api_only = sum(1 for m in trading_markets if 'api' in m['data_sources'] and len(m['data_sources']) == 1) 
            both = sum(1 for m in trading_markets if len(m['data_sources']) > 1)
            
            print(f"   📡 Sources: {blockchain_only} blockchain-only, {api_only} api-only, {both} combined")
            
            # Calculate edges
            print(f"\n📈 Calculating edges for {len(trading_markets)} markets...")
            signals = edge_calculator.calculate_edges(trading_markets)
            print(f"   Generated {len(signals)} signals")
            
            # Run continuous optimization
            result = await continuous_optimizer.optimize_and_execute_async(
                session_id,
                trading_markets,
                signals
            )
            
            if result['success']:
                print(f"\n✅ Optimization complete!")
                if result.get('new_positions'):
                    print(f"   New positions: {result['new_positions']}")
                if result.get('rebalanced_positions'):
                    print(f"   Rebalanced: {result['rebalanced_positions']}")
                if result.get('closed_positions'):
                    print(f"   Closed: {result['closed_positions']}")
                    
                # Show portfolio summary
                print(f"\n📊 Portfolio Summary:")
                print(f"   Total positions: {result['total_positions']}")
                print(f"   Positions value: ${result['positions_value']:.2f}")
                print(f"   Cash balance: ${result['cash_balance']:.2f}")
                
                # Show sample positions with data sources
                positions = session_manager.get_positions(session_id)
                open_positions = [p for p in positions if p['status'] == 'pending']
                
                if open_positions:
                    print(f"\n📋 Current Positions (showing first 5):")
                    for i, pos in enumerate(open_positions[:5]):
                        print(f"\n   {i+1}. {pos['home_team']} vs {pos['away_team']}")
                        print(f"      Bet: {pos['bet_on'].upper()} @ {pos['odds']}")
                        print(f"      Stake: ${pos['stake']}")
                        
                        # Find market to show data source
                        for market in trading_markets:
                            if market['home_team'] == pos['home_team'] and market['away_team'] == pos['away_team']:
                                sources = ', '.join(market['data_sources'])
                                print(f"      Data: {sources}")
                                if market.get('blockchain_id'):
                                    print(f"      Blockchain ID: {market['blockchain_id'][:20]}...")
                                break
            else:
                print(f"\n❌ Optimization failed: {result.get('error', 'Unknown error')}")
            
            # Wait between cycles
            print(f"\n⏳ Waiting 30 seconds before next cycle...")
            await asyncio.sleep(30)
            
        except KeyboardInterrupt:
            print("\n\n⛔ Trading stopped by user")
            break
        except Exception as e:
            print(f"\n❌ Error in trading cycle: {e}")
            traceback.print_exc()
            print("⏳ Waiting 60 seconds before retry...")
            await asyncio.sleep(60)
    
    print("\n👋 Trading session ended")


async def main():
    """Main entry point"""
    try:
        await run_continuous_trading()
    except KeyboardInterrupt:
        print("\n⛔ Stopped by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())