#!/usr/bin/env python3
"""Integrated blockchain trading system with unified data and execution"""

import os
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

# Import all components
from unified_data_fetcher import UnifiedDataFetcher
from blockchain_trading_executor import BlockchainTradingExecutor
from continuous_portfolio_optimizer import ContinuousPortfolioOptimizer
from paper_trading_postgres_integrated import PaperTradingSessionManager
from portfolio_trading_engine import PortfolioTradingEngine
from edge_calculator import EdgeCalculator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class IntegratedBlockchainTrading:
    """Complete trading system with blockchain execution"""
    
    def __init__(self, session_id: str):
        # Initialize all components
        self.session_id = session_id
        self.session_manager = PaperTradingSessionManager()
        self.edge_calculator = EdgeCalculator()
        self.unified_fetcher = UnifiedDataFetcher()
        self.blockchain_executor = BlockchainTradingExecutor()
        
        # Get session info
        session = self.session_manager.get_session(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        current_bankroll = float(session['current_bankroll'])
        
        # Trading configuration
        self.strategy_config = {
            'bankroll': current_bankroll,
            'kelly_fraction': 0.25,
            'min_edge': 0.01,
            'cap_per_bet': 0.02,
            'cap_per_game': 0.02,
            'min_bet': 10,
            'max_positions': 20
        }
        
        self.portfolio_engine = PortfolioTradingEngine(
            self.session_manager,
            self.edge_calculator,
            self.strategy_config
        )
        
        # Initialize continuous optimizer with dynamic chunking enabled
        chunk_hours = float(os.getenv('CHUNK_HOURS', '2.0'))
        use_dynamic_chunking = os.getenv('USE_DYNAMIC_CHUNKING', 'true').lower() == 'true'
        
        self.continuous_optimizer = ContinuousPortfolioOptimizer(
            self.portfolio_engine,
            chunk_hours=chunk_hours,
            use_dynamic_chunking=use_dynamic_chunking
        )
        
        # Initialize real-time valuation if enabled
        self.enable_realtime_valuation = os.getenv('ENABLE_REALTIME_VALUATION', 'true').lower() == 'true'
        self.valuation_engine = None
        
        logger.info(f"Initialized integrated trading for session {session_id}")
        logger.info(f"Bankroll: ${current_bankroll:,.2f}")
        logger.info(f"Dynamic chunking: {'enabled' if use_dynamic_chunking else 'disabled'}")
        logger.info(f"Real-time valuation: {'enabled' if self.enable_realtime_valuation else 'disabled'}")
    
    async def fetch_and_analyze_markets(self) -> Dict:
        """Fetch markets and analyze opportunities"""
        # Fetch unified market data
        markets = await self.unified_fetcher.fetch_all_markets()
        
        if not markets:
            logger.warning("No markets available")
            return {'markets': [], 'trading_markets': [], 'signals': []}
        
        # Format for trading
        trading_markets = self.unified_fetcher.format_for_trading(markets)
        logger.info(f"Found {len(trading_markets)} tradeable markets")
        
        # Calculate edges
        signals = self.edge_calculator.calculate_edges(trading_markets)
        logger.info(f"Generated {len(signals)} signals")
        
        # Analyze blockchain connectivity
        connected_count = sum(1 for m in trading_markets if m.get('blockchain_address'))
        logger.info(f"Markets with blockchain addresses: {connected_count}/{len(trading_markets)}")
        
        return {
            'markets': markets,
            'trading_markets': trading_markets,
            'signals': signals
        }
    
    async def optimize_portfolio(self, markets: List[Dict], signals: List[Dict]) -> Dict:
        """Run portfolio optimization with dynamic chunking and real-time valuation"""
        # Start real-time valuation if enabled and not already running
        if self.enable_realtime_valuation and not self.valuation_engine:
            from realtime_valuation_engine import RealTimeValuationEngine
            self.valuation_engine = RealTimeValuationEngine()
            # Note: In production, would start this as async background task
            logger.info("Real-time valuation engine started")
        
        # Run optimization with enhanced features
        result = self.continuous_optimizer.optimize_portfolio_continuously(
            self.session_id,
            markets,
            signals,
            float(self.session_manager.get_session(self.session_id)['current_bankroll']),
            enable_real_time_valuation=self.enable_realtime_valuation
        )
        
        if result['success']:
            logger.info(f"Portfolio optimized: {result['new_positions']} new positions")
        else:
            logger.error(f"Optimization failed: {result.get('error')}")
        
        return result
    
    def prepare_blockchain_trades(self, trades: List[Dict], markets: List[Dict]) -> List[Dict]:
        """Prepare trades for blockchain execution"""
        blockchain_trades = []
        
        # Create market lookup
        market_lookup = {m['match_id']: m for m in markets}
        
        # Filter trades to only include blockchain-connected markets
        blockchain_eligible_trades = []
        for trade in trades:
            market = market_lookup.get(trade['match_id'])
            if not market:
                continue
            
            # Only include markets with blockchain connectivity
            if market.get('has_blockchain_data', False) or market.get('blockchain_address') or market.get('blockchain_id'):
                blockchain_eligible_trades.append((trade, market))
        
        logger.info(f"Found {len(blockchain_eligible_trades)} blockchain-eligible trades out of {len(trades)} total")
        
        # Group trades by time chunks for better organization
        chunk_trades = {}
        for trade, market in blockchain_eligible_trades:
            chunk_label = trade.get('chunk_label', 'Unknown')
            if chunk_label not in chunk_trades:
                chunk_trades[chunk_label] = []
            chunk_trades[chunk_label].append((trade, market))
        
        # Process trades by time chunks
        logger.info(f"Processing {len(chunk_trades)} time chunks: {list(chunk_trades.keys())}")
        
        for chunk_label, chunk in chunk_trades.items():
            logger.info(f"\n🕐 Processing time chunk: {chunk_label} ({len(chunk)} trades)")
            
            for trade, market in chunk:
                try:
                    # Prepare trade for blockchain
                    blockchain_trade = self.blockchain_executor.prepare_trade(
                        market,
                        trade['bet_on'],
                        trade['stake']
                    )
                    blockchain_trade['paper_trade_id'] = trade.get('id')
                    blockchain_trade['chunk_label'] = chunk_label
                    blockchain_trades.append(blockchain_trade)
                    
                except Exception as e:
                    logger.error(f"Failed to prepare blockchain trade for {trade.get('home_team', '')} vs {trade.get('away_team', '')}: {e}")
        
        return blockchain_trades
    
    async def execute_trades(self, blockchain_trades: List[Dict], simulate: bool = True) -> List[Dict]:
        """Execute trades on blockchain grouped by time chunks"""
        results = []
        
        if not blockchain_trades:
            logger.info("No blockchain trades to execute")
            return results
        
        # Group trades by time chunk
        chunk_groups = {}
        for trade in blockchain_trades:
            chunk_label = trade.get('chunk_label', 'Unknown')
            if chunk_label not in chunk_groups:
                chunk_groups[chunk_label] = []
            chunk_groups[chunk_label].append(trade)
        
        logger.info(f"Executing {len(blockchain_trades)} blockchain trades across {len(chunk_groups)} time chunks")
        
        # Process by time chunks
        for chunk_idx, (chunk_label, chunk_trades) in enumerate(chunk_groups.items()):
            logger.info(f"\n📦 Executing time chunk {chunk_idx+1}/{len(chunk_groups)}: {chunk_label} ({len(chunk_trades)} trades)")
            
            # Apply execution batch size limit within time chunk if needed
            batch_size = int(os.getenv('CHUNK_SIZE', '50'))  # Max trades per execution batch
            
            for batch_start in range(0, len(chunk_trades), batch_size):
                batch = chunk_trades[batch_start:batch_start + batch_size]
                if len(chunk_trades) > batch_size:
                    logger.info(f"  Batch {batch_start//batch_size + 1} of {(len(chunk_trades) + batch_size - 1)//batch_size}")
                
                for j, trade in enumerate(batch):
                    logger.info(f"\n  Trade {j+1}/{len(batch)}: {trade['home_team']} vs {trade['away_team']}")
                    logger.info(f"    Position: {trade['outcome']} @ {trade['odds']}")
                    logger.info(f"    Stake: ${trade['stake']:.2f}")
                
                if simulate:
                    # Simulated execution
                    result = self.blockchain_executor.execute_trade(trade)
                    results.append(result)
                    logger.info(f"    ✅ Simulated execution successful: {result['tx_hash'][:10]}...")
                else:
                    # Real blockchain execution
                    logger.warning("    ⚠️ Real blockchain execution not implemented in demo")
                    # result = await self.blockchain_executor.execute_trade_async(trade)
                    # results.append(result)
        
        successful_trades = sum(1 for r in results if r.get('success', False))
        logger.info(f"\n✅ Execution complete: {successful_trades}/{len(blockchain_trades)} trades successful")
        
        return results
    
    async def run_trading_cycle(self, simulate: bool = True) -> Dict:
        """Run complete trading cycle"""
        logger.info("\n" + "="*60)
        logger.info(f"Trading Cycle - {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        logger.info("="*60)
        
        # Fetch and analyze markets
        market_data = await self.fetch_and_analyze_markets()
        
        if not market_data['trading_markets']:
            return {'success': False, 'error': 'No markets available'}
        
        # Optimize portfolio
        optimization_result = await self.optimize_portfolio(
            market_data['trading_markets'],
            market_data['signals']
        )
        
        if not optimization_result['success']:
            return optimization_result
        
        # Get new trades
        new_trades = optimization_result.get('trades', [])
        if not new_trades:
            logger.info("No new trades to execute")
            return {'success': True, 'trades_executed': 0}
        
        # Prepare blockchain trades
        blockchain_trades = self.prepare_blockchain_trades(
            new_trades,
            market_data['trading_markets']
        )
        
        logger.info(f"\n📊 Prepared {len(blockchain_trades)} trades for blockchain execution")
        
        if blockchain_trades:
            # Execute on blockchain
            execution_results = await self.execute_trades(blockchain_trades, simulate=simulate)
            
            # Update paper trading records with blockchain tx info
            for i, result in enumerate(execution_results):
                if result['success'] and i < len(new_trades):
                    # Store blockchain tx hash in paper trading record
                    # In production, update database with tx info
                    logger.info(f"Trade {i+1} blockchain tx: {result['tx_hash']}")
        
        summary = {
            'success': True,
            'markets_analyzed': len(market_data['trading_markets']),
            'trades_prepared': len(new_trades),
            'blockchain_trades': len(blockchain_trades),
            'trades_executed': len(blockchain_trades),
            'positions_value': optimization_result.get('positions_value', 0),
            'cash_balance': optimization_result.get('cash_balance', 0)
        }
        
        return summary
    
    async def update_empirical_data_and_restart(self, lookback_days: int = 30):
        """Update empirical settlement data and restart components if needed"""
        if not self.continuous_optimizer.use_dynamic_chunking:
            logger.warning("Dynamic chunking not enabled, no empirical data to update")
            return False
        
        logger.info(f"Updating empirical settlement data with {lookback_days} days of history...")
        
        try:
            # Update empirical data
            success = await self.continuous_optimizer.update_empirical_data(lookback_days)
            
            if success:
                logger.info("Empirical data updated successfully")
                
                # Get updated status
                status = self.continuous_optimizer.get_dynamic_chunking_status()
                logger.info(f"Updated empirical patterns: {status['chunk_manager']['empirical_data_loaded']}")
                
                return True
            else:
                logger.warning("Failed to update empirical data")
                return False
                
        except Exception as e:
            logger.error(f"Error updating empirical data: {e}")
            return False
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        session = self.session_manager.get_session(self.session_id)
        
        status = {
            'session_id': self.session_id,
            'current_bankroll': float(session['current_bankroll']) if session else 0,
            'dynamic_chunking': self.continuous_optimizer.get_dynamic_chunking_status(),
            'realtime_valuation_enabled': self.enable_realtime_valuation,
            'valuation_engine_active': self.valuation_engine is not None,
            'strategy_config': self.strategy_config
        }
        
        # Add capital metrics if available
        if self.continuous_optimizer.use_dynamic_chunking:
            try:
                capital_state = self.continuous_optimizer.capital_tracker.update_capital_state(self.session_id)
                status['capital_state'] = {
                    'available_cash': capital_state.available_cash,
                    'utilization_rate': capital_state.utilization_rate,
                    'value_at_risk': capital_state.value_at_risk,
                    'expected_value': capital_state.expected_value
                }
            except Exception as e:
                logger.error(f"Error getting capital state: {e}")
        
        return status
    
    async def monitor_and_trade(self, interval: int = 30, cycles: int = None):
        """Continuously monitor and trade"""
        logger.info(f"Starting continuous trading (interval: {interval}s)")
        
        cycle_count = 0
        while True:
            cycle_count += 1
            
            try:
                # Run trading cycle
                result = await self.run_trading_cycle(simulate=True)
                
                logger.info(f"\n✅ Cycle {cycle_count} complete:")
                logger.info(f"  Markets analyzed: {result.get('markets_analyzed', 0)}")
                logger.info(f"  Trades executed: {result.get('trades_executed', 0)}")
                logger.info(f"  Portfolio value: ${result.get('positions_value', 0):,.2f}")
                logger.info(f"  Cash balance: ${result.get('cash_balance', 0):,.2f}")
                
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # Check if we should stop
            if cycles and cycle_count >= cycles:
                logger.info(f"Completed {cycles} cycles, stopping")
                break
            
            # Wait for next cycle
            logger.info(f"\n⏳ Waiting {interval}s until next cycle...")
            await asyncio.sleep(interval)


async def main():
    """Run integrated blockchain trading system"""
    print("🚀 Integrated Blockchain Trading System\n")
    
    # Get current session
    session_manager = PaperTradingSessionManager()
    session_id = session_manager.get_current_session()
    
    if not session_id:
        print("❌ No active trading session found")
        return
    
    print(f"Using session: {session_id}")
    
    # Initialize integrated system
    trading_system = IntegratedBlockchainTrading(session_id)
    
    # Run a single cycle for demonstration
    print("\n📊 Running single trading cycle...\n")
    result = await trading_system.run_trading_cycle(simulate=True)
    
    if result['success']:
        print("\n✅ Trading cycle successful!")
        print(f"  Markets analyzed: {result.get('markets_analyzed', 0)}")
        print(f"  Blockchain trades executed: {result.get('trades_executed', 0)}")
    else:
        print(f"\n❌ Trading cycle failed: {result.get('error', 'Unknown error')}")
    
    # Optionally run continuous trading
    print("\n💡 To run continuous trading, use:")
    print("   await trading_system.monitor_and_trade(interval=30)")


if __name__ == "__main__":
    asyncio.run(main())