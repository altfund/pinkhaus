#!/usr/bin/env python3
"""
Automated Trading System for Ominari
Runs continuous backtesting and paper trading with real portfolio tracking
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
from sqlalchemy import func

from blockchain_reader import BlockchainReader
from database_v2 import db_manager
from models import Market, Odd, BettingSession, Bet, ChainSyncState
from paper_trading_engine import PaperTradingEngine
from signals import ImpliedRawSignal, VolumeWeightedSignal, BlockchainEnhancedSignal
from web_dashboard_real_odds import calculate_edge, get_active_positions
from trading_strategies import create_trading_strategies
from config.bankroll_config import BankrollConfig
from notifications.slack_notifier import slack_notifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/automated_trading.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PortfolioManager:
    """Manages the trading portfolio with real tracking"""
    
    def __init__(self, bankroll_config: Optional[BankrollConfig] = None):
        self.bankroll_config = bankroll_config or BankrollConfig()
        self.initial_bankroll = self.bankroll_config.get_current_bankroll()
        self.current_bankroll = self.initial_bankroll
        self.positions = {}
        self.performance_history = []
        self.daily_stats = {'date': datetime.now().date(), 'trades': 0, 'pnl': 0.0}
        
    def update_position(self, market_id: str, bet: Dict):
        """Update or create a position"""
        self.positions[market_id] = {
            'bet_id': bet.get('id'),
            'amount': bet.get('amount'),
            'odds': bet.get('odds'),
            'outcome': bet.get('outcome'),
            'timestamp': bet.get('timestamp'),
            'edge': bet.get('edge', 0),
            'status': 'active'
        }
        
    def close_position(self, market_id: str, won: bool, payout: float = 0):
        """Close a position and update bankroll"""
        if market_id in self.positions:
            position = self.positions[market_id]
            position['status'] = 'closed'
            position['won'] = won
            position['payout'] = payout
            
            # Calculate PnL
            pnl = payout - position['amount'] if won else -position['amount']
            
            # Update bankroll
            if won:
                self.current_bankroll += payout - position['amount']
            else:
                self.current_bankroll -= position['amount']
                
            # Update bankroll config with real tracking
            self.bankroll_config.record_bet_result(won, pnl)
            
            # Update daily stats
            self.daily_stats['trades'] += 1
            self.daily_stats['pnl'] += pnl
            
            # Send Slack notification for significant trades
            if abs(pnl) > 100 or abs(position['edge']) > 10:
                slack_notifier.send_trade_alert({
                    'type': 'CLOSE',
                    'market': market_id,
                    'outcome': position['outcome'],
                    'amount': position['amount'],
                    'odds': position['odds'],
                    'edge': position['edge'],
                    'won': won,
                    'pnl': pnl,
                    'bankroll': self.current_bankroll
                })
                
            # Track performance
            self.performance_history.append({
                'timestamp': datetime.now(),
                'market_id': market_id,
                'pnl': payout - position['amount'] if won else -position['amount'],
                'bankroll': self.current_bankroll,
                'roi': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
            })
            
    def get_portfolio_stats(self):
        """Get current portfolio statistics"""
        active_positions = [p for p in self.positions.values() if p['status'] == 'active']
        total_exposure = sum(p['amount'] for p in active_positions)
        
        return {
            'bankroll': self.current_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'pnl': self.current_bankroll - self.initial_bankroll,
            'roi': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100,
            'active_positions': len(active_positions),
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / self.current_bankroll * 100) if self.current_bankroll > 0 else 0,
            'win_rate': self._calculate_win_rate(),
            'avg_edge': self._calculate_avg_edge()
        }
        
    def _calculate_win_rate(self):
        """Calculate win rate from closed positions"""
        closed = [p for p in self.positions.values() if p['status'] == 'closed']
        if not closed:
            return 0
        wins = sum(1 for p in closed if p.get('won', False))
        return (wins / len(closed)) * 100
        
    def _calculate_avg_edge(self):
        """Calculate average edge of all positions"""
        edges = [p.get('edge', 0) for p in self.positions.values()]
        return sum(edges) / len(edges) if edges else 0


class AutomatedTradingSystem:
    """Main automated trading system"""
    
    def __init__(self):
        self.bankroll_config = BankrollConfig()
        self.portfolio = PortfolioManager(self.bankroll_config)
        self.paper_engine = PaperTradingEngine()
        self.blockchain_reader = BlockchainReader()
        self.is_running = False
        
        # Send startup notification
        slack_notifier.send_startup_message()
        
        # Trading parameters
        self.min_edge_threshold = 2.0  # Minimum 2% edge to place bet
        self.max_exposure_pct = 25.0   # Max 25% of bankroll at risk
        self.kelly_fraction = 0.25     # Conservative Kelly
        
        # Initialize signals
        self.signals = [
            ImpliedRawSignal(),
            VolumeWeightedSignal(),
            BlockchainEnhancedSignal()
        ]
        
        # Initialize advanced trading strategies
        self.strategies = create_trading_strategies()
        self.active_strategy = 'ensemble'  # Use ensemble by default
        
    async def start(self):
        """Start the automated trading system"""
        logger.info("Starting Automated Trading System...")
        self.is_running = True
        
        # Create initial paper trading session
        await self._create_trading_session()
        
        # Start concurrent tasks
        await asyncio.gather(
            self._blockchain_sync_loop(),
            self._backtesting_loop(),
            self._paper_trading_loop(),
            self._monitoring_loop()
        )
        
    async def _create_trading_session(self):
        """Create a new paper trading session"""
        with db_manager.get_db_session() as db:
            session = BettingSession(
                bankroll=self.portfolio.current_bankroll,
                created_at=datetime.now(),
                strategy_name="automated_trading_v1",
                min_bet_size=1.0,
                max_bet_size=self.portfolio.current_bankroll * 0.1,  # 10% max bet
                max_exposure=self.portfolio.current_bankroll * self.max_exposure_pct / 100,
                is_paper=True
            )
            db.add(session)
            db.commit()
            self.current_session_id = session.id
            logger.info(f"Created trading session {session.id}")
            
    async def _blockchain_sync_loop(self):
        """Continuously sync blockchain data"""
        logger.info("Starting blockchain sync loop...")
        
        while self.is_running:
            try:
                # Run blockchain sync
                logger.info("Syncing blockchain data...")
                await asyncio.to_thread(self.blockchain_reader.run_daemon_mode)
                
                # Wait before next sync
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Blockchain sync error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
                
    async def _backtesting_loop(self):
        """Run periodic backtests to validate strategies"""
        logger.info("Starting backtesting loop...")
        
        while self.is_running:
            try:
                # Run backtest on recent data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=7)  # Last 7 days
                
                logger.info(f"Running backtest from {start_date} to {end_date}")
                
                # Run vectorized backtest
                from vectorized_backtest import run_backtest
                results = await asyncio.to_thread(
                    run_backtest,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    initial_bankroll=10000,
                    strategies=['implied_raw', 'volume_weighted', 'blockchain_enhanced']
                )
                
                # Log backtest results
                for strategy, metrics in results.items():
                    logger.info(f"Backtest {strategy}: ROI={metrics.get('roi', 0):.2f}%, Sharpe={metrics.get('sharpe', 0):.2f}")
                
                # Wait before next backtest
                await asyncio.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Backtesting error: {e}")
                await asyncio.sleep(600)  # Retry after 10 minutes
                
    async def _paper_trading_loop(self):
        """Main paper trading loop"""
        logger.info("Starting paper trading loop...")
        
        while self.is_running:
            try:
                # Get active markets with edges
                markets_with_edges = await self._find_trading_opportunities()
                
                if markets_with_edges:
                    logger.info(f"Found {len(markets_with_edges)} trading opportunities")
                    
                    # Place bets on high-edge markets
                    for market_data in markets_with_edges:
                        await self._place_paper_bet(market_data)
                        
                # Check and settle completed markets
                await self._settle_completed_markets()
                
                # Log portfolio status
                stats = self.portfolio.get_portfolio_stats()
                logger.info(f"Portfolio: ${stats['bankroll']:.2f} | PnL: ${stats['pnl']:.2f} | ROI: {stats['roi']:.2f}% | Positions: {stats['active_positions']}")
                
                # Wait before next trading cycle
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Paper trading error: {e}")
                await asyncio.sleep(30)  # Retry after 30 seconds
                
    async def _find_trading_opportunities(self) -> List[Dict]:
        """Find markets with positive edge using advanced strategies"""
        opportunities = []
        
        with db_manager.get_db_session() as db:
            # Get active markets
            active_markets = db.query(Market).filter(
                Market.is_active == True,
                Market.sport == 'Soccer',
                Market.start_time > datetime.now()
            ).limit(100).all()
            
            strategy = self.strategies[self.active_strategy]
            
            for market in active_markets:
                # Get latest odds
                odds_query = db.query(Odd).filter(
                    Odd.market_id == market.id
                ).order_by(Odd.created_at.desc()).limit(3).all()
                
                if len(odds_query) >= 2:
                    # Get unique outcomes
                    odds_by_outcome = {}
                    for odd in odds_query:
                        if odd.outcome not in odds_by_outcome:
                            odds_by_outcome[odd.outcome] = odd
                    
                    # Evaluate with strategy
                    signal = strategy.evaluate_opportunity(market, odds_by_outcome)
                    
                    if signal:
                        signal['market'] = market
                        opportunities.append(signal)
                            
        # Sort by edge descending
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        return opportunities[:10]  # Top 10 opportunities
        
    async def _place_paper_bet(self, market_data: Dict):
        """Place a paper bet on a market"""
        try:
            market = market_data['market']
            
            # Check if we already have a position
            if market.id in self.portfolio.positions:
                return
                
            # Use strategy-specific bet sizing
            strategy = self.strategies[market_data.get('strategy', self.active_strategy)]
            bet_amount = strategy.calculate_bet_size(
                edge=market_data['edge'],
                odds=market_data['odds'],
                bankroll=self.portfolio.current_bankroll
            )
            
            # Check exposure limits
            current_exposure = sum(p['amount'] for p in self.portfolio.positions.values() if p['status'] == 'active')
            if current_exposure + bet_amount > self.portfolio.current_bankroll * self.max_exposure_pct / 100:
                logger.warning(f"Exposure limit reached, skipping bet on {market.id}")
                return
                
            # Place bet
            with db_manager.get_db_session() as db:
                bet = Bet(
                    betting_session_id=self.current_session_id,
                    market_id=market.id,
                    sport=market.sport,
                    league=market.league,
                    home_team=market.home,
                    away_team=market.away,
                    outcome=market_data['outcome'],
                    stake=bet_amount,
                    odds=odds,
                    placed_at=datetime.now(),
                    status="pending",
                    is_paper=True
                )
                db.add(bet)
                db.commit()
                
                # Update portfolio
                self.portfolio.update_position(market.id, {
                    'id': bet.id,
                    'amount': bet_amount,
                    'odds': odds,
                    'outcome': market_data['outcome'],
                    'timestamp': datetime.now(),
                    'edge': market_data['edge']
                })
                
                logger.info(f"Placed bet: ${bet_amount:.2f} on {market.home} vs {market.away} - {market_data['outcome']} @ {odds:.2f} (edge: {market_data['edge']:.2f}%)")
                
        except Exception as e:
            logger.error(f"Error placing bet: {e}")
            
    async def _settle_completed_markets(self):
        """Check and settle completed markets"""
        with db_manager.get_db_session() as db:
            # Get pending bets
            pending_bets = db.query(Bet).filter(
                Bet.betting_session_id == self.current_session_id,
                Bet.status == "pending"
            ).all()
            
            for bet in pending_bets:
                market = db.query(Market).filter(Market.id == bet.market_id).first()
                
                if market and not market.is_active and market.winning_outcome:
                    # Settle bet
                    won = (bet.outcome == market.winning_outcome)
                    payout = bet.stake * bet.odds if won else 0
                    
                    bet.status = "won" if won else "lost"
                    bet.payout = payout
                    bet.resolved_at = datetime.now()
                    
                    # Update portfolio
                    self.portfolio.close_position(market.id, won, payout)
                    
                    logger.info(f"Settled bet {bet.id}: {'WON' if won else 'LOST'} - ${payout:.2f}")
                    
            db.commit()
            
    async def _monitoring_loop(self):
        """Monitor system health and performance"""
        logger.info("Starting monitoring loop...")
        
        while self.is_running:
            try:
                # Log system stats
                stats = self.portfolio.get_portfolio_stats()
                
                # Create monitoring report
                report = {
                    'timestamp': datetime.now().isoformat(),
                    'portfolio': stats,
                    'health': await self._check_system_health()
                }
                
                # Save to file
                with open('logs/trading_performance.jsonl', 'a') as f:
                    f.write(json.dumps(report) + '\n')
                    
                # Alert if significant drawdown
                if stats['roi'] < -10:  # 10% drawdown
                    logger.warning(f"ALERT: Significant drawdown detected! ROI: {stats['roi']:.2f}%")
                    
                # Wait before next check
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _check_system_health(self) -> Dict:
        """Check system health status"""
        health = {
            'database': False,
            'blockchain_sync': False,
            'paper_trading': False
        }
        
        try:
            # Check database
            with db_manager.get_db_session() as db:
                db.execute("SELECT 1")
                health['database'] = True
                
            # Check blockchain sync
            with db_manager.get_db_session() as db:
                last_sync = db.query(ChainSyncState).order_by(ChainSyncState.last_updated.desc()).first()
                if last_sync and (datetime.now() - last_sync.last_updated).seconds < 600:  # Within 10 mins
                    health['blockchain_sync'] = True
                    
            # Check paper trading
            health['paper_trading'] = self.is_running and hasattr(self, 'current_session_id')
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            
        return health
        
    async def stop(self):
        """Stop the automated trading system"""
        logger.info("Stopping Automated Trading System...")
        self.is_running = False
        
        # Final portfolio report
        stats = self.portfolio.get_portfolio_stats()
        logger.info("Final Portfolio Report:")
        logger.info(f"  Initial Bankroll: ${stats['initial_bankroll']:.2f}")
        logger.info(f"  Final Bankroll: ${stats['bankroll']:.2f}")
        logger.info(f"  Total PnL: ${stats['pnl']:.2f}")
        logger.info(f"  ROI: {stats['roi']:.2f}%")
        logger.info(f"  Win Rate: {stats['win_rate']:.2f}%")
        logger.info(f"  Avg Edge: {stats['avg_edge']:.2f}%")


async def main():
    """Main entry point"""
    # Create logs directory
    os.makedirs('logs', exist_ok=True)
    
    # Initialize system
    trading_system = AutomatedTradingSystem()
    
    try:
        # Start automated trading
        await trading_system.start()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    finally:
        await trading_system.stop()


if __name__ == "__main__":
    asyncio.run(main())