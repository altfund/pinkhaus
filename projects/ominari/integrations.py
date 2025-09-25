#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Layer
Connects all system components into cohesive workflows.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
import pandas as pd
from sqlalchemy.orm import Session
import json

# Local imports
from config import get_settings
from database_v2 import db_manager
from models import Market, Odd
from signals import SIGNAL_PROVIDERS
from signal_registry import (
    SignalRegistry, DynamicWeightManager, SignalAggregator,
    BaseSignalProvider, SignalPerformance
)
from alpha_research_pipeline import AlphaResearchPipeline, AlphaSignal
from carver_framework import SystematicFramework, TradingRule
from paper_trading_engine import PaperTradingEngine, PaperOrder
from graphql_client import GraphQLAggregator
from blockchain_reader import BlockchainReader

logger = logging.getLogger(__name__)


class SystemIntegrator:
    """
    Main integration class that orchestrates all components.
    Implements workflows for data → signals → trading → monitoring.
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Initialize components
        self.signal_registry = SignalRegistry()
        self.weight_manager = DynamicWeightManager(self.signal_registry)
        self.signal_aggregator = SignalAggregator(self.signal_registry, self.weight_manager)
        self.alpha_pipeline = AlphaResearchPipeline()
        self.paper_engine = PaperTradingEngine(
            initial_capital=self.settings.trading.initial_capital,
            commission_rate=self.settings.trading.commission_rate
        )
        
        # Carver framework
        self.systematic_framework = SystematicFramework(
            capital=self.settings.trading.initial_capital,
            volatility_target=self.settings.trading.volatility_target
        )
        
        # Data sources
        logger.info(f"GraphQL streaming enabled: {self.settings.features.graphql_streaming}")
        logger.info(f"Blockchain sync enabled: {self.settings.features.blockchain_sync}")
        
        if self.settings.features.graphql_streaming:
            logger.info("Initializing GraphQL aggregator")
            self.graphql_aggregator = GraphQLAggregator()
            
        if self.settings.features.blockchain_sync:
            logger.info("Initializing blockchain reader")
            self.blockchain_reader = BlockchainReader(
                network=self.settings.blockchain.network
            )
            
        # Initialize default signals
        self._initialize_signals()
        
    def _initialize_signals(self):
        """Initialize and register default signals."""
        logger.info("Initializing signal providers")
        
        # Register built-in signals
        for provider_instance in SIGNAL_PROVIDERS:
            try:
                # Create instance with registry-compatible interface
                provider = self._wrap_legacy_signal(provider_instance)
                self.signal_registry.register(provider)
                logger.info(f"Registered signal: {provider.name}")
            except Exception as e:
                logger.error(f"Failed to register {provider_instance.name}: {e}")
                
        # Initialize Carver trading rules
        self._initialize_trading_rules()
        
    def _wrap_legacy_signal(self, provider_instance):
        """Wrap legacy signal providers to work with new registry."""
        
        class WrappedSignal(BaseSignalProvider):
            def __init__(self, legacy_provider):
                self.legacy_provider = legacy_provider
                # Create instance of legacy provider if it's a class
                if isinstance(legacy_provider, type):
                    self.legacy_instance = legacy_provider()
                    signal_name = getattr(self.legacy_instance, 'name', legacy_provider.__name__)
                else:
                    self.legacy_instance = legacy_provider
                    signal_name = getattr(legacy_provider, 'name', 'unknown')
                    
                super().__init__(
                    name=signal_name,
                    version="1.0.0"
                )
                
            def get_probs(self, df: pd.DataFrame) -> pd.Series:
                return self.legacy_instance.get_probs(df)
                
            def get_parameters(self) -> Dict[str, Any]:
                return {}
                
            def get_required_columns(self) -> List[str]:
                # Infer from legacy provider
                return ['bet_name', 'source_id', 'implied_raw']
                
        return WrappedSignal(provider_instance)
        
    def _initialize_trading_rules(self):
        """Initialize Carver framework trading rules."""
        # Momentum rules
        self.systematic_framework.add_trading_rule(TradingRule(
            name='momentum_20',
            forecast_scalar=8.0,
            turnover=12.0
        ))
        
        self.systematic_framework.add_trading_rule(TradingRule(
            name='momentum_50',
            forecast_scalar=4.0,
            turnover=6.0
        ))
        
        # Value/carry rules
        self.systematic_framework.add_trading_rule(TradingRule(
            name='carry',
            forecast_scalar=10.0,
            turnover=4.0
        ))
        
    async def run_data_collection_cycle(self):
        """Run a complete data collection cycle."""
        logger.info("Starting data collection cycle")
        
        tasks = []
        
        # Collect from multiple sources in parallel
        if self.settings.features.blockchain_sync:
            logger.info("Adding blockchain data collection task")
            tasks.append(self._collect_blockchain_data())
            
        if self.settings.features.graphql_streaming:
            logger.info("Adding GraphQL data collection task")
            tasks.append(self._collect_graphql_data())
            
        # Always collect from API
        logger.info("Adding REST API data collection task")
        tasks.append(self._collect_api_data())
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Data collection task {i} failed: {result}")
                
        logger.info("Data collection cycle complete")
        
    async def _collect_blockchain_data(self):
        """Collect data from blockchain."""
        try:
            # Get recent blocks
            current_block = self.blockchain_reader.w3.eth.block_number
            from_block = current_block - 100  # Last 100 blocks
            
            # Scan for new markets
            markets = self.blockchain_reader.scan_market_creations(
                from_block, current_block
            )
            logger.info(f"Found {len(markets)} new blockchain markets")
            
            # Scan for trades
            trades = self.blockchain_reader.scan_trades(
                from_block, current_block
            )
            logger.info(f"Found {len(trades)} blockchain trades")
            
        except Exception as e:
            logger.error(f"Blockchain data collection failed: {e}")
            raise
            
    async def _collect_graphql_data(self):
        """Collect data from GraphQL endpoints."""
        try:
            df = await self.graphql_aggregator.get_all_active_markets()
            logger.info(f"Collected {len(df)} markets from GraphQL")
            
            # Store in database
            with db_manager.get_db_session() as db:
                # Convert to market objects and store
                pass
                
        except Exception as e:
            logger.error(f"GraphQL data collection failed: {e}")
            raise
            
    async def _collect_api_data(self):
        """Collect data from REST API."""
        # This would call free_data_pull.py functionality
        logger.info("Collecting data from REST API")
        
    async def run_signal_generation_cycle(self):
        """Generate signals for all active markets."""
        logger.info("Starting signal generation cycle")
        
        with db_manager.get_db_session() as db:
            # Get active markets
            current_time = datetime.now(timezone.utc)
            try:
                markets = db.query(Market).filter(
                    Market.maturity_date > current_time,
                    Market.maturity_date < current_time + timedelta(hours=24)
                ).all()
            except ValueError as e:
                logger.error(f"Error querying markets: {e}")
                markets = []
            
            logger.info(f"Generating signals for {len(markets)} markets")
            
            # Prepare data for signal generation
            market_data = self._prepare_market_data(markets, db)
            
            if market_data.empty:
                logger.warning("No market data available")
                return
                
            # Generate signals
            predictions = self.signal_aggregator.get_combined_predictions(market_data)
            
            # Store signal performance
            self._track_signal_performance(predictions, market_data)
            
            return predictions, market_data
            
    def _prepare_market_data(self, markets: List[Market], db: Session) -> pd.DataFrame:
        """Prepare market data for signal generation."""
        data = []
        
        for market in markets:
            # Get latest odds
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            for odd in odds:
                data.append({
                    'market_id': market.source_id,
                    'source_id': market.source_id,
                    'match_id': market.source_id,  # Using source_id as match identifier
                    'sport': market.sport,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'bet_name': odd.outcome,
                    'normalized_outcome': odd.outcome,  # Signal providers need this
                    'odds': odd.decimal_odds,
                    'implied_raw': 100.0 / odd.decimal_odds if odd.decimal_odds and odd.decimal_odds > 0 else 0,
                    'timestamp': odd.updated_at,
                    'commence_time': market.maturity_date,
                    'maturity_date': market.maturity_date,  # GrantSignal needs this
                    'time': odd.updated_at,  # Some signals use 'time' field
                    'league_name': market.league_name or 'Unknown'
                })
                
        return pd.DataFrame(data)
        
    def _track_signal_performance(self, predictions: pd.Series, 
                                market_data: pd.DataFrame):
        """Track performance of signal predictions."""
        # This would compare predictions to actual outcomes
        # For now, simulate performance tracking
        
        for signal_name in self.signal_registry.list_signals():
            perf = SignalPerformance(
                signal_name=signal_name,
                timestamp=datetime.now(timezone.utc),
                period='hourly',
                n_predictions=len(predictions),
                accuracy=0.52,  # Would calculate from outcomes
                sharpe_ratio=0.8,
                information_ratio=0.5,
                max_drawdown=-0.05,
                correlation_with_others={},
                regime=self._detect_market_regime()
            )
            
            self.signal_registry.update_performance(signal_name, perf)
            
    def _detect_market_regime(self) -> str:
        """Detect current market regime."""
        hour = datetime.now(timezone.utc).hour
        
        if 0 <= hour < 6:
            return "overnight"
        elif 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        else:
            return "evening"
            
    async def run_trading_cycle(self, predictions: pd.Series = None, market_data: pd.DataFrame = None):
        """Execute trading decisions based on signals."""
        logger.info("Starting trading cycle")
        
        if predictions is None or predictions.empty:
            logger.warning("No predictions available for trading")
            return
            
        # Apply position sizing
        positions = self._calculate_positions(predictions, market_data)
        
        # Execute trades
        if self.settings.features.paper_trading:
            await self._execute_paper_trades(positions)
            
        if self.settings.features.live_trading:
            await self._execute_live_trades(positions)
            
    def _calculate_positions(self, predictions: pd.Series, market_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate position sizes using Kelly criterion."""
        positions = pd.DataFrame(index=predictions.index)
        
        positions['probability'] = predictions
        
        # Get odds from market_data if predictions don't have multi-index
        if hasattr(predictions.index, 'levels') and 'odds' in predictions.index.names:
            odds = predictions.index.get_level_values('odds')
        elif market_data is not None and 'odds' in market_data.columns:
            odds = market_data.loc[predictions.index, 'odds']
        else:
            logger.warning("No odds data available for position sizing")
            return pd.DataFrame()  # Return empty if no odds
            
        positions['odds'] = odds
        positions['edge'] = predictions - (1.0 / odds)
        
        # Simple Kelly sizing
        positions['kelly_fraction'] = positions['edge'] / (odds - 1)
        max_pos_size = getattr(self.settings.trading, 'max_position_size', 0.25)
        positions['kelly_fraction'] = positions['kelly_fraction'].clip(0, max_pos_size)
        
        # Apply capital allocation
        total_capital = getattr(self.settings.trading, 'initial_capital', 10000.0)
        positions['bet_size'] = positions['kelly_fraction'] * total_capital
        
        # Filter by minimum bet size
        min_bet = getattr(self.settings.trading, 'min_bet_size', 1.0)
        positions = positions[positions['bet_size'] >= min_bet]
        
        return positions
        
    async def _execute_paper_trades(self, positions: pd.DataFrame):
        """Execute paper trades."""
        for idx, position in positions.iterrows():
            if position['bet_size'] > 0:
                order = PaperOrder(
                    order_id=f"SYS_{datetime.now().timestamp()}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=idx[0],  # Assuming multi-index
                    market_type="winner",
                    bet_name=idx[1],
                    side="buy",
                    size=position['bet_size'],
                    limit_price=None,
                    signal_name="aggregated",
                    expected_edge=position['edge']
                )
                
                fill = await self.paper_engine.submit_order(order)
                if fill:
                    logger.info(f"Paper trade executed: {fill.fill_size} @ {fill.fill_price}")
                    
    async def _execute_live_trades(self, positions: pd.DataFrame):
        """Execute live trades."""
        logger.warning("Live trading not implemented - would execute real trades here")
        
    def run_alpha_research_cycle(self):
        """Run alpha research pipeline for signal development."""
        logger.info("Running alpha research cycle")
        
        # Get signals in research
        for signal_id, signal in self.alpha_pipeline.signals.items():
            if signal.current_stage in ['raw_rd', 'in_sample', 'out_of_sample']:
                # Run appropriate backtest
                self._run_research_backtest(signal)
                
                # Check for promotion
                can_promote, reason = self.alpha_pipeline.evaluate_stage_progression(signal_id)
                if can_promote:
                    logger.info(f"Promoting {signal_id}: {reason}")
                    self.alpha_pipeline.promote_signal(signal_id, reason)
                    
    def _run_research_backtest(self, signal: AlphaSignal):
        """Run backtest for research signal."""
        # Get historical data
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=90)
        
        with db_manager.get_db_session() as db:
            # Load data (simplified)
            data = pd.DataFrame()  # Would load actual data
            
            result = self.alpha_pipeline.run_backtest(
                signal.signal_id,
                signal.current_stage,
                data,
                start_date,
                end_date
            )
            
            logger.info(f"Research backtest for {signal.name}: "
                       f"Sharpe={result.sharpe_ratio:.2f}, "
                       f"p-value={result.p_value:.3f}")
                       
    async def run_monitoring_cycle(self):
        """Run monitoring and alerting cycle."""
        logger.info("Running monitoring cycle")
        
        # Collect metrics
        metrics = self._collect_system_metrics()
        
        # Check alerts
        alerts = self._check_alert_conditions(metrics)
        
        if alerts:
            await self._send_alerts(alerts)
            
        # Export metrics
        if self.settings.monitoring.enabled:
            self._export_metrics(metrics)
            
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system performance metrics."""
        metrics = {
            'timestamp': datetime.now(timezone.utc),
            'signals': {
                'active': len(self.signal_registry.list_signals()),
                'weights': self.weight_manager.get_current_weights()
            },
            'trading': {
                'paper_trades': len(self.paper_engine.fills),
                'paper_pnl': self.paper_engine.calculate_performance().get('total_pnl', 0)
            },
            'research': {
                'signals_in_research': len(self.alpha_pipeline.signals),
                'signals_by_stage': self._count_signals_by_stage()
            }
        }
        
        return metrics
        
    def _count_signals_by_stage(self) -> Dict[str, int]:
        """Count signals by research stage."""
        counts = {}
        for signal in self.alpha_pipeline.signals.values():
            counts[signal.current_stage] = counts.get(signal.current_stage, 0) + 1
        return counts
        
    def _check_alert_conditions(self, metrics: Dict[str, Any]) -> List[str]:
        """Check for alert conditions."""
        alerts = []
        
        # Check drawdown
        paper_pnl = metrics['trading']['paper_pnl']
        if paper_pnl < -self.settings.trading.max_drawdown * self.settings.trading.initial_capital:
            alerts.append(f"Max drawdown exceeded: {paper_pnl}")
            
        # Check signal health
        if len(self.signal_registry.list_signals()) == 0:
            alerts.append("No active signals available")
            
        return alerts
        
    async def _send_alerts(self, alerts: List[str]):
        """Send alerts via configured channels."""
        for alert in alerts:
            logger.warning(f"ALERT: {alert}")
            
            if self.settings.monitoring.slack_webhook:
                # Send to Slack
                pass
                
    def _export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to monitoring system."""
        # This would export to Prometheus/Datadog
        logger.debug(f"Exporting metrics: {metrics}")
        
    async def run_full_cycle(self):
        """Run a complete system cycle."""
        logger.info("Starting full system cycle")
        
        try:
            # 1. Collect data
            await self.run_data_collection_cycle()
            
            # 2. Generate signals
            result = await self.run_signal_generation_cycle()
            
            # 3. Execute trades
            if result is not None:
                predictions, market_data = result
                await self.run_trading_cycle(predictions, market_data)
                
            # 4. Run research
            self.run_alpha_research_cycle()
            
            # 5. Monitor system
            await self.run_monitoring_cycle()
            
            logger.info("Full system cycle complete")
            
        except Exception as e:
            logger.error(f"System cycle failed: {e}", exc_info=True)
            
    def generate_daily_report(self) -> Dict[str, Any]:
        """Generate comprehensive daily report."""
        report = {
            'date': datetime.now(timezone.utc).date().isoformat(),
            'summary': self._collect_system_metrics(),
            'signal_performance': self._get_signal_performance_summary(),
            'trading_performance': self._get_trading_performance_summary(),
            'research_pipeline': self._get_research_summary(),
            'system_health': self._get_system_health()
        }
        
        # Save report
        report_path = self.settings.report_dir / f"daily_report_{report['date']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Daily report saved to {report_path}")
        return report
        
    def _get_signal_performance_summary(self) -> Dict[str, Any]:
        """Get signal performance summary."""
        summary = {}
        
        for signal_name in self.signal_registry.list_signals():
            signal = self.signal_registry.get_signal(signal_name)
            if signal and hasattr(signal, 'metadata'):
                summary[signal_name] = signal.metadata.performance_stats
                
        return summary
        
    def _get_trading_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        return self.paper_engine.calculate_performance()
        
    def _get_research_summary(self) -> Dict[str, Any]:
        """Get research pipeline summary."""
        return {
            'total_signals': len(self.alpha_pipeline.signals),
            'by_stage': self._count_signals_by_stage(),
            'recent_promotions': []  # Would track recent stage transitions
        }
        
    def _get_system_health(self) -> Dict[str, bool]:
        """Get system health status."""
        return {
            'database': self._check_database_health(),
            'signals': len(self.signal_registry.list_signals()) > 0,
            'trading': self.paper_engine.current_capital > 0,
            'monitoring': self.settings.monitoring.enabled
        }
        
    def _check_database_health(self) -> bool:
        """Check database connectivity."""
        try:
            from sqlalchemy import text
            with db_manager.get_db_session() as db:
                db.execute(text("SELECT 1"))
                db.commit()
            return True
        except Exception:
            return False


# Convenience functions for common workflows
async def run_paper_trading_session(duration_hours: int = 24):
    """Run a paper trading session for specified duration."""
    integrator = SystemIntegrator()
    
    end_time = datetime.now(timezone.utc) + timedelta(hours=duration_hours)
    
    while datetime.now(timezone.utc) < end_time:
        await integrator.run_full_cycle()
        await asyncio.sleep(300)  # Run every 5 minutes
        
    # Generate report
    report = integrator.generate_daily_report()
    return report


def backtest_all_signals(start_date: datetime, end_date: datetime):
    """Run backtests for all registered signals."""
    integrator = SystemIntegrator()
    
    results = {}
    
    for signal_name in integrator.signal_registry.list_signals():
        try:
            # TODO: Adapt to use vectorized_backtest function
            # For now, return placeholder results
            results[signal_name] = {
                "sharpe_ratio": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "error": "Backtest integration pending"
            }
            
        except Exception as e:
            logger.error(f"Backtest failed for {signal_name}: {e}")
            results[signal_name] = {"error": str(e)}
            
    return results


async def main():
    """Example usage of integration layer."""
    logger.info("Starting Ominari system integration")
    
    # Create integrator
    integrator = SystemIntegrator()
    
    # Run one full cycle
    await integrator.run_full_cycle()
    
    # Generate report
    report = integrator.generate_daily_report()
    print(f"System report: {json.dumps(report, indent=2, default=str)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())