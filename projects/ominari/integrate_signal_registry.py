#!/usr/bin/env python3
"""
Integration of dynamic signal registry with main trading system.
Enables hot-pluggable signals with adaptive weighting strategies.
"""

import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from signal_registry import (
    SignalRegistry, 
    DynamicWeightManager, 
    SignalAggregator,
    BaseSignalProvider,
    SignalPerformance
)
from database_v2 import db_manager
from models import Bet, BettingSession
from signals import SignalProvider, ImpliedRawSignal, SIGNAL_PROVIDERS
from sqlalchemy import and_, func

logger = logging.getLogger(__name__)


class SignalProviderAdapter(BaseSignalProvider):
    """Adapts existing SignalProvider to work with registry."""
    
    def __init__(self, signal: SignalProvider):
        self.signal = signal
        super().__init__(signal.name, "1.0.0")
        
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """Delegate to wrapped signal."""
        return self.signal.get_probs(df)
        
    def get_parameters(self) -> Dict[str, any]:
        """Return signal parameters."""
        # Most existing signals don't have parameters
        return {}
        
    def get_required_columns(self) -> List[str]:
        """Return required columns based on signal type."""
        if "implied" in self.name.lower():
            return ['implied_raw']
        else:
            return ['source_id', 'normalized_outcome']


class PerformanceTracker:
    """Tracks signal performance and updates registry."""
    
    def __init__(self, registry: SignalRegistry):
        self.registry = registry
        
    def calculate_signal_performance(self, 
                                   signal_name: str,
                                   session_id: str,
                                   period: str = "daily") -> Optional[SignalPerformance]:
        """Calculate performance metrics for a signal."""
        with db_manager.get_db_session() as db:
            # Get bets made using this signal
            session = db.query(BettingSession).filter_by(session_id=session_id).first()
            if not session:
                return None
                
            # Get recent bets
            if period == "hourly":
                since = datetime.now(timezone.utc) - timedelta(hours=1)
            elif period == "daily":
                since = datetime.now(timezone.utc) - timedelta(days=1)
            else:  # weekly
                since = datetime.now(timezone.utc) - timedelta(days=7)
                
            bets = db.query(Bet).filter(
                and_(
                    Bet.session_id == session_id,
                    Bet.created_at >= since,
                    Bet.strategy_name == signal_name
                )
            ).all()
            
            if not bets:
                return None
                
            # Calculate metrics
            settled_bets = [b for b in bets if b.settled]
            if not settled_bets:
                return None
                
            winning_bets = [b for b in settled_bets if b.net_payout > 0]
            accuracy = len(winning_bets) / len(settled_bets) if settled_bets else 0
            
            # Calculate returns
            returns = [b.net_payout / b.stake if b.stake > 0 else 0 for b in settled_bets]
            if returns:
                sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(365) if np.std(returns) > 0 else 0
            else:
                sharpe_ratio = 0
                
            # Simple information ratio (vs baseline 50% accuracy)
            information_ratio = (accuracy - 0.5) / 0.1  # Assuming 10% std
            
            # Max drawdown
            cumulative_returns = np.cumsum(returns)
            if len(cumulative_returns) > 0:
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (running_max - cumulative_returns) / np.maximum(running_max, 1)
                max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
            else:
                max_drawdown = 0
                
            return SignalPerformance(
                signal_name=signal_name,
                timestamp=datetime.now(timezone.utc),
                period=period,
                n_predictions=len(bets),
                accuracy=accuracy,
                sharpe_ratio=sharpe_ratio,
                information_ratio=information_ratio,
                max_drawdown=max_drawdown,
                correlation_with_others={},  # Would calculate if multiple signals
                regime=self._detect_regime()
            )
            
    def _detect_regime(self) -> str:
        """Simple regime detection based on time."""
        hour = datetime.now(timezone.utc).hour
        if 9 <= hour <= 16:
            return "high_volume"
        elif 16 <= hour <= 20:
            return "low_volume"
        else:
            return "overnight"


def initialize_signal_registry():
    """Initialize the signal registry with existing signals."""
    logger.info("Initializing signal registry...")
    
    # Create registry
    registry = SignalRegistry()
    
    # Register existing signals
    for signal in SIGNAL_PROVIDERS:
        adapter = SignalProviderAdapter(signal)
        registry.register(adapter)
        logger.info(f"Registered existing signal: {signal.name}")
    
    # Add some enhanced signals
    from signal_registry import EnhancedImpliedSignal
    registry.register(EnhancedImpliedSignal(adjustment_factor=0.98))
    
    # Initialize weight manager
    weight_manager = DynamicWeightManager(registry)
    
    # Calculate initial weights
    weights = weight_manager.calculate_weights(method='equal')
    weight_manager.update_weights(weights, method='equal', notes='Initial setup')
    
    return registry, weight_manager


def update_signal_weights(registry: SignalRegistry, 
                         weight_manager: DynamicWeightManager,
                         method: str = "bayesian"):
    """Update signal weights based on recent performance."""
    logger.info(f"Updating signal weights using {method} method...")
    
    # Calculate new weights
    weights = weight_manager.calculate_weights(method=method)
    
    # Log weight changes
    old_weights = weight_manager.get_current_weights()
    for signal, new_weight in weights.items():
        old_weight = old_weights.get(signal, 0)
        if abs(new_weight - old_weight) > 0.01:
            logger.info(f"Weight change for {signal}: {old_weight:.3f} -> {new_weight:.3f}")
    
    # Update weights
    weight_manager.update_weights(weights, method=method)
    
    return weights


def get_combined_signal(df: pd.DataFrame,
                       registry: SignalRegistry,
                       weight_manager: DynamicWeightManager) -> pd.Series:
    """Get weighted combination of all active signals."""
    aggregator = SignalAggregator(registry, weight_manager)
    return aggregator.get_combined_predictions(df)


def run_performance_update(registry: SignalRegistry,
                          session_id: str,
                          period: str = "daily"):
    """Update performance metrics for all signals."""
    logger.info(f"Updating {period} performance metrics...")
    
    tracker = PerformanceTracker(registry)
    
    for signal_name in registry.list_signals():
        perf = tracker.calculate_signal_performance(signal_name, session_id, period)
        if perf:
            registry.update_performance(signal_name, perf)
            logger.info(f"Updated {signal_name}: accuracy={perf.accuracy:.2%}, "
                       f"sharpe={perf.sharpe_ratio:.2f}")


def demonstrate_dynamic_signals():
    """Demonstrate the dynamic signal system."""
    # Initialize
    registry, weight_manager = initialize_signal_registry()
    
    logger.info("\n=== Signal Registry Status ===")
    logger.info(f"Active signals: {registry.list_signals()}")
    logger.info(f"Current weights: {weight_manager.get_current_weights()}")
    
    # Create sample data
    test_data = pd.DataFrame({
        'implied_raw': [45.0, 55.0, 48.0, 52.0, 60.0, 40.0],
        'source_id': ['m1', 'm1', 'm2', 'm2', 'm3', 'm3'],
        'normalized_outcome': ['home', 'away', 'home', 'away', 'home', 'away'],
        'odds': [2.22, 1.82, 2.08, 1.92, 1.67, 2.5]
    })
    
    # Get predictions from individual signals
    logger.info("\n=== Individual Signal Predictions ===")
    for signal_name in registry.list_signals():
        signal = registry.get_signal(signal_name)
        if signal and signal.validate_input(test_data):
            probs = signal.get_probs(test_data)
            logger.info(f"{signal_name}: {probs.values}")
    
    # Get combined prediction
    combined = get_combined_signal(test_data, registry, weight_manager)
    logger.info(f"\nCombined prediction: {combined.values}")
    
    # Test different weighting methods
    logger.info("\n=== Testing Different Weight Methods ===")
    for method in ['equal', 'inverse_variance', 'bayesian']:
        weights = weight_manager.calculate_weights(method=method)
        logger.info(f"{method}: {weights}")
    
    # Update weights based on "performance"
    update_signal_weights(registry, weight_manager, method='bayesian')
    
    # Get new combined prediction
    new_combined = get_combined_signal(test_data, registry, weight_manager)
    logger.info(f"\nNew combined prediction: {new_combined.values}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_dynamic_signals()