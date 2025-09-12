#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Signal Registry and Weight Management
Provides hot-pluggable signals with adaptive weighting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
import logging
import json
from abc import ABC, abstractmethod
import sqlite3

logger = logging.getLogger(__name__)


@dataclass
class SignalMetadata:
    """Metadata for a registered signal."""
    name: str
    version: str
    description: str
    author: str
    created_at: datetime
    updated_at: datetime
    parameters: Dict[str, Any]
    performance_stats: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class SignalPerformance:
    """Track signal performance over time."""
    signal_name: str
    timestamp: datetime
    period: str  # 'hourly', 'daily', 'weekly'
    n_predictions: int
    accuracy: float
    sharpe_ratio: float
    information_ratio: float
    max_drawdown: float
    correlation_with_others: Dict[str, float]
    regime: str  # Market regime identifier


class BaseSignalProvider(ABC):
    """Enhanced base class for signal providers."""
    
    def __init__(self, name: str, version: str = "1.0.0"):
        self.name = name
        self.version = version
        self.metadata = SignalMetadata(
            name=name,
            version=version,
            description=self.__class__.__doc__ or "",
            author=self.__class__.__module__,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            parameters=self.get_parameters()
        )
        
    @abstractmethod
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """Generate probability predictions."""
        pass
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Return current parameters."""
        pass
        
    def update_parameters(self, params: Dict[str, Any]):
        """Update signal parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.metadata.parameters = self.get_parameters()
        self.metadata.updated_at = datetime.now(timezone.utc)
        
    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate input data."""
        required_columns = self.get_required_columns()
        return all(col in df.columns for col in required_columns)
        
    @abstractmethod
    def get_required_columns(self) -> List[str]:
        """Return list of required DataFrame columns."""
        pass


class SignalRegistry:
    """Central registry for all signal providers."""
    
    def __init__(self, db_path: str = "signal_registry.db"):
        self.db_path = db_path
        self.signals: Dict[str, BaseSignalProvider] = {}
        self.performance_history: List[SignalPerformance] = []
        self._init_database()
        self._load_registered_signals()
        
    def _init_database(self):
        """Initialize registry database."""
        conn = sqlite3.connect(self.db_path)
        
        # Signal metadata table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_metadata (
                name TEXT PRIMARY KEY,
                version TEXT,
                description TEXT,
                author TEXT,
                created_at DATETIME,
                updated_at DATETIME,
                parameters TEXT,
                is_active BOOLEAN,
                tags TEXT
            )
        """)
        
        # Performance history table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_name TEXT,
                timestamp DATETIME,
                period TEXT,
                n_predictions INTEGER,
                accuracy REAL,
                sharpe_ratio REAL,
                information_ratio REAL,
                max_drawdown REAL,
                correlations TEXT,
                regime TEXT,
                FOREIGN KEY (signal_name) REFERENCES signal_metadata(name)
            )
        """)
        
        # Signal weights table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_weights (
                timestamp DATETIME PRIMARY KEY,
                weights TEXT,
                method TEXT,
                performance_window INTEGER,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
    def register(self, signal: BaseSignalProvider, persist: bool = True):
        """Register a new signal provider."""
        logger.info(f"Registering signal: {signal.name} v{signal.version}")
        
        # Validate signal
        if not isinstance(signal, BaseSignalProvider):
            raise TypeError("Signal must inherit from BaseSignalProvider")
            
        # Check for conflicts
        if signal.name in self.signals:
            logger.warning(f"Replacing existing signal: {signal.name}")
            
        self.signals[signal.name] = signal
        
        if persist:
            self._persist_signal_metadata(signal.metadata)
            
    def unregister(self, signal_name: str):
        """Remove a signal from the registry."""
        if signal_name in self.signals:
            del self.signals[signal_name]
            logger.info(f"Unregistered signal: {signal_name}")
            
    def get_signal(self, name: str) -> Optional[BaseSignalProvider]:
        """Get a registered signal by name."""
        return self.signals.get(name)
        
    def list_signals(self, active_only: bool = True) -> List[str]:
        """List all registered signals."""
        if active_only:
            return [
                name for name, signal in self.signals.items()
                if signal.metadata.is_active
            ]
        return list(self.signals.keys())
        
    def update_performance(self, signal_name: str, performance: SignalPerformance):
        """Update performance metrics for a signal."""
        if signal_name not in self.signals:
            logger.warning(f"Unknown signal: {signal_name}")
            return
            
        self.performance_history.append(performance)
        self._persist_performance(performance)
        
        # Update signal metadata
        signal = self.signals[signal_name]
        signal.metadata.performance_stats = {
            'accuracy': performance.accuracy,
            'sharpe_ratio': performance.sharpe_ratio,
            'information_ratio': performance.information_ratio
        }
        
    def _persist_signal_metadata(self, metadata: SignalMetadata):
        """Save signal metadata to database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT OR REPLACE INTO signal_metadata (
                name, version, description, author,
                created_at, updated_at, parameters,
                is_active, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata.name,
            metadata.version,
            metadata.description,
            metadata.author,
            metadata.created_at,
            metadata.updated_at,
            json.dumps(metadata.parameters),
            metadata.is_active,
            json.dumps(metadata.tags)
        ))
        
        conn.commit()
        conn.close()
        
    def _persist_performance(self, performance: SignalPerformance):
        """Save performance data to database."""
        conn = sqlite3.connect(self.db_path)
        
        conn.execute("""
            INSERT INTO signal_performance (
                signal_name, timestamp, period, n_predictions,
                accuracy, sharpe_ratio, information_ratio,
                max_drawdown, correlations, regime
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            performance.signal_name,
            performance.timestamp,
            performance.period,
            performance.n_predictions,
            performance.accuracy,
            performance.sharpe_ratio,
            performance.information_ratio,
            performance.max_drawdown,
            json.dumps(performance.correlation_with_others),
            performance.regime
        ))
        
        conn.commit()
        conn.close()
        
    def _load_registered_signals(self):
        """Load previously registered signals from database."""
        # This would load signal metadata and attempt to 
        # reconstruct signals from saved state
        pass


class DynamicWeightManager:
    """Manages dynamic weight allocation for signals."""
    
    def __init__(self, registry: SignalRegistry, db_path: str = "signal_weights.db"):
        self.registry = registry
        self.db_path = db_path
        self.current_weights: Dict[str, float] = {}
        self.weight_history: List[Tuple[datetime, Dict[str, float]]] = []
        
    def calculate_weights(self, 
                         method: str = 'bayesian',
                         lookback_days: int = 30) -> Dict[str, float]:
        """Calculate optimal weights for all active signals."""
        
        if method == 'equal':
            return self._equal_weights()
        elif method == 'inverse_variance':
            return self._inverse_variance_weights(lookback_days)
        elif method == 'bayesian':
            return self._bayesian_weights(lookback_days)
        elif method == 'online_learning':
            return self._online_learning_weights()
        elif method == 'regime_based':
            return self._regime_based_weights(lookback_days)
        else:
            raise ValueError(f"Unknown weighting method: {method}")
            
    def _equal_weights(self) -> Dict[str, float]:
        """Simple equal weighting."""
        active_signals = self.registry.list_signals(active_only=True)
        n_signals = len(active_signals)
        
        if n_signals == 0:
            return {}
            
        weight = 1.0 / n_signals
        return {signal: weight for signal in active_signals}
        
    def _inverse_variance_weights(self, lookback_days: int) -> Dict[str, float]:
        """Inverse variance weighting based on recent performance."""
        # Get performance data
        perf_data = self._get_performance_data(lookback_days)
        
        if perf_data.empty:
            return self._equal_weights()
            
        # Calculate variances
        variances = {}
        for signal in self.registry.list_signals(active_only=True):
            signal_perf = perf_data[perf_data['signal_name'] == signal]
            if not signal_perf.empty:
                # Use accuracy variance as proxy
                variances[signal] = signal_perf['accuracy'].var()
            else:
                variances[signal] = 1.0  # Default high variance
                
        # Calculate inverse variance weights
        inv_vars = {s: 1/v for s, v in variances.items() if v > 0}
        total_inv_var = sum(inv_vars.values())
        
        if total_inv_var > 0:
            weights = {s: iv/total_inv_var for s, iv in inv_vars.items()}
        else:
            weights = self._equal_weights()
            
        return weights
        
    def _bayesian_weights(self, lookback_days: int) -> Dict[str, float]:
        """Bayesian weight updating based on performance."""
        # Start with prior (equal weights)
        prior_weights = self._equal_weights()
        
        # Get recent performance
        perf_data = self._get_performance_data(lookback_days)
        
        if perf_data.empty:
            return prior_weights
            
        # Update weights based on Bayesian inference
        weights = {}
        for signal, prior in prior_weights.items():
            signal_perf = perf_data[perf_data['signal_name'] == signal]
            
            if not signal_perf.empty:
                # Use accuracy as likelihood
                avg_accuracy = signal_perf['accuracy'].mean()
                
                # Simple Bayesian update
                # posterior ∝ prior × likelihood
                posterior = prior * avg_accuracy
                weights[signal] = posterior
            else:
                weights[signal] = prior
                
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w/total_weight for s, w in weights.items()}
            
        return weights
        
    def _online_learning_weights(self) -> Dict[str, float]:
        """Online learning with exponential weighted average."""
        # Initialize with equal weights if no history
        if not self.weight_history:
            return self._equal_weights()
            
        # Get latest weights
        _, latest_weights = self.weight_history[-1]
        
        # Learning rate
        eta = 0.01
        
        # Get recent performance (1 day)
        perf_data = self._get_performance_data(1)
        
        # Update weights using gradient
        new_weights = {}
        for signal in self.registry.list_signals(active_only=True):
            old_weight = latest_weights.get(signal, 0.1)
            
            # Calculate gradient based on recent performance
            signal_perf = perf_data[perf_data['signal_name'] == signal]
            if not signal_perf.empty:
                # Gradient proportional to accuracy
                gradient = signal_perf['accuracy'].mean() - 0.5
                new_weight = old_weight * np.exp(eta * gradient)
            else:
                new_weight = old_weight
                
            new_weights[signal] = new_weight
            
        # Normalize
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {s: w/total_weight for s, w in new_weights.items()}
            
        return new_weights
        
    def _regime_based_weights(self, lookback_days: int) -> Dict[str, float]:
        """Adjust weights based on detected market regime."""
        # Detect current regime
        regime = self._detect_regime()
        
        # Get regime-specific performance
        perf_data = self._get_performance_data(lookback_days)
        regime_perf = perf_data[perf_data['regime'] == regime]
        
        if regime_perf.empty:
            # Fall back to overall performance
            regime_perf = perf_data
            
        # Calculate weights based on regime performance
        weights = {}
        for signal in self.registry.list_signals(active_only=True):
            signal_perf = regime_perf[regime_perf['signal_name'] == signal]
            
            if not signal_perf.empty:
                # Weight based on Sharpe ratio in this regime
                avg_sharpe = signal_perf['sharpe_ratio'].mean()
                weights[signal] = max(0, avg_sharpe)  # Only positive Sharpe
            else:
                weights[signal] = 0.0
                
        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {s: w/total_weight for s, w in weights.items()}
        else:
            # If no positive Sharpe, use equal weights
            weights = self._equal_weights()
            
        return weights
        
    def _detect_regime(self) -> str:
        """Detect current market regime."""
        # Simplified regime detection
        # In practice, this would use market indicators
        
        # Mock implementation
        hour = datetime.now(timezone.utc).hour
        
        if 9 <= hour <= 16:  # Market hours
            return "high_volume"
        elif 16 <= hour <= 20:  # After hours
            return "low_volume"
        else:  # Night
            return "overnight"
            
    def _get_performance_data(self, lookback_days: int) -> pd.DataFrame:
        """Get performance data from database."""
        conn = sqlite3.connect(self.registry.db_path)
        
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        
        df = pd.read_sql_query("""
            SELECT * FROM signal_performance
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, conn, params=(cutoff_date,))
        
        conn.close()
        return df
        
    def update_weights(self, weights: Dict[str, float], method: str, notes: str = ""):
        """Update and persist current weights."""
        self.current_weights = weights
        self.weight_history.append((datetime.now(timezone.utc), weights.copy()))
        
        # Persist to database
        conn = sqlite3.connect(self.registry.db_path)
        
        conn.execute("""
            INSERT INTO signal_weights (
                timestamp, weights, method, performance_window, notes
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc),
            json.dumps(weights),
            method,
            30,  # Default lookback
            notes
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Updated weights using {method}: {weights}")
        
    def get_current_weights(self) -> Dict[str, float]:
        """Get current active weights."""
        if not self.current_weights:
            # Load from database or use equal weights
            self.current_weights = self._equal_weights()
            
        return self.current_weights


class SignalAggregator:
    """Aggregates predictions from multiple signals."""
    
    def __init__(self, registry: SignalRegistry, weight_manager: DynamicWeightManager):
        self.registry = registry
        self.weight_manager = weight_manager
        
    def get_combined_predictions(self, 
                               df: pd.DataFrame,
                               signal_names: Optional[List[str]] = None) -> pd.Series:
        """Get weighted combination of signal predictions."""
        
        if signal_names is None:
            signal_names = self.registry.list_signals(active_only=True)
            
        weights = self.weight_manager.get_current_weights()
        
        # Collect predictions
        predictions = {}
        for signal_name in signal_names:
            signal = self.registry.get_signal(signal_name)
            
            if signal and signal_name in weights:
                try:
                    # Validate input
                    if signal.validate_input(df):
                        preds = signal.get_probs(df)
                        predictions[signal_name] = preds
                    else:
                        logger.warning(f"Invalid input for signal: {signal_name}")
                except Exception as e:
                    logger.error(f"Error in signal {signal_name}: {e}")
                    
        if not predictions:
            logger.warning("No valid predictions available")
            return pd.Series(0.5, index=df.index)  # Default to 50%
            
        # Weighted combination
        combined = pd.Series(0.0, index=df.index)
        total_weight = 0.0
        
        for signal_name, preds in predictions.items():
            weight = weights.get(signal_name, 0.0)
            combined += weight * preds
            total_weight += weight
            
        if total_weight > 0:
            combined /= total_weight
            
        return combined.clip(0.0, 1.0)


# Example signal implementations
class EnhancedImpliedSignal(BaseSignalProvider):
    """Enhanced implied probability signal with adjustments."""
    
    def __init__(self, adjustment_factor: float = 1.0):
        self.adjustment_factor = adjustment_factor
        super().__init__("enhanced_implied", "2.0.0")
        
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """Calculate adjusted implied probabilities."""
        raw_implied = df["implied_raw"].astype(float) / 100.0
        
        # Apply adjustment based on market conditions
        adjusted = raw_implied * self.adjustment_factor
        
        # Normalize within matches
        if 'match_id' in df.columns:
            normalized = df.groupby('match_id')[adjusted].transform(
                lambda x: x / x.sum() if x.sum() > 0 else x
            )
            return normalized
        else:
            return adjusted
            
    def get_parameters(self) -> Dict[str, Any]:
        return {'adjustment_factor': self.adjustment_factor}
        
    def get_required_columns(self) -> List[str]:
        return ['implied_raw']


def main():
    """Example usage."""
    # Initialize registry
    registry = SignalRegistry()
    
    # Register signals
    from signals import ImpliedRawSignal
    
    registry.register(ImpliedRawSignal())
    registry.register(EnhancedImpliedSignal(adjustment_factor=0.95))
    
    # Initialize weight manager
    weight_mgr = DynamicWeightManager(registry)
    
    # Calculate weights
    weights = weight_mgr.calculate_weights(method='bayesian')
    print(f"Calculated weights: {weights}")
    
    # Update weights
    weight_mgr.update_weights(weights, method='bayesian')
    
    # Create aggregator
    aggregator = SignalAggregator(registry, weight_mgr)
    
    # Example data
    test_data = pd.DataFrame({
        'implied_raw': [45.0, 55.0, 48.0, 52.0],
        'match_id': ['m1', 'm1', 'm2', 'm2']
    })
    
    # Get combined predictions
    combined = aggregator.get_combined_predictions(test_data)
    print(f"\nCombined predictions: {combined.values}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()