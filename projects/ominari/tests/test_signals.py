#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for signal system
"""

import pandas as pd
from datetime import datetime, timezone

from signals import ImpliedRawSignal
from signal_registry import SignalRegistry, DynamicWeightManager, SignalAggregator


class TestSignals:
    """Test signal providers."""
    
    def test_implied_raw_signal(self):
        """Test implied raw signal calculation."""
        signal = ImpliedRawSignal()
        
        # Create test data
        data = pd.DataFrame({
            'implied_raw': [45.0, 55.0, 60.0, 40.0],
            'match_id': ['m1', 'm1', 'm2', 'm2']
        })
        
        probs = signal.get_probs(data)
        
        # Check output
        assert len(probs) == 4
        assert all(0 <= p <= 1 for p in probs)
        
        # Check normalization within matches
        m1_probs = probs[data['match_id'] == 'm1']
        assert abs(m1_probs.sum() - 1.0) < 0.01
        
    def test_signal_registry(self):
        """Test signal registration and management."""
        registry = SignalRegistry()
        
        # Register signal
        signal = ImpliedRawSignal()
        registry.register(signal)
        
        # Check registration
        assert signal.name in registry.list_signals()
        assert registry.get_signal(signal.name) is not None
        
        # Unregister
        registry.unregister(signal.name)
        assert signal.name not in registry.list_signals()
        
    def test_weight_manager(self):
        """Test dynamic weight calculation."""
        registry = SignalRegistry()
        manager = DynamicWeightManager(registry)
        
        # Register some signals
        for i in range(3):
            signal = ImpliedRawSignal()
            signal.name = f"signal_{i}"
            registry.register(signal)
            
        # Calculate equal weights
        weights = manager.calculate_weights(method='equal')
        assert len(weights) == 3
        assert all(abs(w - 1/3) < 0.01 for w in weights.values())
        
        # Update weights
        manager.update_weights(weights, method='equal')
        current = manager.get_current_weights()
        assert current == weights
        
    def test_signal_aggregator(self):
        """Test signal aggregation."""
        registry = SignalRegistry()
        manager = DynamicWeightManager(registry)
        aggregator = SignalAggregator(registry, manager)
        
        # Register signal
        signal = ImpliedRawSignal()
        registry.register(signal)
        
        # Set weights
        weights = {signal.name: 1.0}
        manager.update_weights(weights, method='manual')
        
        # Create test data
        data = pd.DataFrame({
            'implied_raw': [45.0, 55.0],
            'match_id': ['m1', 'm1']
        })
        
        # Get predictions
        combined = aggregator.get_combined_predictions(data)
        
        assert len(combined) == 2
        assert all(0 <= p <= 1 for p in combined)


class TestSignalPerformance:
    """Test signal performance tracking."""
    
    def test_performance_metrics(self):
        """Test performance metric calculation."""
        from signal_registry import SignalPerformance
        
        perf = SignalPerformance(
            signal_name="test_signal",
            timestamp=datetime.now(timezone.utc),
            period="daily",
            n_predictions=100,
            accuracy=0.55,
            sharpe_ratio=1.2,
            information_ratio=0.8,
            max_drawdown=-0.10,
            correlation_with_others={"other": 0.3},
            regime="normal"
        )
        
        assert perf.signal_name == "test_signal"
        assert perf.accuracy == 0.55
        assert perf.sharpe_ratio == 1.2