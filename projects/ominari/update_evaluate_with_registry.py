#!/usr/bin/env python3
"""
Updates evaluate_open_markets.py to use the dynamic signal registry.
This allows hot-swappable signals with adaptive weighting.
"""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional

from signal_registry import SignalRegistry, DynamicWeightManager, SignalAggregator
from integrate_signal_registry import (
    initialize_signal_registry, 
    update_signal_weights,
    get_combined_signal,
    SignalProviderAdapter
)
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS

logger = logging.getLogger(__name__)

# Global registry instances
_registry = None
_weight_manager = None


def get_or_create_registry():
    """Get or create the global signal registry."""
    global _registry, _weight_manager
    
    if _registry is None:
        _registry, _weight_manager = initialize_signal_registry()
        logger.info("Initialized signal registry")
    
    return _registry, _weight_manager


def aggregate_signals_with_registry(
    df,
    providers: Optional[List] = None,  # Can be None to use registry
    weights: Optional[Dict[str, float]] = None  # Can be None to use dynamic weights
):
    """
    Enhanced version of aggregate_signals that can use the signal registry.
    
    If providers/weights are provided, uses them directly (backward compatible).
    If not provided, uses the dynamic signal registry.
    """
    df = df.copy()
    
    # Get registry if not using explicit providers
    if providers is None:
        registry, weight_manager = get_or_create_registry()
        
        # Use registry to get predictions
        aggregator = SignalAggregator(registry, weight_manager)
        df["probability"] = aggregator.get_combined_predictions(df)
        
        # Add individual signal contributions for debugging
        current_weights = weight_manager.get_current_weights()
        for signal_name in registry.list_signals():
            signal = registry.get_signal(signal_name)
            if signal and signal.validate_input(df):
                try:
                    probs = signal.get_probs(df)
                    df[signal_name] = probs
                    weight = current_weights.get(signal_name, 0)
                    df[f"contrib_{signal_name}"] = probs * weight
                except Exception as e:
                    logger.error(f"Error in signal {signal_name}: {e}")
        
        return df
    
    else:
        # Original implementation for backward compatibility
        import numpy as np
        import pandas as pd
        
        # collect per-provider prob Series in a dict
        prob_dict = {}
        for provider in providers:
            name = provider.name
            w = weights.get(name, 0.0)
            raw = provider.get_probs(df)
            # map error sentinel to NaN
            probs = raw.replace(-1.0, np.nan).astype(float)
            prob_dict[name] = probs
            df[name] = probs
        
        # build a DataFrame of all probs
        probs_df = pd.DataFrame(prob_dict)
        
        # create a weights vector aligned to columns
        weight_vector = pd.Series({p.name: weights.get(p.name, 0.0) for p in providers})
        
        # numerator: sum across providers of (weight * prob)
        numer = (probs_df * weight_vector).sum(axis=1)
        
        # denominator: sum of weights for providers that have a prob (i.e. not-NaN)
        has_signal = probs_df.notna().astype(float)  # 1.0 where present, 0.0 where NaN
        denom = (has_signal * weight_vector).sum(axis=1).replace(0, np.nan)
        
        # final blended probability; rows with zero available weight become NaN
        df["probability"] = numer.div(denom)
        
        # store per-provider actual contribution (w_i * p_i / denom)
        for name in prob_dict:
            df[f"contrib_{name}"] = (probs_df[name] * weight_vector[name]).div(denom)
        
        return df


def update_registry_weights(method: str = "bayesian"):
    """Update the signal registry weights based on recent performance."""
    registry, weight_manager = get_or_create_registry()
    
    logger.info(f"Updating weights using {method} method...")
    old_weights = weight_manager.get_current_weights()
    
    # Update weights
    new_weights = update_signal_weights(registry, weight_manager, method)
    
    # Log significant changes
    for signal, new_weight in new_weights.items():
        old_weight = old_weights.get(signal, 0)
        change = new_weight - old_weight
        if abs(change) > 0.05:  # 5% change threshold
            logger.info(f"Weight change for {signal}: {old_weight:.2%} -> {new_weight:.2%} ({change:+.2%})")
    
    return new_weights


def get_registry_signal_list():
    """Get list of signals from registry for display/reporting."""
    registry, _ = get_or_create_registry()
    
    signal_info = []
    for signal_name in registry.list_signals():
        signal = registry.get_signal(signal_name)
        if signal:
            info = {
                'name': signal_name,
                'version': signal.metadata.version,
                'active': signal.metadata.is_active,
                'performance': signal.metadata.performance_stats
            }
            signal_info.append(info)
    
    return signal_info


# Monkey patch the original aggregate_signals function
# This allows existing code to use the new registry without changes
def patch_evaluate_open_markets():
    """Patches evaluate_open_markets.py to use the signal registry."""
    import evaluate_open_markets
    
    # Replace the aggregate_signals function
    evaluate_open_markets.aggregate_signals = aggregate_signals_with_registry
    
    logger.info("Patched evaluate_open_markets to use signal registry")


def demonstrate_registry_integration():
    """Demonstrate the registry integration."""
    import pandas as pd
    
    # Initialize registry
    registry, weight_manager = get_or_create_registry()
    
    logger.info("=== Signal Registry Status ===")
    logger.info(f"Active signals: {registry.list_signals()}")
    logger.info(f"Current weights: {weight_manager.get_current_weights()}")
    
    # Create test data
    test_data = pd.DataFrame({
        'implied_raw': [45.0, 55.0, 48.0, 52.0],
        'source_id': ['m1', 'm1', 'm2', 'm2'],
        'normalized_outcome': ['home', 'away', 'home', 'away'],
        'odds': [2.22, 1.82, 2.08, 1.92]
    })
    
    # Test with registry (no providers/weights specified)
    result_registry = aggregate_signals_with_registry(test_data)
    logger.info(f"\nRegistry predictions: {result_registry['probability'].values}")
    
    # Test with explicit providers (backward compatibility)
    result_explicit = aggregate_signals_with_registry(
        test_data, 
        providers=SIGNAL_PROVIDERS,
        weights=SIGNAL_WEIGHTS
    )
    logger.info(f"Explicit predictions: {result_explicit['probability'].values}")
    
    # Update weights
    new_weights = update_registry_weights("bayesian")
    logger.info(f"\nUpdated weights: {new_weights}")
    
    # Show signal info
    signal_list = get_registry_signal_list()
    logger.info("\n=== Registered Signals ===")
    for sig in signal_list:
        logger.info(f"{sig['name']} v{sig['version']} - Active: {sig['active']}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    demonstrate_registry_integration()