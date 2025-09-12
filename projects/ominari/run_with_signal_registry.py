#!/usr/bin/env python3
"""
Run evaluate_open_markets with the dynamic signal registry.
Demonstrates hot-swappable signals with adaptive weighting.
"""

import logging
from datetime import datetime, timezone

# Patch the evaluate_open_markets module to use the registry
from update_evaluate_with_registry import patch_evaluate_open_markets, update_registry_weights
patch_evaluate_open_markets()

# Now import the patched functions
from evaluate_open_markets import generate_betting_session_report_and_save

logger = logging.getLogger(__name__)


def run_with_dynamic_signals():
    """Run betting evaluation with dynamic signal registry."""
    
    # Update weights based on recent performance
    # This could be scheduled to run periodically
    logger.info("Updating signal weights based on recent performance...")
    new_weights = update_registry_weights(method="bayesian")
    logger.info(f"Updated weights: {new_weights}")
    
    # Generate betting report using the registry
    # Note: We don't pass signal_providers or signal_weights
    # The patched function will use the registry automatically
    result = generate_betting_session_report_and_save(
        execution_bankroll=100,
        kelly_fraction=0.5,  # 50% Kelly
        cap_per_game=0.25,
        cap_per_bet=0.25,
        cap_per_game_market=0.10,
        min_bet_abs=0.5,
        min_bet_pct=0.0025,
        avg_game_duration_minutes=120.0,
        min_break_minutes=60.0,
        abs_game_limit=None,
        signal_providers=None,  # Use registry
        signal_weights=None,    # Use dynamic weights
        mode="paper_trading",
        strat="dynamic_signals",
        display_md=False
    )
    
    return result


def demonstrate_weight_adaptation():
    """Demonstrate different weighting strategies."""
    
    methods = ["equal", "inverse_variance", "bayesian", "online_learning", "regime_based"]
    
    for method in methods:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {method.upper()} weighting")
        logger.info(f"{'='*60}")
        
        # Update weights
        weights = update_registry_weights(method=method)
        logger.info(f"Weights: {weights}")
        
        # Run evaluation
        result = generate_betting_session_report_and_save(
            execution_bankroll=100,
            kelly_fraction=0.25,  # Conservative 25% Kelly for testing
            avg_game_duration_minutes=120.0,
            min_break_minutes=60.0,
            signal_providers=None,  # Use registry
            signal_weights=None,    # Use dynamic weights
            mode="simulation",
            strat=f"dynamic_{method}",
            display_md=False
        )
        
        if result.get("session_id"):
            logger.info(f"Created session {result['session_id']} with {method} weights")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run main evaluation
    logger.info("Running betting evaluation with dynamic signals...")
    run_with_dynamic_signals()
    
    # Demonstrate weight adaptation
    # demonstrate_weight_adaptation()  # Uncomment to test different methods