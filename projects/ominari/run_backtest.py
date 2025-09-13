#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main backtest runner using vectorized implementation.
This replaces the original backtest.py main function.
"""

from vectorized_backtest import run_vectorized_backtest_with_chunks
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS


def main():
    """Run vectorized backtest with configurable parameters."""
    # Backtest configuration
    min_break_minutes = 1 * 60.0  # 1 hour break between sessions
    avg_game_duration_minutes = 120.0  # 2 hour average game duration
    
    # Strategy configuration
    kelly_fraction = 0.25  # Quarter Kelly
    execution_bankroll = 1000.0
    
    # Risk limits
    cap_per_game = 0.25
    cap_per_bet = 0.25
    cap_per_game_market = 0.10
    min_bet_abs = 0.5
    min_bet_pct = 0.0025
    
    # Create strategies
    # 1. Combined strategy using all signals
    combined_strategy = {
        "name": "combined_signals",
        "providers": SIGNAL_PROVIDERS,
        "weights": SIGNAL_WEIGHTS,
        "bankroll": execution_bankroll,
        "kelly_fraction": kelly_fraction,
        "cap_per_game": cap_per_game,
        "cap_per_bet": cap_per_bet,
        "cap_per_game_market": cap_per_game_market,
        "min_bet_abs": min_bet_abs,
        "min_bet_pct": min_bet_pct,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None,
    }
    
    # 2. Individual strategies for each signal provider
    individual_strategies = []
    for provider in SIGNAL_PROVIDERS:
        strategy = {
            "name": f"single_{provider.name}",
            "providers": [provider],
            "weights": [1.0],
            "bankroll": execution_bankroll,
            "kelly_fraction": kelly_fraction,
            "cap_per_game": cap_per_game,
            "cap_per_bet": cap_per_bet,
            "cap_per_game_market": cap_per_game_market,
            "min_bet_abs": min_bet_abs,
            "min_bet_pct": min_bet_pct,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None,
        }
        individual_strategies.append(strategy)
    
    # 3. Weighted combination strategies (different weight configurations)
    weighted_strategies = [
        {
            "name": "implied_heavy",
            "providers": SIGNAL_PROVIDERS,
            "weights": {
                "implied_probability": 0.7,
                "coin_flip": 0.2,
                "grant": 0.1,
            },
            "bankroll": execution_bankroll,
            "kelly_fraction": kelly_fraction,
            "cap_per_game": cap_per_game,
            "cap_per_bet": cap_per_bet,
            "cap_per_game_market": cap_per_game_market,
            "min_bet_abs": min_bet_abs,
            "min_bet_pct": min_bet_pct,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None,
        },
        {
            "name": "model_heavy",
            "providers": SIGNAL_PROVIDERS,
            "weights": {
                "implied_probability": 0.3,
                "coin_flip": 0.35,
                "grant": 0.35,
            },
            "bankroll": execution_bankroll,
            "kelly_fraction": kelly_fraction,
            "cap_per_game": cap_per_game,
            "cap_per_bet": cap_per_bet,
            "cap_per_game_market": cap_per_game_market,
            "min_bet_abs": min_bet_abs,
            "min_bet_pct": min_bet_pct,
            "correlation_matrix": None,
            "risk_adjusted": True,
            "max_stake_per_bet": None,
        },
    ]
    
    # Combine all strategies
    all_strategies = [combined_strategy] + individual_strategies + weighted_strategies
    
    print(f"Running backtest with {len(all_strategies)} strategies:")
    for s in all_strategies:
        print(f"  - {s['name']}")
    print()
    
    # Run the vectorized backtest
    results = run_vectorized_backtest_with_chunks(
        strategies=all_strategies,
        min_break_minutes=min_break_minutes,
        avg_game_duration_minutes=avg_game_duration_minutes,
        save_results=True,
    )
    
    if results is not None:
        print("\n=== BACKTEST COMPLETE ===")
        print(f"Total results: {len(results)} betting decisions")
        
        # Get performance summary for each strategy
        print("\n=== DETAILED PERFORMANCE BY STRATEGY ===")
        for strategy in all_strategies:
            strat_name = strategy["name"]
            strat_results = results[results["strategy_name"] == strat_name]
            
            if not strat_results.empty:
                total_staked = strat_results["total_staked"].sum()
                total_net = strat_results["net"].sum()
                avg_roi = strat_results["roi"].mean()
                # Win rate based on actual wins (result_multiplier > 0 means the bet won)
                win_rate = (strat_results["result_multiplier"] > 0).mean()
                
                print(f"\n{strat_name}:")
                print(f"  Total Staked: ${total_staked:.2f}")
                
                # Include fee information if available
                if "fee_amount" in strat_results.columns:
                    total_fees = strat_results["fee_amount"].sum()
                    print(f"  Total Fees: ${total_fees:.2f}")
                if "execution_stake" in strat_results.columns:
                    total_execution = strat_results["execution_stake"].sum()
                    print(f"  Total Execution Stake: ${total_execution:.2f}")
                    
                print(f"  Net Return: ${total_net:.2f}")
                print(f"  Average ROI: {avg_roi:.2%}")
                print(f"  Win Rate: {win_rate:.2%}")
                print(f"  Number of Bets: {len(strat_results)}")


if __name__ == "__main__":
    main()