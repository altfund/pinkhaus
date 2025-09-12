#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to run multiple backtest iterations with different configurations.
"""

import pandas as pd
from datetime import datetime
from vectorized_backtest import run_vectorized_backtest_with_chunks
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS


def run_backtest_experiment(name, description, strategies, **kwargs):
    """Run a backtest experiment and return results."""
    print(f"\n{'='*80}")
    print(f"EXPERIMENT: {name}")
    print(f"Description: {description}")
    print(f"{'='*80}")
    
    # Default parameters
    params = {
        "min_break_minutes": 60.0,
        "avg_game_duration_minutes": 120.0,
        "save_results": True,
    }
    params.update(kwargs)
    
    # Run backtest
    results = run_vectorized_backtest_with_chunks(
        strategies=strategies,
        **params
    )
    
    return {
        "name": name,
        "description": description,
        "results": results,
        "params": params,
        "strategies": [s["name"] for s in strategies]
    }


def main():
    """Run multiple backtest experiments."""
    experiments = []
    
    # Base configuration for all strategies
    base_config = {
        "bankroll": 1000,
        "correlation_matrix": None,
        "risk_adjusted": True,
        "max_stake_per_bet": None,
    }
    
    # Experiment 1: Compare individual signal providers
    print("\n" + "="*80)
    print("STARTING BACKTEST EXPERIMENTS")
    print("="*80)
    
    individual_strategies = []
    for provider in SIGNAL_PROVIDERS:
        strategy = {
            "name": f"only_{provider.name}",
            "providers": [provider],
            "weights": [1.0],
            **base_config
        }
        individual_strategies.append(strategy)
    
    exp1 = run_backtest_experiment(
        name="Individual Signals",
        description="Test each signal provider independently",
        strategies=individual_strategies
    )
    experiments.append(exp1)
    
    # Experiment 2: Different weight combinations
    weight_configs = [
        {
            "name": "equal_weights",
            "weights": {p.name: 1.0 for p in SIGNAL_PROVIDERS},
            "description": "Equal weights for all signals"
        },
        {
            "name": "implied_dominant",
            "weights": {
                "implied_probability": 0.8,
                "coin_flip": 0.1,
                "grant": 0.1,
            },
            "description": "80% implied, 10% each for models"
        },
        {
            "name": "models_dominant",
            "weights": {
                "implied_probability": 0.2,
                "coin_flip": 0.4,
                "grant": 0.4,
            },
            "description": "20% implied, 40% each for models"
        },
        {
            "name": "grant_focus",
            "weights": {
                "implied_probability": 0.3,
                "coin_flip": 0.2,
                "grant": 0.5,
            },
            "description": "50% grant, 30% implied, 20% coin_flip"
        },
    ]
    
    weight_strategies = []
    for config in weight_configs:
        strategy = {
            "name": config["name"],
            "providers": SIGNAL_PROVIDERS,
            "weights": config["weights"],
            **base_config
        }
        weight_strategies.append(strategy)
    
    exp2 = run_backtest_experiment(
        name="Weight Variations",
        description="Test different weight combinations",
        strategies=weight_strategies
    )
    experiments.append(exp2)
    
    # Experiment 3: Risk parameter variations
    risk_strategies = []
    kelly_fractions = [0.1, 0.25, 0.5]
    
    for kf in kelly_fractions:
        strategy = {
            "name": f"kelly_{int(kf*100)}pct",
            "providers": SIGNAL_PROVIDERS,
            "weights": SIGNAL_WEIGHTS,
            "kelly_fraction": kf,
            **base_config
        }
        risk_strategies.append(strategy)
    
    exp3 = run_backtest_experiment(
        name="Kelly Fraction Variations",
        description="Test different Kelly fraction values",
        strategies=risk_strategies,
        min_break_minutes=120.0  # Longer breaks for this test
    )
    experiments.append(exp3)
    
    # Experiment 4: Time window variations
    time_strategies = [{
        "name": "standard_timing",
        "providers": SIGNAL_PROVIDERS,
        "weights": SIGNAL_WEIGHTS,
        **base_config
    }]
    
    # Test with different break/game durations
    for min_break, avg_game in [(30.0, 120.0), (60.0, 90.0), (120.0, 150.0)]:
        exp = run_backtest_experiment(
            name=f"Timing_{int(min_break)}min_break_{int(avg_game)}min_game",
            description=f"Break: {min_break}min, Game: {avg_game}min",
            strategies=time_strategies,
            min_break_minutes=min_break,
            avg_game_duration_minutes=avg_game
        )
        experiments.append(exp)
    
    # Generate summary report
    print("\n" + "="*80)
    print("BACKTEST EXPERIMENTS SUMMARY")
    print("="*80)
    
    all_summaries = []
    
    for exp in experiments:
        print(f"\n{exp['name']}:")
        print(f"  Description: {exp['description']}")
        
        if exp['results'] is not None and not exp['results'].empty:
            results = exp['results']
            
            # Calculate metrics for each strategy
            for strat_name in exp['strategies']:
                strat_results = results[results['strategy_name'] == strat_name]
                
                if not strat_results.empty:
                    metrics = {
                        'experiment': exp['name'],
                        'strategy': strat_name,
                        'total_bets': len(strat_results),
                        'total_staked': strat_results['total_staked'].sum(),
                        'total_net': strat_results['net'].sum(),
                        'avg_roi': strat_results['roi'].mean(),
                        'std_roi': strat_results['roi'].std(),
                        'win_rate': (strat_results['net'] > 0).mean(),
                        'sharpe': strat_results['roi'].mean() / strat_results['roi'].std() if strat_results['roi'].std() > 0 else 0
                    }
                    all_summaries.append(metrics)
                    
                    print(f"\n  {strat_name}:")
                    print(f"    Bets: {metrics['total_bets']}")
                    print(f"    Total Staked: ${metrics['total_staked']:.2f}")
                    print(f"    Net Return: ${metrics['total_net']:.2f}")
                    print(f"    ROI: {metrics['avg_roi']:.2%} ± {metrics['std_roi']:.2%}")
                    print(f"    Win Rate: {metrics['win_rate']:.2%}")
                    print(f"    Sharpe: {metrics['sharpe']:.3f}")
        else:
            print("  No results generated")
    
    # Save summary to CSV
    if all_summaries:
        summary_df = pd.DataFrame(all_summaries)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = f"backtests/experiment_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n\nSummary saved to: {summary_file}")
        
        # Find best performing strategies
        print("\n" + "="*80)
        print("TOP PERFORMING STRATEGIES")
        print("="*80)
        
        # Sort by total return
        summary_df['total_return'] = summary_df['total_net'] / summary_df['total_staked']
        top_by_return = summary_df.nlargest(5, 'total_return')
        
        print("\nTop 5 by Total Return:")
        for _, row in top_by_return.iterrows():
            print(f"  {row['experiment']} - {row['strategy']}: {row['total_return']:.2%}")
        
        # Sort by Sharpe ratio
        top_by_sharpe = summary_df.nlargest(5, 'sharpe')
        
        print("\nTop 5 by Sharpe Ratio:")
        for _, row in top_by_sharpe.iterrows():
            print(f"  {row['experiment']} - {row['strategy']}: {row['sharpe']:.3f}")
        
        # Sort by win rate
        top_by_winrate = summary_df.nlargest(5, 'win_rate')
        
        print("\nTop 5 by Win Rate:")
        for _, row in top_by_winrate.iterrows():
            print(f"  {row['experiment']} - {row['strategy']}: {row['win_rate']:.2%}")


if __name__ == "__main__":
    main()