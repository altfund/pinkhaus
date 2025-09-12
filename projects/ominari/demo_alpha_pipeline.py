#!/usr/bin/env python3
"""
Demo of the alpha research pipeline with synthetic data.
"""

import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from alpha_research_pipeline import AlphaResearchPipeline, AlphaSignal

logger = logging.getLogger(__name__)


def generate_synthetic_market_data(n_days=180, n_matches_per_day=50):
    """Generate synthetic market data for testing."""
    logger.info(f"Generating synthetic data for {n_days} days...")
    
    data = []
    base_date = datetime.now(timezone.utc) - timedelta(days=n_days)
    
    for day in range(n_days):
        date = base_date + timedelta(days=day)
        
        for match in range(n_matches_per_day):
            # Generate realistic odds movement
            true_prob = np.random.beta(2, 2)  # True probability
            
            # Add noise and time decay
            for hour in range(24):
                timestamp = date + timedelta(hours=hour)
                
                # Market gets more accurate closer to event
                noise_factor = (24 - hour) / 24 * 0.1
                observed_prob = true_prob + np.random.normal(0, noise_factor)
                observed_prob = np.clip(observed_prob, 0.01, 0.99)
                
                # Convert to decimal odds
                decimal_odds = 1 / observed_prob
                
                data.append({
                    'timestamp': timestamp,
                    'match_id': f"{date.strftime('%Y%m%d')}_{match:03d}",
                    'true_prob': true_prob,
                    'observed_prob': observed_prob,
                    'decimal_odds_best': decimal_odds * (1 - np.random.uniform(0, 0.05)),
                    'decimal_odds_avg': decimal_odds,
                    'decimal_odds_worst': decimal_odds * (1 + np.random.uniform(0, 0.05)),
                    'volume': np.random.lognormal(6, 2),
                    'n_bookmakers': np.random.randint(3, 15),
                    'sport': np.random.choice(['Soccer', 'Basketball', 'Tennis', 'NFL']),
                    'home_team': f"Team_{match*2}",
                    'away_team': f"Team_{match*2+1}"
                })
    
    df = pd.DataFrame(data)
    df = df.set_index('timestamp').sort_index()
    
    # Add some technical indicators
    df = df.sort_values(['match_id', 'timestamp'])
    df['returns'] = df.groupby('match_id')['decimal_odds_avg'].pct_change()
    df['ma_20'] = df.groupby('match_id')['decimal_odds_avg'].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df['volatility'] = df.groupby('match_id')['returns'].transform(lambda x: x.rolling(20, min_periods=1).std())
    
    return df


def test_signal_performance(signal: AlphaSignal, data: pd.DataFrame):
    """Test a signal's performance on data."""
    # Simple momentum strategy
    if 'momentum' in signal.signal_id:
        window = signal.parameters.get('window', 20)
        data['signal'] = data.groupby('match_id')['returns'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        data['prediction'] = 0.5 + np.clip(data['signal'] * 10, -0.3, 0.3)
    
    # Mean reversion
    elif 'mean_reversion' in signal.signal_id:
        window = signal.parameters.get('window', 50)
        z_threshold = signal.parameters.get('z_threshold', 2)
        
        data['ma'] = data.groupby('match_id')['decimal_odds_avg'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        data['std'] = data.groupby('match_id')['decimal_odds_avg'].transform(lambda x: x.rolling(window, min_periods=1).std())
        data['z_score'] = (data['decimal_odds_avg'] - data['ma']) / data['std'].fillna(1)
        
        # Bet against extreme moves
        data['signal'] = -data['z_score'] / z_threshold
        data['prediction'] = 0.5 + np.clip(data['signal'], -0.3, 0.3)
    
    # Arbitrage detection
    elif 'arbitrage' in signal.signal_id:
        min_edge = signal.parameters.get('min_edge', 0.01)
        data['edge'] = 1/data['decimal_odds_best'] - 1/data['decimal_odds_avg']
        data['signal'] = np.where(data['edge'] > min_edge, data['edge'] * 10, 0)
        data['prediction'] = 0.5 + np.clip(data['signal'], -0.3, 0.3)
    
    else:
        # Default no edge
        data['prediction'] = 0.5
    
    return data


def run_pipeline_demo():
    """Run the alpha research pipeline demo."""
    # Initialize pipeline
    pipeline = AlphaResearchPipeline()
    
    # Create test signals
    signals = [
        AlphaSignal(
            signal_id="momentum_fast",
            name="Fast Momentum (5 period)",
            description="Very short-term momentum signal",
            formula="momentum(returns, window=5)",
            parameters={'window': 5, 'threshold': 0.02},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="momentum_slow",
            name="Slow Momentum (50 period)",
            description="Longer-term momentum signal",
            formula="momentum(returns, window=50)",
            parameters={'window': 50, 'threshold': 0.01},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="mean_reversion_2std",
            name="Mean Reversion (2 STD)",
            description="Trade reversals at 2 standard deviations",
            formula="mean_reversion(odds, window=30, z_threshold=2)",
            parameters={'window': 30, 'z_threshold': 2},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        )
    ]
    
    # Register signals
    for signal in signals:
        pipeline.register_signal(signal)
        logger.info(f"Registered: {signal.name}")
    
    # Generate synthetic data
    data = generate_synthetic_market_data(n_days=90, n_matches_per_day=20)
    logger.info(f"Generated {len(data)} data points")
    
    # Test each signal through the pipeline stages
    results_summary = []
    
    for signal in signals:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing {signal.name}")
        logger.info(f"{'='*60}")
        
        # Add predictions to data
        test_data = test_signal_performance(signal, data.copy())
        
        # Stage 1: Raw R&D (first 30 days)
        stage1_end = data.index.min() + timedelta(days=30)
        result = pipeline.run_backtest(
            signal.signal_id,
            'raw_rd',
            test_data,
            data.index.min(),
            stage1_end
        )
        
        logger.info(f"\nRaw R&D Results:")
        logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Max DD: {result.max_drawdown:.2%}")
        logger.info(f"  P-value: {result.p_value:.4f}")
        
        # Check promotion criteria
        can_promote, reason = pipeline.evaluate_stage_progression(signal.signal_id)
        
        if can_promote:
            pipeline.promote_signal(signal.signal_id, "Passed raw R&D")
            
            # Stage 2: In-Sample (next 30 days)
            stage2_start = stage1_end
            stage2_end = stage2_start + timedelta(days=30)
            
            result = pipeline.run_backtest(
                signal.signal_id,
                'in_sample',
                test_data,
                stage2_start,
                stage2_end
            )
            
            logger.info(f"\nIn-Sample Results:")
            logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
            logger.info(f"  Win Rate: {result.win_rate:.2%}")
            logger.info(f"  Info Ratio: {result.information_ratio:.2f}")
            
            # Out of sample test
            can_promote, _ = pipeline.evaluate_stage_progression(signal.signal_id)
            if can_promote:
                pipeline.promote_signal(signal.signal_id, "Passed in-sample")
                
                # Stage 3: Out-of-Sample (final 30 days)
                stage3_start = stage2_end
                stage3_end = data.index.max()
                
                result = pipeline.run_backtest(
                    signal.signal_id,
                    'out_of_sample',
                    test_data,
                    stage3_start,
                    stage3_end
                )
                
                logger.info(f"\nOut-of-Sample Results:")
                logger.info(f"  Sharpe: {result.sharpe_ratio:.2f}")
                logger.info(f"  Performance vs In-Sample: {result.sharpe_ratio/1.0:.1%}")
        
        # Get research report
        report = pipeline.generate_research_report(signal.signal_id)
        
        results_summary.append({
            'Signal': signal.name,
            'Current Stage': report['current_stage'],
            'Total Tests': report['total_backtests'],
            'Best Sharpe': report['best_sharpe_ratio'],
            'Recommendation': 'Deploy' if report['recommendation'] == 'Ready for production' else 'More Testing'
        })
    
    # Print summary
    logger.info(f"\n\n{'='*80}")
    logger.info("RESEARCH PIPELINE SUMMARY")
    logger.info(f"{'='*80}")
    
    summary_df = pd.DataFrame(results_summary)
    for _, row in summary_df.iterrows():
        logger.info(f"\n{row['Signal']}:")
        logger.info(f"  Stage: {row['Current Stage']}")
        logger.info(f"  Tests Run: {row['Total Tests']}")
        logger.info(f"  Best Sharpe: {row['Best Sharpe']:.2f}")
        logger.info(f"  Status: {row['Recommendation']}")
    
    # Show signals ready for deployment
    logger.info(f"\n{'='*60}")
    logger.info("DEPLOYMENT RECOMMENDATIONS")
    logger.info(f"{'='*60}")
    
    for signal_id in pipeline.signals:
        signal = pipeline.signals[signal_id]
        if signal.current_stage in ['walk_forward', 'paper_trading', 'live_trading']:
            logger.info(f"✅ {signal.name} - Ready for paper trading")
        elif signal.current_stage == 'out_of_sample':
            logger.info(f"🔄 {signal.name} - Needs walk-forward analysis")
        else:
            logger.info(f"🔬 {signal.name} - Still in research")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_pipeline_demo()