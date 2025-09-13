#!/usr/bin/env python3
"""
Integration script to connect the alpha research pipeline
with the existing trading system.
"""

import logging
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

from alpha_research_pipeline import AlphaResearchPipeline, AlphaSignal
from database_v2 import db_manager
from models import Market, Odd
from signals import SignalProvider, SIGNAL_PROVIDERS
from sqlalchemy import and_, func

logger = logging.getLogger(__name__)


class ResearchSignalAdapter(SignalProvider):
    """
    Adapter to use alpha research signals in the main trading system.
    This allows signals in research to be tested in production-like conditions.
    """
    
    def __init__(self, signal_id: str, formula: str, parameters: Dict):
        self.signal_id = signal_id
        self.formula = formula
        self.parameters = parameters
        self.name = f"research_{signal_id}"
        
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """Generate probability predictions based on research formula."""
        try:
            # Simple implementation - in practice would parse and execute formula
            if 'momentum' in self.formula.lower():
                # Momentum strategy
                window = self.parameters.get('window', 20)
                if 'normalized_implied_home' in df.columns:
                    returns = df['normalized_implied_home'].pct_change()
                    signal = returns.rolling(window).mean()
                    # Convert to probability
                    probs = 0.5 + np.clip(signal * 10, -0.3, 0.3)
                    return probs
            elif 'mean_reversion' in self.formula.lower():
                # Mean reversion strategy
                window = self.parameters.get('window', 50)
                if 'normalized_implied_home' in df.columns:
                    ma = df['normalized_implied_home'].rolling(window).mean()
                    deviation = (df['normalized_implied_home'] - ma) / ma
                    # Bet against extreme deviations
                    probs = 0.5 - np.clip(deviation * 5, -0.3, 0.3)
                    return probs
            elif 'arbitrage' in self.formula.lower():
                # Cross-bookmaker arbitrage
                if 'decimal_odds_best' in df.columns and 'decimal_odds_avg' in df.columns:
                    edge = (1/df['decimal_odds_best'] - 1/df['decimal_odds_avg'])
                    probs = 0.5 + np.clip(edge * 20, -0.3, 0.3)
                    return probs
        except Exception as e:
            logger.error(f"Error in research signal {self.signal_id}: {e}")
        
        # Default to no edge
        return pd.Series(0.5, index=df.index)


def create_research_signals():
    """Create a set of research signals to test."""
    signals = [
        AlphaSignal(
            signal_id="momentum_short",
            name="Short-term Momentum",
            description="Momentum based on recent price movements",
            formula="momentum(odds, window=10)",
            parameters={'window': 10, 'threshold': 0.02},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="momentum_long",
            name="Long-term Momentum",
            description="Momentum based on longer price trends",
            formula="momentum(odds, window=50)",
            parameters={'window': 50, 'threshold': 0.01},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="mean_reversion",
            name="Mean Reversion",
            description="Bet against extreme odds movements",
            formula="mean_reversion(odds, window=50)",
            parameters={'window': 50, 'z_threshold': 2},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="arbitrage_detector",
            name="Bookmaker Arbitrage",
            description="Detect pricing inefficiencies across bookmakers",
            formula="arbitrage(best_odds, avg_odds)",
            parameters={'min_edge': 0.01},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="volume_indicator",
            name="Volume-based Signal",
            description="Use betting volume as predictive signal",
            formula="volume_weighted(odds, volume)",
            parameters={'volume_threshold': 1000},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        ),
        AlphaSignal(
            signal_id="sentiment_tracker",
            name="Market Sentiment",
            description="Track overall market sentiment shifts",
            formula="sentiment_score(odds_changes, time)",
            parameters={'decay_factor': 0.95, 'min_changes': 5},
            created_at=datetime.now(timezone.utc),
            current_stage='raw_rd'
        )
    ]
    return signals


def fetch_historical_data(days_back: int = 90) -> pd.DataFrame:
    """Fetch historical odds data for research."""
    with db_manager.get_db_session() as db:
        since = datetime.now(timezone.utc) - timedelta(days=days_back)
        
        # Get aggregated odds data
        query = db.query(
            Market.source_id,
            Market.sport,
            Market.home_team,
            Market.away_team,
            Market.maturity_date,
            func.min(Odd.decimal_odds).label('decimal_odds_best'),
            func.avg(Odd.decimal_odds).label('decimal_odds_avg'),
            func.max(Odd.decimal_odds).label('decimal_odds_worst'),
            func.count(Odd.id).label('n_bookmakers'),
            func.avg(Odd.normalized_implied).label('avg_implied'),
            func.min(Odd.updated_at).label('first_seen'),
            func.max(Odd.updated_at).label('last_seen')
        ).join(
            Odd, Market.source_id == Odd.source_id
        ).filter(
            and_(
                Market.maturity_date >= since,
                Market.maturity_date <= datetime.now(timezone.utc),
                Odd.outcome == 'option_1'  # Home team
            )
        ).group_by(
            Market.source_id,
            Market.sport,
            Market.home_team,
            Market.away_team,
            Market.maturity_date
        ).all()
        
        # Convert to dataframe
        data = []
        for row in query:
            data.append({
                'source_id': row.source_id,
                'sport': row.sport,
                'home_team': row.home_team,
                'away_team': row.away_team,
                'maturity_date': row.maturity_date,
                'decimal_odds_best': row.decimal_odds_best,
                'decimal_odds_avg': row.decimal_odds_avg,
                'decimal_odds_worst': row.decimal_odds_worst,
                'n_bookmakers': row.n_bookmakers,
                'normalized_implied_home': row.avg_implied,
                'first_seen': row.first_seen,
                'last_seen': row.last_seen
            })
        
        df = pd.DataFrame(data)
        if not df.empty:
            df = df.set_index('maturity_date').sort_index()
        
        return df


def run_research_pipeline():
    """Run the complete alpha research pipeline."""
    # Initialize pipeline
    pipeline = AlphaResearchPipeline()
    
    # Register research signals
    signals = create_research_signals()
    for signal in signals:
        pipeline.register_signal(signal)
        logger.info(f"Registered signal: {signal.name}")
    
    # Fetch historical data
    logger.info("Fetching historical data...")
    data = fetch_historical_data(days_back=180)
    
    if data.empty:
        logger.warning("No historical data available")
        return
    
    logger.info(f"Loaded {len(data)} historical matches")
    
    # Run backtests for each signal
    results = {}
    for signal in signals:
        logger.info(f"\nTesting signal: {signal.name}")
        
        # Raw R&D backtest (first 60 days)
        start_date = data.index.min()
        mid_date = start_date + timedelta(days=60)
        
        result = pipeline.run_backtest(
            signal.signal_id,
            'raw_rd',
            data,
            start_date,
            mid_date
        )
        
        results[signal.signal_id] = result
        
        logger.info(f"Results - Sharpe: {result.sharpe_ratio:.2f}, "
                   f"Win Rate: {result.win_rate:.2%}, "
                   f"P-value: {result.p_value:.4f}")
        
        # Check if ready for promotion
        can_promote, reason = pipeline.evaluate_stage_progression(signal.signal_id)
        if can_promote:
            logger.info(f"Signal ready for promotion to in-sample testing")
            pipeline.promote_signal(signal.signal_id, "Passed raw R&D criteria")
        else:
            logger.info(f"Not ready for promotion: {reason}")
    
    # Generate summary report
    logger.info("\n" + "="*60)
    logger.info("RESEARCH SUMMARY")
    logger.info("="*60)
    
    for signal_id, result in results.items():
        signal = next(s for s in signals if s.signal_id == signal_id)
        logger.info(f"\n{signal.name}:")
        logger.info(f"  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  Win Rate: {result.win_rate:.2%}")
        logger.info(f"  Max Drawdown: {result.max_drawdown:.2%}")
        logger.info(f"  Total Bets: {result.n_bets}")
        logger.info(f"  P-value: {result.p_value:.4f}")
        
        # Add to main signal providers if performing well
        if result.sharpe_ratio > 1.0 and result.p_value < 0.05:
            adapter = ResearchSignalAdapter(
                signal.signal_id,
                signal.formula,
                signal.parameters
            )
            if adapter.name not in [p.name for p in SIGNAL_PROVIDERS]:
                SIGNAL_PROVIDERS.append(adapter)
                logger.info(f"  ✅ Added to active signal providers!")


def monitor_live_performance():
    """Monitor how research signals perform in paper trading."""
    # This would connect to paper trading results
    # and feed performance back to the research pipeline
    pass


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    run_research_pipeline()