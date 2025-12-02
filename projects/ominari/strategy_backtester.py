#!/usr/bin/env python3
"""
Strategy Backtester
Tests strategy versions against historical data from the database
"""

import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from strategy_versions import StrategyVersion
from sqlalchemy import and_

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyBacktester:
    """
    Backtests a strategy version against historical data
    Uses actual market and odds data from database
    """

    def __init__(self, strategy: StrategyVersion):
        self.strategy = strategy
        self.initial_bankroll = 10000.0
        self.bankroll = self.initial_bankroll

        # Track results
        self.trades: List[Dict] = []
        self.bankroll_history: List[float] = [self.initial_bankroll]
        self.daily_pnl: Dict[str, float] = {}

    def calculate_kelly_bet(self, prob: float, odds: float) -> float:
        """Calculate Kelly bet size"""
        b = odds - 1
        q = 1 - prob
        f = (prob * b - q) / b

        # Apply Kelly fraction
        f = f * self.strategy.kelly_fraction

        if f <= 0:
            return 0

        bet_amount = self.bankroll * f

        # Apply position limits
        max_position = self.bankroll * self.strategy.max_position_pct
        bet_amount = min(bet_amount, max_position)

        return round(bet_amount, 2)

    def calculate_edge(self, fair_prob: float, market_odds: float) -> float:
        """Calculate betting edge"""
        fair_odds = 1 / fair_prob
        edge = ((market_odds / fair_odds) - 1) * 100
        return edge

    def run_backtest(self,
                    start_date: datetime,
                    end_date: datetime,
                    simulate_settlement: bool = True) -> Dict:
        """
        Run backtest over date range

        Args:
            start_date: Start of backtest period
            end_date: End of backtest period
            simulate_settlement: If True, simulate outcomes based on probabilities
                               If False, need actual results (not yet implemented)

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {self.strategy.version} from {start_date.date()} to {end_date.date()}")
        logger.info(f"Strategy: {self.strategy.sports}, {self.strategy.time_horizon_hours}h horizon, {self.strategy.market_query_limit} markets")

        # Get historical markets in date range
        with db_manager.get_db_session() as db:
            # Get markets that matured during our backtest period
            # (so we can simulate them being available for betting earlier)
            markets = db.query(Market).filter(
                and_(
                    Market.maturity_date >= start_date,
                    Market.maturity_date <= end_date,
                    Market.sport.in_(self.strategy.sports)
                )
            ).limit(self.strategy.market_query_limit * 10).all()  # Get extra for filtering

            logger.info(f"Found {len(markets)} historical markets in date range")

            # Simulate trading day by day
            current_date = start_date
            while current_date < end_date:
                self._simulate_trading_day(db, current_date, simulate_settlement)
                current_date += timedelta(days=1)

        # Calculate final metrics
        results = self._calculate_results()

        logger.info(f"Backtest complete: {results['total_trades']} trades, {results['roi']:.1%} ROI")
        return results

    def _simulate_trading_day(self, db, date: datetime, simulate_settlement: bool):
        """Simulate trading on a specific day"""

        # Get markets available for trading on this day
        # (markets that mature within our time horizon)
        horizon_end = date + timedelta(hours=self.strategy.time_horizon_hours)

        available_markets = db.query(Market).filter(
            and_(
                Market.maturity_date > date,
                Market.maturity_date <= horizon_end,
                Market.sport.in_(self.strategy.sports)
            )
        ).limit(self.strategy.market_query_limit).all()

        if not available_markets:
            return

        # Find opportunities
        opportunities = []
        for market in available_markets:
            # Get odds for this market
            odds_records = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).all()

            if not odds_records:
                continue

            # Group by outcome
            odds_by_outcome = {}
            for odd in odds_records:
                if odd.outcome not in odds_by_outcome or odd.decimal_odds > odds_by_outcome[odd.outcome].decimal_odds:
                    odds_by_outcome[odd.outcome] = odd

            # Calculate fair probabilities
            total_prob = sum(1/odd.decimal_odds for odd in odds_by_outcome.values())

            for outcome, odd in odds_by_outcome.items():
                implied_prob = 1 / odd.decimal_odds
                fair_prob = implied_prob / total_prob
                edge = self.calculate_edge(fair_prob, odd.decimal_odds)

                # Check if meets edge threshold
                if edge >= self.strategy.min_conservative_edge:
                    opportunities.append({
                        'market': market,
                        'outcome': outcome,
                        'odds': odd.decimal_odds,
                        'fair_prob': fair_prob,
                        'edge': edge,
                        'date': date
                    })

        # Sort by edge
        opportunities.sort(key=lambda x: x['edge'], reverse=True)

        # Place bets (respecting position limits)
        current_exposure = sum(
            trade['stake'] for trade in self.trades
            if trade['status'] == 'open'
        )

        max_exposure = self.bankroll * self.strategy.max_portfolio_pct
        open_positions = sum(1 for t in self.trades if t['status'] == 'open')

        for opp in opportunities:
            # Check position limits
            if open_positions >= self.strategy.max_open_positions:
                break

            # Calculate bet size
            bet_size = self.calculate_kelly_bet(opp['fair_prob'], opp['odds'])

            if bet_size < 10.0:  # Min bet
                continue

            # Check exposure limit
            if current_exposure + bet_size > max_exposure:
                continue

            # Place bet
            trade = {
                'date': date.isoformat(),
                'market_id': opp['market'].source_id,
                'outcome': opp['outcome'],
                'stake': bet_size,
                'odds': opp['odds'],
                'fair_prob': opp['fair_prob'],
                'edge': opp['edge'],
                'maturity_date': opp['market'].maturity_date.isoformat(),
                'status': 'open',
                'pnl': 0
            }

            self.trades.append(trade)
            current_exposure += bet_size
            open_positions += 1

            logger.debug(f"Placed bet: ${bet_size:.2f} on {opp['outcome']} @ {opp['odds']:.2f} (edge: {opp['edge']:.2f}%)")

        # Settle mature positions
        if simulate_settlement:
            self._settle_mature_positions(date)

    def _settle_mature_positions(self, current_date: datetime):
        """Settle positions that have matured"""

        for trade in self.trades:
            if trade['status'] != 'open':
                continue

            # Check if position has matured
            maturity = datetime.fromisoformat(trade['maturity_date'])
            if maturity.replace(tzinfo=None) <= current_date.replace(tzinfo=None):
                # Simulate outcome based on fair probability
                won = np.random.random() < trade['fair_prob']

                if won:
                    # Win
                    pnl = trade['stake'] * (trade['odds'] - 1)
                    trade['status'] = 'won'
                else:
                    # Loss
                    pnl = -trade['stake']
                    trade['status'] = 'lost'

                trade['pnl'] = pnl
                trade['settlement_date'] = current_date.isoformat()

                # Update bankroll
                self.bankroll += pnl
                self.bankroll_history.append(self.bankroll)

                # Track daily P&L
                date_key = current_date.date().isoformat()
                if date_key not in self.daily_pnl:
                    self.daily_pnl[date_key] = 0
                self.daily_pnl[date_key] += pnl

                logger.debug(f"Settled: {'WON' if won else 'LOST'} ${abs(pnl):.2f} (Bankroll: ${self.bankroll:.2f})")

    def _calculate_results(self) -> Dict:
        """Calculate final backtest results"""

        settled_trades = [t for t in self.trades if t['status'] in ['won', 'lost']]

        if not settled_trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'max_consecutive_losses': 0,
                'final_bankroll': self.initial_bankroll
            }

        # Basic metrics
        wins = sum(1 for t in settled_trades if t['status'] == 'won')
        total_trades = len(settled_trades)
        win_rate = wins / total_trades

        total_pnl = sum(t['pnl'] for t in settled_trades)
        roi = total_pnl / self.initial_bankroll

        # Calculate Sharpe ratio
        if len(settled_trades) > 1:
            pnl_per_trade = [t['pnl'] for t in settled_trades]
            avg_pnl = np.mean(pnl_per_trade)
            std_pnl = np.std(pnl_per_trade)
            sharpe_ratio = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown
        max_drawdown = 0
        peak = self.bankroll_history[0]
        for value in self.bankroll_history:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # Calculate max consecutive losses
        max_consecutive_losses = 0
        current_streak = 0
        for trade in settled_trades:
            if trade['status'] == 'lost':
                current_streak += 1
                max_consecutive_losses = max(max_consecutive_losses, current_streak)
            else:
                current_streak = 0

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_consecutive_losses': max_consecutive_losses,
            'final_bankroll': self.bankroll,
            'avg_edge': np.mean([t['edge'] for t in self.trades]),
            'trades_detail': settled_trades[:10]  # First 10 for inspection
        }


def run_strategy_backtest(strategy_version: str,
                         start_date: datetime,
                         end_date: datetime) -> Dict:
    """
    Convenience function to run backtest for a strategy version
    """
    from strategy_versions import StrategyRegistry

    registry = StrategyRegistry()
    strategy = registry.get_version(strategy_version)

    if not strategy:
        raise ValueError(f"Strategy {strategy_version} not found")

    backtester = StrategyBacktester(strategy)
    results = backtester.run_backtest(start_date, end_date)

    return results


if __name__ == "__main__":
    # Example: Backtest v1.1.0 against historical data
    from strategy_versions import StrategyRegistry, create_enhanced_version

    # Register strategies
    registry = StrategyRegistry()

    # Create v1.1.0 if not exists
    enhanced = create_enhanced_version()
    registry.register_version(enhanced)

    # Run backtest
    print("\n=== Running Backtest for v1.1.0 ===")

    # Backtest over last 30 days of available data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=30)

    results = run_strategy_backtest("1.1.0", start_date, end_date)

    print(f"\nBacktest Results:")
    print(f"  Total Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']:.1%}")
    print(f"  Total P&L: ${results['total_pnl']:.2f}")
    print(f"  ROI: {results['roi']:.1%}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.1%}")
    print(f"  Max Consecutive Losses: {results['max_consecutive_losses']}")
    print(f"  Final Bankroll: ${results['final_bankroll']:.2f}")
    print(f"  Average Edge: {results['avg_edge']:.2f}%")

    # Check if would pass promotion criteria
    from strategy_testing_pipeline import DEFAULT_CRITERIA, TestingStage

    criteria = DEFAULT_CRITERIA[TestingStage.BACKTEST]
    passed, reasons = criteria.check_criteria(results)

    print(f"\nPromotion Decision: {'PASS' if passed else 'FAIL'}")
    for reason in reasons:
        print(f"  - {reason}")
