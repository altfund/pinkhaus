#!/usr/bin/env python3
"""
A/B Testing Framework for Strategy Validation
Enables safe parallel testing of strategy variants with statistical analysis
"""

import json
import logging
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import numpy as np
from scipy import stats
from enum import Enum

from strategy_versions import StrategyRegistry, StrategyVersion

logger = logging.getLogger(__name__)


class TestStatus(Enum):
    """Status of an A/B test"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    TERMINATED = "terminated"  # Auto-stopped due to poor performance
    CANCELLED = "cancelled"


class TestResult(Enum):
    """Result of an A/B test"""
    CONTROL_WINS = "control_wins"
    VARIANT_WINS = "variant_wins"
    NO_SIGNIFICANT_DIFF = "no_significant_difference"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class TestMetrics:
    """Performance metrics for a strategy variant"""

    # Trade statistics
    total_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_edge: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0

    # Per-trade metrics (for statistical testing)
    pnl_per_trade: List[float] = field(default_factory=list)
    edge_per_trade: List[float] = field(default_factory=list)

    # Temporal tracking
    daily_pnl: Dict[str, float] = field(default_factory=dict)
    cumulative_pnl: List[float] = field(default_factory=list)

    def update_from_trade(self, pnl: float, edge: float, date: str):
        """Update metrics with a new trade"""
        self.total_trades += 1
        self.total_pnl += pnl
        self.pnl_per_trade.append(pnl)
        self.edge_per_trade.append(edge)

        # Update daily PnL
        if date not in self.daily_pnl:
            self.daily_pnl[date] = 0.0
        self.daily_pnl[date] += pnl

        # Update cumulative PnL
        self.cumulative_pnl.append(self.total_pnl)

        # Recalculate metrics
        self._recalculate_metrics()

    def _recalculate_metrics(self):
        """Recalculate derived metrics"""
        if self.total_trades == 0:
            return

        # Win rate
        wins = sum(1 for pnl in self.pnl_per_trade if pnl > 0)
        self.win_rate = wins / self.total_trades

        # Average edge
        self.avg_edge = np.mean(self.edge_per_trade)

        # Max drawdown
        if self.cumulative_pnl:
            peak = self.cumulative_pnl[0]
            max_dd = 0
            for value in self.cumulative_pnl:
                if value > peak:
                    peak = value
                dd = peak - value
                if dd > max_dd:
                    max_dd = dd
            self.max_drawdown = max_dd

        # Volatility (standard deviation of PnL)
        if len(self.pnl_per_trade) > 1:
            self.volatility = np.std(self.pnl_per_trade)

        # Sharpe ratio (assuming risk-free rate = 0)
        if self.volatility > 0 and self.total_trades > 0:
            avg_pnl = np.mean(self.pnl_per_trade)
            self.sharpe_ratio = (avg_pnl / self.volatility) * np.sqrt(252)  # Annualized

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'total_trades': self.total_trades,
            'total_pnl': self.total_pnl,
            'win_rate': self.win_rate,
            'avg_edge': self.avg_edge,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'daily_pnl': self.daily_pnl,
            'cumulative_pnl': self.cumulative_pnl
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TestMetrics':
        """Create from dictionary"""
        metrics = cls()
        metrics.total_trades = data.get('total_trades', 0)
        metrics.total_pnl = data.get('total_pnl', 0.0)
        metrics.win_rate = data.get('win_rate', 0.0)
        metrics.avg_edge = data.get('avg_edge', 0.0)
        metrics.max_drawdown = data.get('max_drawdown', 0.0)
        metrics.volatility = data.get('volatility', 0.0)
        metrics.sharpe_ratio = data.get('sharpe_ratio', 0.0)
        metrics.daily_pnl = data.get('daily_pnl', {})
        metrics.cumulative_pnl = data.get('cumulative_pnl', [])
        return metrics


@dataclass
class ABTest:
    """Represents an A/B test comparing two strategy versions"""

    test_id: str  # Unique identifier
    name: str  # Human-readable name
    description: str

    control_version: str  # Version string (e.g., "1.0.0")
    variant_version: str  # Version string (e.g., "1.1.0")

    # Test parameters
    start_date: str
    planned_duration_days: int = 14
    min_trades_required: int = 100  # Minimum trades before statistical test
    significance_level: float = 0.05  # Alpha for hypothesis testing

    # Auto-termination rules
    max_drawdown_pct: float = 10.0  # Stop if variant draws down > 10%
    min_trades_before_stop: int = 30  # Don't stop before this many trades

    # Allocation
    control_allocation_pct: float = 80.0  # 80% to control, 20% to variant
    variant_allocation_pct: float = 20.0

    # Metrics
    control_metrics: TestMetrics = field(default_factory=TestMetrics)
    variant_metrics: TestMetrics = field(default_factory=TestMetrics)

    # Status
    status: str = TestStatus.RUNNING.value
    result: Optional[str] = None
    end_date: Optional[str] = None
    termination_reason: Optional[str] = None

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "system"

    def should_allocate_to_variant(self) -> bool:
        """Randomly decide if this trade should go to variant (based on allocation %)"""
        return np.random.random() * 100 < self.variant_allocation_pct

    def check_auto_termination(self) -> Tuple[bool, Optional[str]]:
        """Check if variant should be auto-terminated due to poor performance"""

        # Need minimum trades before considering termination
        if self.variant_metrics.total_trades < self.min_trades_before_stop:
            return False, None

        # Check drawdown
        if self.variant_metrics.max_drawdown > self.max_drawdown_pct:
            return True, f"Variant exceeded max drawdown: {self.variant_metrics.max_drawdown:.1f}% > {self.max_drawdown_pct}%"

        # Check if variant is significantly worse (early stopping)
        if (self.variant_metrics.total_trades >= 50 and
            self.control_metrics.total_trades >= 50):

            # T-test on PnL per trade
            if (len(self.variant_metrics.pnl_per_trade) >= 50 and
                len(self.control_metrics.pnl_per_trade) >= 50):

                t_stat, p_value = stats.ttest_ind(
                    self.variant_metrics.pnl_per_trade,
                    self.control_metrics.pnl_per_trade
                )

                # If variant is significantly worse (one-tailed test)
                if t_stat < 0 and p_value < self.significance_level:
                    return True, f"Variant significantly underperforming (p={p_value:.4f})"

        return False, None

    def is_complete(self) -> bool:
        """Check if test has run for planned duration"""
        if not self.start_date:
            return False

        start = datetime.fromisoformat(self.start_date)
        elapsed_days = (datetime.now(timezone.utc) - start).days

        return elapsed_days >= self.planned_duration_days

    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for statistical analysis"""
        return (self.control_metrics.total_trades >= self.min_trades_required and
                self.variant_metrics.total_trades >= self.min_trades_required)

    def analyze_results(self) -> Dict[str, Any]:
        """Perform statistical analysis of test results"""

        if not self.has_sufficient_data():
            return {
                'result': TestResult.INSUFFICIENT_DATA.value,
                'confidence': 0.0,
                'details': 'Insufficient trades for analysis',
                'control_trades': self.control_metrics.total_trades,
                'variant_trades': self.variant_metrics.total_trades,
                'required': self.min_trades_required
            }

        # Primary metric: PnL per trade
        control_pnl = np.array(self.control_metrics.pnl_per_trade)
        variant_pnl = np.array(self.variant_metrics.pnl_per_trade)

        # T-test for difference in means
        t_stat, p_value = stats.ttest_ind(variant_pnl, control_pnl)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control_pnl) + np.var(variant_pnl)) / 2)
        cohens_d = (np.mean(variant_pnl) - np.mean(control_pnl)) / pooled_std if pooled_std > 0 else 0

        # Determine winner
        if p_value < self.significance_level:
            if t_stat > 0:
                result = TestResult.VARIANT_WINS.value
                winner = self.variant_version
            else:
                result = TestResult.CONTROL_WINS.value
                winner = self.control_version
        else:
            result = TestResult.NO_SIGNIFICANT_DIFF.value
            winner = None

        # Additional metrics comparison
        metrics_comparison = {
            'total_pnl': {
                'control': self.control_metrics.total_pnl,
                'variant': self.variant_metrics.total_pnl,
                'diff': self.variant_metrics.total_pnl - self.control_metrics.total_pnl,
                'diff_pct': ((self.variant_metrics.total_pnl - self.control_metrics.total_pnl) /
                            abs(self.control_metrics.total_pnl) * 100) if self.control_metrics.total_pnl != 0 else 0
            },
            'win_rate': {
                'control': self.control_metrics.win_rate,
                'variant': self.variant_metrics.win_rate,
                'diff': self.variant_metrics.win_rate - self.control_metrics.win_rate
            },
            'sharpe_ratio': {
                'control': self.control_metrics.sharpe_ratio,
                'variant': self.variant_metrics.sharpe_ratio,
                'diff': self.variant_metrics.sharpe_ratio - self.control_metrics.sharpe_ratio
            },
            'max_drawdown': {
                'control': self.control_metrics.max_drawdown,
                'variant': self.variant_metrics.max_drawdown,
                'diff': self.variant_metrics.max_drawdown - self.control_metrics.max_drawdown
            }
        }

        return {
            'result': result,
            'winner': winner,
            'p_value': p_value,
            'confidence': 1 - p_value,
            't_statistic': t_stat,
            'effect_size': cohens_d,
            'significance_level': self.significance_level,
            'metrics_comparison': metrics_comparison,
            'recommendation': self._get_recommendation(result, cohens_d, metrics_comparison)
        }

    def _get_recommendation(self, result: str, effect_size: float,
                          metrics_comparison: Dict) -> str:
        """Generate human-readable recommendation"""

        if result == TestResult.VARIANT_WINS.value:
            if effect_size > 0.5:  # Large effect
                return f"Strong recommendation to promote variant {self.variant_version}. " \
                       f"Significantly better performance with large effect size ({effect_size:.2f})."
            elif effect_size > 0.2:  # Medium effect
                return f"Recommend promoting variant {self.variant_version}. " \
                       f"Statistically significant improvement with medium effect size ({effect_size:.2f})."
            else:  # Small effect
                return f"Variant {self.variant_version} shows improvement but effect is small ({effect_size:.2f}). " \
                       f"Consider longer testing or larger sample."

        elif result == TestResult.CONTROL_WINS.value:
            return f"Keep control version {self.control_version}. " \
                   f"Variant {self.variant_version} underperforms (effect size: {effect_size:.2f})."

        elif result == TestResult.NO_SIGNIFICANT_DIFF.value:
            # Check if variant has other advantages
            if (metrics_comparison['sharpe_ratio']['diff'] > 0.1 or
                metrics_comparison['max_drawdown']['diff'] < -1.0):
                return f"No significant PnL difference, but variant shows better risk metrics. " \
                       f"Consider promoting {self.variant_version} for risk reduction."
            else:
                return f"No significant difference detected. Keep control {self.control_version} " \
                       f"unless variant has strategic advantages (features, maintainability)."

        else:  # INSUFFICIENT_DATA
            return "Continue testing to gather more data."

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'test_id': self.test_id,
            'name': self.name,
            'description': self.description,
            'control_version': self.control_version,
            'variant_version': self.variant_version,
            'start_date': self.start_date,
            'planned_duration_days': self.planned_duration_days,
            'min_trades_required': self.min_trades_required,
            'significance_level': self.significance_level,
            'max_drawdown_pct': self.max_drawdown_pct,
            'min_trades_before_stop': self.min_trades_before_stop,
            'control_allocation_pct': self.control_allocation_pct,
            'variant_allocation_pct': self.variant_allocation_pct,
            'control_metrics': self.control_metrics.to_dict(),
            'variant_metrics': self.variant_metrics.to_dict(),
            'status': self.status,
            'result': self.result,
            'end_date': self.end_date,
            'termination_reason': self.termination_reason,
            'created_at': self.created_at,
            'created_by': self.created_by
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ABTest':
        """Create from dictionary"""
        control_metrics = TestMetrics.from_dict(data.get('control_metrics', {}))
        variant_metrics = TestMetrics.from_dict(data.get('variant_metrics', {}))

        return cls(
            test_id=data['test_id'],
            name=data['name'],
            description=data['description'],
            control_version=data['control_version'],
            variant_version=data['variant_version'],
            start_date=data['start_date'],
            planned_duration_days=data.get('planned_duration_days', 14),
            min_trades_required=data.get('min_trades_required', 100),
            significance_level=data.get('significance_level', 0.05),
            max_drawdown_pct=data.get('max_drawdown_pct', 10.0),
            min_trades_before_stop=data.get('min_trades_before_stop', 30),
            control_allocation_pct=data.get('control_allocation_pct', 80.0),
            variant_allocation_pct=data.get('variant_allocation_pct', 20.0),
            control_metrics=control_metrics,
            variant_metrics=variant_metrics,
            status=data.get('status', TestStatus.RUNNING.value),
            result=data.get('result'),
            end_date=data.get('end_date'),
            termination_reason=data.get('termination_reason'),
            created_at=data.get('created_at', datetime.now(timezone.utc).isoformat()),
            created_by=data.get('created_by', 'system')
        )


class ABTestManager:
    """Manages A/B tests and their lifecycle"""

    def __init__(self, tests_file: str = "ab_tests.json",
                 strategy_registry: Optional[StrategyRegistry] = None):
        self.tests_file = Path(tests_file)
        self.tests: Dict[str, ABTest] = {}
        self.strategy_registry = strategy_registry or StrategyRegistry()
        self._load_tests()

    def _load_tests(self):
        """Load existing tests from disk"""
        if self.tests_file.exists():
            try:
                with open(self.tests_file, 'r') as f:
                    data = json.load(f)

                for test_id, test_data in data.items():
                    self.tests[test_id] = ABTest.from_dict(test_data)

                logger.info(f"Loaded {len(self.tests)} A/B tests from registry")
            except Exception as e:
                logger.error(f"Error loading tests: {e}")
                self.tests = {}
        else:
            logger.info("No existing tests found, starting fresh")

    def _save_tests(self):
        """Save tests to disk"""
        try:
            data = {
                test_id: test.to_dict()
                for test_id, test in self.tests.items()
            }

            with open(self.tests_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.tests)} tests to registry")
        except Exception as e:
            logger.error(f"Error saving tests: {e}")

    def create_test(self,
                   name: str,
                   control_version: str,
                   variant_version: str,
                   description: str = "",
                   duration_days: int = 14,
                   variant_allocation: float = 20.0,
                   created_by: str = "system") -> ABTest:
        """Create a new A/B test"""

        # Validate versions exist
        if not self.strategy_registry.get_version(control_version):
            raise ValueError(f"Control version {control_version} not found in registry")
        if not self.strategy_registry.get_version(variant_version):
            raise ValueError(f"Variant version {variant_version} not found in registry")

        # Generate test ID
        test_id = f"test_{control_version}_vs_{variant_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Check if test already exists
        if test_id in self.tests:
            logger.warning(f"Test {test_id} already exists, returning existing test")
            return self.tests[test_id]

        # Create test
        test = ABTest(
            test_id=test_id,
            name=name,
            description=description,
            control_version=control_version,
            variant_version=variant_version,
            start_date=datetime.now(timezone.utc).isoformat(),
            planned_duration_days=duration_days,
            control_allocation_pct=100 - variant_allocation,
            variant_allocation_pct=variant_allocation,
            created_by=created_by
        )

        self.tests[test_id] = test
        self._save_tests()

        logger.info(f"Created A/B test: {test_id}")
        logger.info(f"  Control: {control_version} ({100-variant_allocation}%)")
        logger.info(f"  Variant: {variant_version} ({variant_allocation}%)")
        logger.info(f"  Duration: {duration_days} days")

        return test

    def get_test(self, test_id: str) -> Optional[ABTest]:
        """Get a specific test"""
        return self.tests.get(test_id)

    def get_active_tests(self) -> List[ABTest]:
        """Get all currently running tests"""
        return [
            test for test in self.tests.values()
            if test.status == TestStatus.RUNNING.value
        ]

    def update_test_metrics(self, test_id: str,
                          variant: str,  # "control" or "variant"
                          pnl: float,
                          edge: float,
                          date: str):
        """Update metrics for a test variant"""

        if test_id not in self.tests:
            logger.error(f"Test {test_id} not found")
            return

        test = self.tests[test_id]

        if variant == "control":
            test.control_metrics.update_from_trade(pnl, edge, date)
        elif variant == "variant":
            test.variant_metrics.update_from_trade(pnl, edge, date)
        else:
            logger.error(f"Invalid variant: {variant}")
            return

        # Check auto-termination
        should_stop, reason = test.check_auto_termination()
        if should_stop:
            self.terminate_test(test_id, reason)

        self._save_tests()

    def monitor_test(self, test_id: str) -> Dict[str, Any]:
        """Monitor test progress and status"""

        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]

        # Calculate progress
        start = datetime.fromisoformat(test.start_date)
        elapsed_days = (datetime.now(timezone.utc) - start).days
        progress_pct = min(100, (elapsed_days / test.planned_duration_days) * 100)

        # Check if complete
        is_complete = test.is_complete()
        has_data = test.has_sufficient_data()

        return {
            'test_id': test_id,
            'name': test.name,
            'status': test.status,
            'progress_pct': progress_pct,
            'elapsed_days': elapsed_days,
            'planned_days': test.planned_duration_days,
            'is_complete': is_complete,
            'has_sufficient_data': has_data,
            'control': {
                'version': test.control_version,
                'trades': test.control_metrics.total_trades,
                'pnl': test.control_metrics.total_pnl,
                'win_rate': test.control_metrics.win_rate,
                'sharpe': test.control_metrics.sharpe_ratio
            },
            'variant': {
                'version': test.variant_version,
                'trades': test.variant_metrics.total_trades,
                'pnl': test.variant_metrics.total_pnl,
                'win_rate': test.variant_metrics.win_rate,
                'sharpe': test.variant_metrics.sharpe_ratio
            },
            'can_analyze': has_data
        }

    def analyze_test(self, test_id: str) -> Dict[str, Any]:
        """Analyze test results"""

        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]
        analysis = test.analyze_results()

        # Update test result
        test.result = analysis['result']
        self._save_tests()

        return analysis

    def complete_test(self, test_id: str) -> Dict[str, Any]:
        """Mark test as completed and analyze final results"""

        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]
        test.status = TestStatus.COMPLETED.value
        test.end_date = datetime.now(timezone.utc).isoformat()

        analysis = test.analyze_results()
        test.result = analysis['result']

        self._save_tests()

        logger.info(f"Completed test {test_id}")
        logger.info(f"  Result: {analysis['result']}")
        logger.info(f"  Winner: {analysis.get('winner', 'None')}")

        return analysis

    def terminate_test(self, test_id: str, reason: str):
        """Terminate a test early (auto-stop)"""

        if test_id not in self.tests:
            logger.error(f"Test {test_id} not found")
            return

        test = self.tests[test_id]
        test.status = TestStatus.TERMINATED.value
        test.end_date = datetime.now(timezone.utc).isoformat()
        test.termination_reason = reason

        self._save_tests()

        logger.warning(f"TERMINATED test {test_id}: {reason}")

    def promote_winner(self, test_id: str) -> bool:
        """Promote the winning variant to active (if variant won)"""

        if test_id not in self.tests:
            raise ValueError(f"Test {test_id} not found")

        test = self.tests[test_id]

        if test.status != TestStatus.COMPLETED.value:
            raise ValueError(f"Test must be completed before promoting winner")

        if not test.result:
            analysis = test.analyze_results()
            test.result = analysis['result']

        if test.result == TestResult.VARIANT_WINS.value:
            # Promote variant to active
            self.strategy_registry.promote_version(test.variant_version)
            logger.info(f"Promoted winning variant {test.variant_version} to active")
            return True
        else:
            logger.info(f"Not promoting - result was {test.result}")
            return False

    def list_tests(self) -> List[Dict[str, Any]]:
        """List all tests with summary info"""
        return [
            {
                'test_id': test.test_id,
                'name': test.name,
                'control': test.control_version,
                'variant': test.variant_version,
                'status': test.status,
                'result': test.result,
                'start_date': test.start_date,
                'control_trades': test.control_metrics.total_trades,
                'variant_trades': test.variant_metrics.total_trades,
                'control_pnl': f"${test.control_metrics.total_pnl:.2f}",
                'variant_pnl': f"${test.variant_metrics.total_pnl:.2f}"
            }
            for test in sorted(self.tests.values(),
                             key=lambda x: x.start_date, reverse=True)
        ]


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Initialize
    strategy_registry = StrategyRegistry()
    test_manager = ABTestManager(strategy_registry=strategy_registry)

    # Create a test
    test = test_manager.create_test(
        name="48h Horizon Test",
        control_version="1.0.0",
        variant_version="1.1.0",
        description="Testing 48h horizon expansion with liquidity checks",
        duration_days=14,
        variant_allocation=20.0,
        created_by="admin"
    )

    print(f"\n=== Created Test: {test.test_id} ===")
    print(f"Control: {test.control_version} ({test.control_allocation_pct}%)")
    print(f"Variant: {test.variant_version} ({test.variant_allocation_pct}%)")

    # Simulate some trades
    print("\n=== Simulating Trades ===")
    for i in range(150):
        # Randomly allocate to control or variant
        if test.should_allocate_to_variant():
            # Variant trade (slightly better performance)
            pnl = np.random.normal(25, 50)
            edge = np.random.normal(3.5, 1.0)
            test_manager.update_test_metrics(
                test.test_id, "variant", pnl, edge,
                datetime.now().strftime("%Y-%m-%d")
            )
        else:
            # Control trade
            pnl = np.random.normal(20, 50)
            edge = np.random.normal(3.0, 1.0)
            test_manager.update_test_metrics(
                test.test_id, "control", pnl, edge,
                datetime.now().strftime("%Y-%m-%d")
            )

    # Monitor progress
    print("\n=== Test Progress ===")
    status = test_manager.monitor_test(test.test_id)
    print(f"Progress: {status['progress_pct']:.1f}%")
    print(f"Control: {status['control']['trades']} trades, ${status['control']['pnl']:.2f} PnL")
    print(f"Variant: {status['variant']['trades']} trades, ${status['variant']['pnl']:.2f} PnL")

    # Analyze results
    if status['can_analyze']:
        print("\n=== Analysis Results ===")
        analysis = test_manager.analyze_test(test.test_id)
        print(f"Result: {analysis['result']}")
        print(f"Winner: {analysis.get('winner', 'None')}")
        print(f"P-value: {analysis.get('p_value', 0):.4f}")
        print(f"Effect size: {analysis.get('effect_size', 0):.3f}")
        print(f"\nRecommendation: {analysis.get('recommendation', 'N/A')}")

    # List all tests
    print("\n=== All Tests ===")
    for test_summary in test_manager.list_tests():
        print(f"{test_summary['test_id']:50} [{test_summary['status']:10}] "
              f"{test_summary['control']} vs {test_summary['variant']}")
