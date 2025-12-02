#!/usr/bin/env python3
"""
Strategy Testing Pipeline
Multi-stage validation pipeline that progressively validates strategies
before risking real money.

Pipeline Stages:
1. Backtest (historical data) - Fast, no risk
2. Paper Trading (live data, no execution) - Real conditions, no risk
3. Shadow Trading (parallel with production) - Side-by-side comparison, no risk
4. Micro A/B Test (1-5% real money) - Validate execution
5. Full A/B Test (20% real money) - Statistical validation
6. Gradual Rollout (20% → 50% → 80% → 100%) - Phased deployment

Each stage has promotion criteria that must be met before advancing.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_versions import StrategyRegistry, StrategyVersion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestingStage(Enum):
    """Stages in the testing pipeline"""
    BACKTEST = "backtest"
    PAPER_TRADING = "paper_trading"
    SHADOW_TRADING = "shadow_trading"
    MICRO_AB_TEST = "micro_ab_test"
    FULL_AB_TEST = "full_ab_test"
    GRADUAL_ROLLOUT = "gradual_rollout"
    PRODUCTION = "production"


class StageStatus(Enum):
    """Status of a testing stage"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class PromotionCriteria:
    """Criteria that must be met to advance to next stage"""

    # Performance requirements
    min_trades: int = 0
    min_win_rate: Optional[float] = None  # e.g., 0.45 = 45%
    min_roi: Optional[float] = None  # e.g., 0.10 = 10%
    min_sharpe_ratio: Optional[float] = None  # e.g., 1.0

    # Risk requirements
    max_drawdown: Optional[float] = None  # e.g., 0.20 = 20%
    max_consecutive_losses: Optional[int] = None

    # Comparison requirements (vs baseline)
    must_beat_baseline: bool = False
    min_improvement_pct: Optional[float] = None  # e.g., 0.05 = 5% better than baseline

    # Statistical requirements
    min_confidence_level: Optional[float] = None  # e.g., 0.95 = 95% confidence
    max_p_value: Optional[float] = None  # e.g., 0.05

    # Time requirements
    min_days: int = 0

    def check_criteria(self, results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Check if results meet promotion criteria
        Returns (passed, reasons)
        """
        passed = True
        reasons = []

        # Check minimum trades
        if results.get('total_trades', 0) < self.min_trades:
            passed = False
            reasons.append(f"Insufficient trades: {results.get('total_trades', 0)} < {self.min_trades}")

        # Check win rate
        if self.min_win_rate is not None:
            win_rate = results.get('win_rate', 0)
            if win_rate < self.min_win_rate:
                passed = False
                reasons.append(f"Win rate too low: {win_rate:.1%} < {self.min_win_rate:.1%}")

        # Check ROI
        if self.min_roi is not None:
            roi = results.get('roi', 0)
            if roi < self.min_roi:
                passed = False
                reasons.append(f"ROI too low: {roi:.1%} < {self.min_roi:.1%}")

        # Check Sharpe ratio
        if self.min_sharpe_ratio is not None:
            sharpe = results.get('sharpe_ratio', 0)
            if sharpe < self.min_sharpe_ratio:
                passed = False
                reasons.append(f"Sharpe ratio too low: {sharpe:.2f} < {self.min_sharpe_ratio:.2f}")

        # Check drawdown
        if self.max_drawdown is not None:
            drawdown = results.get('max_drawdown', 0)
            if drawdown > self.max_drawdown:
                passed = False
                reasons.append(f"Drawdown too high: {drawdown:.1%} > {self.max_drawdown:.1%}")

        # Check consecutive losses
        if self.max_consecutive_losses is not None:
            consec_losses = results.get('max_consecutive_losses', 0)
            if consec_losses > self.max_consecutive_losses:
                passed = False
                reasons.append(f"Too many consecutive losses: {consec_losses} > {self.max_consecutive_losses}")

        # Check time requirement
        if 'duration_days' in results:
            if results['duration_days'] < self.min_days:
                passed = False
                reasons.append(f"Insufficient duration: {results['duration_days']} days < {self.min_days} days")

        # Check baseline comparison
        if self.must_beat_baseline and baseline_results:
            variant_roi = results.get('roi', 0)
            baseline_roi = baseline_results.get('roi', 0)

            if variant_roi <= baseline_roi:
                passed = False
                reasons.append(f"Does not beat baseline: {variant_roi:.1%} <= {baseline_roi:.1%}")

            if self.min_improvement_pct:
                improvement = (variant_roi - baseline_roi) / abs(baseline_roi) if baseline_roi != 0 else 0
                if improvement < self.min_improvement_pct:
                    passed = False
                    reasons.append(f"Insufficient improvement: {improvement:.1%} < {self.min_improvement_pct:.1%}")

        # Check statistical significance
        if self.max_p_value is not None:
            p_value = results.get('p_value', 1.0)
            if p_value > self.max_p_value:
                passed = False
                reasons.append(f"Not statistically significant: p={p_value:.4f} > {self.max_p_value}")

        if passed:
            reasons = ["All criteria met"]

        return passed, reasons


# Default criteria for each stage
DEFAULT_CRITERIA = {
    TestingStage.BACKTEST: PromotionCriteria(
        min_trades=100,
        min_roi=0.05,  # 5% ROI minimum
        max_drawdown=0.25,  # 25% max drawdown
        min_days=0  # Historical data, no time requirement
    ),

    TestingStage.PAPER_TRADING: PromotionCriteria(
        min_trades=50,
        min_roi=0.03,  # 3% ROI minimum
        max_drawdown=0.20,  # 20% max drawdown
        min_days=7  # At least 1 week
    ),

    TestingStage.SHADOW_TRADING: PromotionCriteria(
        min_trades=30,
        must_beat_baseline=True,
        min_improvement_pct=0.0,  # Must at least match baseline
        max_drawdown=0.20,
        min_days=7  # At least 1 week
    ),

    TestingStage.MICRO_AB_TEST: PromotionCriteria(
        min_trades=20,
        must_beat_baseline=False,  # Just validating execution works
        max_drawdown=0.15,  # Tighter limits for real money
        min_days=3  # Short duration for micro test
    ),

    TestingStage.FULL_AB_TEST: PromotionCriteria(
        min_trades=100,
        must_beat_baseline=True,
        min_improvement_pct=0.05,  # 5% better than baseline
        max_p_value=0.05,  # 95% confidence
        max_drawdown=0.20,
        min_days=14  # 2 weeks minimum
    ),

    TestingStage.GRADUAL_ROLLOUT: PromotionCriteria(
        min_trades=50,
        must_beat_baseline=True,
        max_drawdown=0.20,
        min_days=7  # 1 week at each allocation level
    )
}


@dataclass
class StageResult:
    """Results from a testing stage"""
    stage: str
    status: str
    start_date: str
    end_date: Optional[str] = None

    # Performance metrics
    total_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    roi: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    duration_days: float = 0.0

    # Statistical metrics (for comparison stages)
    p_value: Optional[float] = None
    effect_size: Optional[float] = None

    # Promotion decision
    promotion_decision: Optional[bool] = None
    promotion_reasons: List[str] = field(default_factory=list)

    # Raw data
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StageResult':
        return cls(**{k: v for k, v in data.items() if k in cls.__annotations__})


@dataclass
class StrategyTestingPipeline:
    """
    Complete testing pipeline for a strategy version
    Tracks progress through all stages
    """

    strategy_version: str
    baseline_version: str  # Version to compare against

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "system"

    # Current stage
    current_stage: str = TestingStage.BACKTEST.value

    # Results for each stage
    stage_results: Dict[str, StageResult] = field(default_factory=dict)

    # Custom criteria (if overriding defaults)
    custom_criteria: Dict[str, PromotionCriteria] = field(default_factory=dict)

    # Overall status
    pipeline_status: str = "in_progress"  # in_progress, completed, failed, aborted

    def get_current_criteria(self) -> PromotionCriteria:
        """Get promotion criteria for current stage"""
        stage_enum = TestingStage(self.current_stage)

        # Check for custom criteria first
        if self.current_stage in self.custom_criteria:
            return self.custom_criteria[self.current_stage]

        # Fall back to defaults
        return DEFAULT_CRITERIA.get(stage_enum, PromotionCriteria())

    def start_stage(self, stage: TestingStage):
        """Start a new testing stage"""
        self.current_stage = stage.value

        self.stage_results[stage.value] = StageResult(
            stage=stage.value,
            status=StageStatus.IN_PROGRESS.value,
            start_date=datetime.now(timezone.utc).isoformat()
        )

        logger.info(f"Started stage: {stage.value} for strategy {self.strategy_version}")

    def complete_stage(self, results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None) -> Tuple[bool, List[str]]:
        """
        Complete current stage and check promotion criteria
        Returns (should_promote, reasons)
        """
        if self.current_stage not in self.stage_results:
            raise ValueError(f"Stage {self.current_stage} not started")

        stage_result = self.stage_results[self.current_stage]

        # Update result with metrics
        for key, value in results.items():
            if hasattr(stage_result, key):
                setattr(stage_result, key, value)

        stage_result.end_date = datetime.now(timezone.utc).isoformat()

        # Calculate duration
        start = datetime.fromisoformat(stage_result.start_date)
        end = datetime.fromisoformat(stage_result.end_date)
        stage_result.duration_days = (end - start).total_seconds() / 86400

        # Check promotion criteria
        criteria = self.get_current_criteria()
        passed, reasons = criteria.check_criteria(results, baseline_results)

        stage_result.promotion_decision = passed
        stage_result.promotion_reasons = reasons
        stage_result.status = StageStatus.PASSED.value if passed else StageStatus.FAILED.value

        logger.info(f"Completed stage {self.current_stage}: {'PASSED' if passed else 'FAILED'}")
        for reason in reasons:
            logger.info(f"  - {reason}")

        return passed, reasons

    def advance_to_next_stage(self) -> Optional[TestingStage]:
        """Advance to next stage in pipeline"""
        stage_order = [
            TestingStage.BACKTEST,
            TestingStage.PAPER_TRADING,
            TestingStage.SHADOW_TRADING,
            TestingStage.MICRO_AB_TEST,
            TestingStage.FULL_AB_TEST,
            TestingStage.GRADUAL_ROLLOUT,
            TestingStage.PRODUCTION
        ]

        current = TestingStage(self.current_stage)
        current_idx = stage_order.index(current)

        if current_idx >= len(stage_order) - 1:
            # Already at production
            self.pipeline_status = "completed"
            logger.info(f"Strategy {self.strategy_version} reached PRODUCTION")
            return None

        next_stage = stage_order[current_idx + 1]
        self.start_stage(next_stage)

        return next_stage

    def abort_pipeline(self, reason: str):
        """Abort the testing pipeline"""
        self.pipeline_status = "aborted"

        if self.current_stage in self.stage_results:
            self.stage_results[self.current_stage].status = StageStatus.FAILED.value
            self.stage_results[self.current_stage].promotion_reasons = [reason]

        logger.warning(f"ABORTED pipeline for {self.strategy_version}: {reason}")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline progress"""
        stages_completed = sum(
            1 for r in self.stage_results.values()
            if r.status == StageStatus.PASSED.value
        )

        total_stages = len(TestingStage) - 1  # Exclude PRODUCTION

        return {
            'strategy_version': self.strategy_version,
            'baseline_version': self.baseline_version,
            'current_stage': self.current_stage,
            'pipeline_status': self.pipeline_status,
            'stages_completed': stages_completed,
            'total_stages': total_stages,
            'progress_pct': (stages_completed / total_stages) * 100,
            'stage_results': {
                stage: result.to_dict()
                for stage, result in self.stage_results.items()
            }
        }

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'strategy_version': self.strategy_version,
            'baseline_version': self.baseline_version,
            'created_at': self.created_at,
            'created_by': self.created_by,
            'current_stage': self.current_stage,
            'stage_results': {
                stage: result.to_dict()
                for stage, result in self.stage_results.items()
            },
            'pipeline_status': self.pipeline_status
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyTestingPipeline':
        """Create from dictionary"""
        pipeline = cls(
            strategy_version=data['strategy_version'],
            baseline_version=data['baseline_version'],
            created_at=data.get('created_at', datetime.now(timezone.utc).isoformat()),
            created_by=data.get('created_by', 'system'),
            current_stage=data.get('current_stage', TestingStage.BACKTEST.value),
            pipeline_status=data.get('pipeline_status', 'in_progress')
        )

        # Reconstruct stage results
        if 'stage_results' in data:
            for stage, result_data in data['stage_results'].items():
                pipeline.stage_results[stage] = StageResult.from_dict(result_data)

        return pipeline


class PipelineManager:
    """Manages testing pipelines for multiple strategies"""

    def __init__(self,
                 pipelines_file: str = "strategy_testing_pipelines.json",
                 strategy_registry: Optional[StrategyRegistry] = None):
        self.pipelines_file = Path(pipelines_file)
        self.pipelines: Dict[str, StrategyTestingPipeline] = {}
        self.strategy_registry = strategy_registry or StrategyRegistry()
        self._load_pipelines()

    def _load_pipelines(self):
        """Load existing pipelines from disk"""
        if self.pipelines_file.exists():
            try:
                with open(self.pipelines_file, 'r') as f:
                    data = json.load(f)

                for version, pipeline_data in data.items():
                    self.pipelines[version] = StrategyTestingPipeline.from_dict(pipeline_data)

                logger.info(f"Loaded {len(self.pipelines)} testing pipelines")
            except Exception as e:
                logger.error(f"Error loading pipelines: {e}")
                self.pipelines = {}
        else:
            logger.info("No existing pipelines found")

    def _save_pipelines(self):
        """Save pipelines to disk"""
        try:
            data = {
                version: pipeline.to_dict()
                for version, pipeline in self.pipelines.items()
            }

            with open(self.pipelines_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.pipelines)} pipelines")
        except Exception as e:
            logger.error(f"Error saving pipelines: {e}")

    def create_pipeline(self,
                       strategy_version: str,
                       baseline_version: str = "1.0.0",
                       created_by: str = "system") -> StrategyTestingPipeline:
        """Create a new testing pipeline for a strategy"""

        # Validate strategy exists
        if not self.strategy_registry.get_version(strategy_version):
            raise ValueError(f"Strategy version {strategy_version} not found")

        if not self.strategy_registry.get_version(baseline_version):
            raise ValueError(f"Baseline version {baseline_version} not found")

        # Check if pipeline already exists
        if strategy_version in self.pipelines:
            logger.warning(f"Pipeline for {strategy_version} already exists, returning existing")
            return self.pipelines[strategy_version]

        # Create new pipeline
        pipeline = StrategyTestingPipeline(
            strategy_version=strategy_version,
            baseline_version=baseline_version,
            created_by=created_by
        )

        # Start with backtest stage
        pipeline.start_stage(TestingStage.BACKTEST)

        self.pipelines[strategy_version] = pipeline
        self._save_pipelines()

        logger.info(f"Created testing pipeline for {strategy_version} vs {baseline_version}")
        return pipeline

    def get_pipeline(self, strategy_version: str) -> Optional[StrategyTestingPipeline]:
        """Get pipeline for a strategy version"""
        return self.pipelines.get(strategy_version)

    def list_pipelines(self) -> List[Dict[str, Any]]:
        """List all pipelines with summary info"""
        return [
            {
                'strategy_version': pipeline.strategy_version,
                'current_stage': pipeline.current_stage,
                'status': pipeline.pipeline_status,
                'progress': f"{pipeline.get_summary()['progress_pct']:.0f}%",
                'created_at': pipeline.created_at
            }
            for pipeline in sorted(self.pipelines.values(),
                                 key=lambda p: p.created_at, reverse=True)
        ]


if __name__ == "__main__":
    # Example usage
    from strategy_versions import create_enhanced_version, create_experimental_version

    # Initialize
    registry = StrategyRegistry()
    manager = PipelineManager(strategy_registry=registry)

    # Register strategy versions if not already done
    enhanced = create_enhanced_version()
    registry.register_version(enhanced)

    # Create testing pipeline for v1.1.0
    print("\n=== Creating Testing Pipeline ===")
    pipeline = manager.create_pipeline(
        strategy_version="1.1.0",
        baseline_version="1.0.0",
        created_by="admin"
    )

    print(f"Created pipeline for {pipeline.strategy_version}")
    print(f"Current stage: {pipeline.current_stage}")

    # Simulate backtest results
    print("\n=== Simulating Backtest Stage ===")
    backtest_results = {
        'total_trades': 150,
        'win_rate': 0.52,
        'total_pnl': 1200,
        'roi': 0.12,  # 12%
        'sharpe_ratio': 1.5,
        'max_drawdown': 0.18,
        'max_consecutive_losses': 7
    }

    passed, reasons = pipeline.complete_stage(backtest_results)
    print(f"Backtest stage: {'PASSED' if passed else 'FAILED'}")
    for reason in reasons:
        print(f"  - {reason}")

    if passed:
        # Advance to next stage
        print("\n=== Advancing to Paper Trading ===")
        next_stage = pipeline.advance_to_next_stage()
        print(f"Advanced to: {next_stage.value if next_stage else 'PRODUCTION'}")

    # Show pipeline summary
    print("\n=== Pipeline Summary ===")
    summary = pipeline.get_summary()
    print(f"Strategy: {summary['strategy_version']}")
    print(f"Current Stage: {summary['current_stage']}")
    print(f"Progress: {summary['progress_pct']:.0f}%")
    print(f"Status: {summary['pipeline_status']}")

    # List all pipelines
    print("\n=== All Pipelines ===")
    for p in manager.list_pipelines():
        print(f"{p['strategy_version']:10} | {p['current_stage']:20} | {p['status']:15} | {p['progress']}")
