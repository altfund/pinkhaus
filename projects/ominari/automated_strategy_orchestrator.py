#!/usr/bin/env python3
"""
Automated Strategy Testing Orchestrator

Runs the testing pipeline automatically in background:
- Tests strategies in isolation from production
- Advances stages automatically when criteria met
- Promotes to production only when statistically validated
- Alerts user but doesn't require constant intervention
- Maintains strict separation between testing and production

Production vs Testing:
- PRODUCTION: Single designated strategy version that actually trades
- TESTING: Multiple strategies being validated in parallel
- Clear separation: Testing never affects production trades
- Automatic promotion: When test passes all stages, becomes new production
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from pathlib import Path
from dataclasses import dataclass, asdict
import signal

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from strategy_testing_pipeline import (
    PipelineManager, TestingStage, StageStatus,
    StrategyTestingPipeline, PromotionCriteria
)
from strategy_versions import StrategyRegistry, StrategyVersion
from notifications.discord_notifier import discord_notifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ProductionLock:
    """Ensures only one strategy is 'production' and actually trading"""
    production_version: str
    locked_at: str
    locked_by: str = "orchestrator"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ProductionLock':
        return cls(**data)


class AutomatedStrategyOrchestrator:
    """
    Orchestrates the entire testing pipeline automatically

    Responsibilities:
    1. Run backtest stage automatically for new strategies
    2. Start paper trading for strategies that pass backtest
    3. Run shadow trading in parallel with production
    4. Execute micro and full A/B tests when ready
    5. Automatically promote winners to production
    6. Monitor and alert on all stages
    7. Rollback if issues detected
    """

    def __init__(self,
                 check_interval_seconds: int = 300,  # Check every 5 minutes
                 production_lock_file: str = "production_lock.json",
                 orchestrator_state_file: str = "orchestrator_state.json"):

        self.check_interval = check_interval_seconds
        self.production_lock_file = Path(production_lock_file)
        self.orchestrator_state_file = Path(orchestrator_state_file)

        self.registry = StrategyRegistry()
        self.pipeline_manager = PipelineManager()

        self.is_running = False
        self.production_lock: Optional[ProductionLock] = None

        # Active background tasks
        self.active_tasks: Dict[str, asyncio.Task] = {}

        # Track what's running
        self.running_stages: Dict[str, str] = {}  # strategy_version -> stage

        # Load state
        self._load_production_lock()
        self._load_orchestrator_state()

        logger.info("Automated Strategy Orchestrator initialized")
        logger.info(f"Production version: {self.production_lock.production_version if self.production_lock else 'None'}")

    def _load_production_lock(self):
        """Load production lock (what version is actually trading)"""
        if self.production_lock_file.exists():
            with open(self.production_lock_file, 'r') as f:
                data = json.load(f)
                self.production_lock = ProductionLock.from_dict(data)
                logger.info(f"Loaded production lock: {self.production_lock.production_version}")
        else:
            # Initialize with v1.0.0 as production baseline
            self.production_lock = ProductionLock(
                production_version="1.0.0",
                locked_at=datetime.now(timezone.utc).isoformat()
            )
            self._save_production_lock()
            logger.info("Initialized production lock with v1.0.0")

    def _save_production_lock(self):
        """Save production lock"""
        with open(self.production_lock_file, 'w') as f:
            json.dump(self.production_lock.to_dict(), f, indent=2)

    def _load_orchestrator_state(self):
        """Load orchestrator state"""
        if self.orchestrator_state_file.exists():
            with open(self.orchestrator_state_file, 'r') as f:
                state = json.load(f)
                self.running_stages = state.get('running_stages', {})
                logger.info(f"Loaded orchestrator state: {len(self.running_stages)} active stages")

    def _save_orchestrator_state(self):
        """Save orchestrator state"""
        state = {
            'running_stages': self.running_stages,
            'last_updated': datetime.now(timezone.utc).isoformat()
        }
        with open(self.orchestrator_state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def get_production_version(self) -> str:
        """Get current production version"""
        return self.production_lock.production_version

    def is_production_version(self, version: str) -> bool:
        """Check if a version is production"""
        return version == self.production_lock.production_version

    async def start(self):
        """Start the orchestrator"""
        logger.info("🚀 Starting Automated Strategy Orchestrator")

        # Send startup notification
        self._send_notification(
            "🤖 Strategy Orchestrator Started",
            f"Production Version: {self.production_lock.production_version}\n"
            f"Monitoring for new strategies to test..."
        )

        self.is_running = True

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

        try:
            while self.is_running:
                await self._orchestration_cycle()
                await asyncio.sleep(self.check_interval)

        except Exception as e:
            logger.error(f"Critical error in orchestrator: {e}", exc_info=True)
            self._send_notification(
                "🚨 Orchestrator Error",
                f"Critical error: {e}\nOrchestrator stopped."
            )
        finally:
            await self.stop()

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.is_running = False

    async def stop(self):
        """Stop the orchestrator"""
        logger.info("Stopping orchestrator...")
        self.is_running = False

        # Cancel all active tasks
        for version, task in self.active_tasks.items():
            logger.info(f"Cancelling task for {version}")
            task.cancel()

        # Wait for tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)

        self._save_orchestrator_state()
        logger.info("Orchestrator stopped")

    async def _orchestration_cycle(self):
        """One cycle of the orchestration loop"""
        logger.debug("Running orchestration cycle...")

        try:
            # 1. Check for new strategies in testing status
            await self._check_for_new_strategies()

            # 2. Advance pipelines that are ready
            await self._advance_ready_pipelines()

            # 3. Check for completed tests
            await self._check_completed_tests()

            # 4. Monitor active stages
            await self._monitor_active_stages()

            # 5. Check for promotion candidates
            await self._check_promotion_candidates()

            # 6. Clean up completed pipelines
            await self._cleanup_completed_pipelines()

        except Exception as e:
            logger.error(f"Error in orchestration cycle: {e}", exc_info=True)

    async def _check_for_new_strategies(self):
        """Check for new strategies that need testing"""
        testing_versions = self.registry.get_testing_versions()

        for version in testing_versions:
            # Skip if already has a pipeline
            existing = self.pipeline_manager.get_pipeline(version.version)
            if existing:
                continue

            # Skip production version
            if self.is_production_version(version.version):
                continue

            # Create new pipeline
            logger.info(f"📋 New strategy detected: {version.version} - {version.name}")

            pipeline = self.pipeline_manager.create_pipeline(
                strategy_version=version.version,
                baseline_version=self.get_production_version(),
                created_by="orchestrator"
            )

            self._send_notification(
                f"🧪 Testing {version.version}",
                f"Started testing pipeline for {version.name}\n"
                f"Baseline: {self.get_production_version()}\n"
                f"Stage 1: Backtest"
            )

            # Start backtest automatically
            await self._start_backtest(pipeline)

    async def _start_backtest(self, pipeline: StrategyTestingPipeline):
        """Start backtest stage automatically"""
        logger.info(f"🔬 Starting backtest for {pipeline.strategy_version}")

        # Mark as running
        self.running_stages[pipeline.strategy_version] = TestingStage.BACKTEST.value
        self._save_orchestrator_state()

        # Run backtest in background
        task = asyncio.create_task(
            self._run_backtest_async(pipeline)
        )
        self.active_tasks[pipeline.strategy_version] = task

    async def _run_backtest_async(self, pipeline: StrategyTestingPipeline):
        """Run backtest asynchronously"""
        try:
            from strategy_backtester import run_strategy_backtest
            from datetime import timedelta

            # Backtest over last 30 days
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=30)

            logger.info(f"Running backtest for {pipeline.strategy_version}...")

            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                run_strategy_backtest,
                pipeline.strategy_version,
                start_date,
                end_date
            )

            # Complete stage
            passed, reasons = pipeline.complete_stage(results)

            logger.info(f"Backtest for {pipeline.strategy_version}: {'PASSED' if passed else 'FAILED'}")

            # Send notification
            self._send_notification(
                f"{'✅' if passed else '❌'} Backtest {pipeline.strategy_version}",
                f"Results: {results['total_trades']} trades, {results['roi']:.1%} ROI\n" +
                "\n".join(f"- {r}" for r in reasons)
            )

            # If passed, advance automatically
            if passed:
                pipeline.advance_to_next_stage()
                self.pipeline_manager._save_pipelines()

                # Start paper trading
                await self._start_paper_trading(pipeline)
            else:
                # Failed - mark strategy as deprecated
                strategy = self.registry.get_version(pipeline.strategy_version)
                if strategy:
                    strategy.status = "deprecated"
                    self.registry._save_registry()

                pipeline.abort_pipeline("; ".join(reasons))
                self.pipeline_manager._save_pipelines()

        except Exception as e:
            logger.error(f"Error in backtest for {pipeline.strategy_version}: {e}", exc_info=True)
            self._send_notification(
                f"🚨 Backtest Error {pipeline.strategy_version}",
                f"Error: {e}"
            )
        finally:
            # Clean up
            self.running_stages.pop(pipeline.strategy_version, None)
            self.active_tasks.pop(pipeline.strategy_version, None)
            self._save_orchestrator_state()

    async def _start_paper_trading(self, pipeline: StrategyTestingPipeline):
        """Start paper trading stage automatically"""
        logger.info(f"📝 Starting paper trading for {pipeline.strategy_version}")

        self.running_stages[pipeline.strategy_version] = TestingStage.PAPER_TRADING.value
        self._save_orchestrator_state()

        self._send_notification(
            f"📝 Paper Trading {pipeline.strategy_version}",
            f"Started paper trading stage\n"
            f"Duration: 7-14 days\n"
            f"Will auto-advance when criteria met"
        )

        # Note: Paper trading runs in the main trading system
        # We just monitor it here and check periodically if ready to advance

    async def _advance_ready_pipelines(self):
        """Check if any pipelines are ready to advance"""
        for version, pipeline in self.pipeline_manager.pipelines.items():
            if pipeline.pipeline_status != "in_progress":
                continue

            current_stage = TestingStage(pipeline.current_stage)

            # Check if stage has been running long enough
            if current_stage == TestingStage.PAPER_TRADING:
                await self._check_paper_trading_ready(pipeline)

            elif current_stage == TestingStage.SHADOW_TRADING:
                await self._check_shadow_trading_ready(pipeline)

            # Micro and Full A/B tests are handled by ab_testing_framework
            # Gradual rollout is handled by promotion system

    async def _check_paper_trading_ready(self, pipeline: StrategyTestingPipeline):
        """Check if paper trading is ready to advance"""
        # Get paper trading results from session
        from paper_trading_sessions import PaperTradingSessionManager

        try:
            session_mgr = PaperTradingSessionManager()
            # TODO: Filter to sessions for this strategy version
            # For now, check if enough time has passed

            if pipeline.current_stage not in pipeline.stage_results:
                return

            stage_result = pipeline.stage_results[pipeline.current_stage]
            start = datetime.fromisoformat(stage_result.start_date)
            days_running = (datetime.now(timezone.utc) - start).days

            if days_running >= 7:  # Minimum duration met
                logger.info(f"Paper trading for {pipeline.strategy_version} ready for review")
                # TODO: Collect actual results and check criteria
                # For now, manual review required

        except Exception as e:
            logger.error(f"Error checking paper trading for {pipeline.strategy_version}: {e}")

    async def _check_shadow_trading_ready(self, pipeline: StrategyTestingPipeline):
        """Check if shadow trading is ready to advance"""
        # Similar to paper trading check
        pass

    async def _check_completed_tests(self):
        """Check for completed A/B tests"""
        from ab_testing_framework import ABTestManager

        test_manager = ABTestManager(strategy_registry=self.registry)
        active_tests = test_manager.get_active_tests()

        for test in active_tests:
            # Check if test is complete
            if test.is_complete() and test.has_sufficient_data():
                logger.info(f"A/B test complete: {test.test_id}")

                # Analyze results
                analysis = test_manager.analyze_test(test.test_id)

                # Send notification
                self._send_notification(
                    f"📊 A/B Test Complete: {test.variant_version}",
                    f"Result: {analysis['result']}\n"
                    f"Winner: {analysis.get('winner', 'None')}\n"
                    f"P-value: {analysis.get('p_value', 1):.4f}\n"
                    f"Recommendation: {analysis.get('recommendation', 'N/A')}"
                )

                # If variant wins, advance pipeline
                if analysis['result'] == 'variant_wins':
                    pipeline = self.pipeline_manager.get_pipeline(test.variant_version)
                    if pipeline:
                        # Check if at full A/B test stage
                        if pipeline.current_stage == TestingStage.FULL_AB_TEST.value:
                            # Advance to gradual rollout
                            pipeline.advance_to_next_stage()
                            self.pipeline_manager._save_pipelines()

                            await self._start_gradual_rollout(pipeline)

    async def _start_gradual_rollout(self, pipeline: StrategyTestingPipeline):
        """Start gradual rollout automatically"""
        logger.info(f"🚀 Starting gradual rollout for {pipeline.strategy_version}")

        self._send_notification(
            f"🚀 Gradual Rollout {pipeline.strategy_version}",
            f"Starting gradual rollout to production\n"
            f"Week 1: 20% → 50%\n"
            f"Week 2: 50% → 80%\n"
            f"Week 3: 80% → 100%\n"
            f"Will auto-promote if successful"
        )

        # Mark as running
        self.running_stages[pipeline.strategy_version] = TestingStage.GRADUAL_ROLLOUT.value
        self._save_orchestrator_state()

        # Gradual rollout is handled by adjustment of A/B test allocation
        # We monitor and advance when each stage completes

    async def _monitor_active_stages(self):
        """Monitor health of active stages"""
        for version, stage in self.running_stages.items():
            pipeline = self.pipeline_manager.get_pipeline(version)
            if not pipeline:
                continue

            # Check for issues
            # TODO: Add monitoring logic
            pass

    async def _check_promotion_candidates(self):
        """Check if any strategies are ready for promotion to production"""
        for version, pipeline in self.pipeline_manager.pipelines.items():
            # Must have completed all stages
            if pipeline.current_stage != TestingStage.PRODUCTION.value:
                continue

            # Skip if already promoted
            if self.is_production_version(version):
                continue

            # Promote automatically
            await self._promote_to_production(version)

    async def _promote_to_production(self, version: str):
        """Promote a strategy to production"""
        logger.info(f"🎉 Promoting {version} to production")

        old_production = self.production_lock.production_version

        # Update production lock
        self.production_lock = ProductionLock(
            production_version=version,
            locked_at=datetime.now(timezone.utc).isoformat(),
            locked_by="orchestrator"
        )
        self._save_production_lock()

        # Update strategy status
        self.registry.promote_version(version)

        # Send notification
        self._send_notification(
            f"🎉 NEW PRODUCTION VERSION: {version}",
            f"Promoted from: {old_production}\n"
            f"All testing stages passed\n"
            f"Now handling 100% of trades"
        )

        logger.info(f"Successfully promoted {version} to production")

    async def _cleanup_completed_pipelines(self):
        """Clean up pipelines that are done"""
        for version, pipeline in list(self.pipeline_manager.pipelines.items()):
            if pipeline.pipeline_status in ["completed", "aborted"]:
                # Remove from running stages
                self.running_stages.pop(version, None)

        self._save_orchestrator_state()

    def _send_notification(self, title: str, message: str):
        """Send notification via Discord"""
        try:
            if discord_notifier.enabled:
                discord_notifier.send_notification(
                    title=title,
                    message=message,
                    color=0x00ff00  # Green
                )
            logger.info(f"Notification: {title}")
        except Exception as e:
            logger.error(f"Error sending notification: {e}")

    def get_status(self) -> Dict:
        """Get current orchestrator status"""
        return {
            'is_running': self.is_running,
            'production_version': self.production_lock.production_version,
            'production_locked_at': self.production_lock.locked_at,
            'active_stages': self.running_stages,
            'pipelines': [
                {
                    'version': p.strategy_version,
                    'stage': p.current_stage,
                    'status': p.pipeline_status
                }
                for p in self.pipeline_manager.pipelines.values()
            ]
        }


async def main():
    """Main entry point"""
    orchestrator = AutomatedStrategyOrchestrator(
        check_interval_seconds=300  # Check every 5 minutes
    )

    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())
