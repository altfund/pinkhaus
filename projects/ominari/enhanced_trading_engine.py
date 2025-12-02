#!/usr/bin/env python3
"""
Enhanced Trading Engine with Strategy Versioning and A/B Testing
Provides unified interface for strategy-aware trading with version control
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Set up environment
os.environ['PG_PORT'] = '5999'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from notifications.discord_notifier import discord_notifier
from strategy_versions import StrategyRegistry, StrategyVersion
from ab_testing_framework import ABTestManager, ABTest
from paper_trading_sessions import PaperTradingSessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TradingOpportunity:
    """Represents a betting opportunity with all necessary context"""
    market: Market
    outcome: str
    odds: float
    fair_prob: float
    edge: float
    strategy_version: str
    metadata: Dict[str, Any]


class EnhancedTradingEngine:
    """
    Trading engine that supports:
    - Strategy versioning
    - A/B testing
    - Dynamic strategy switching
    - Performance tracking per version
    """

    def __init__(self,
                 strategy_registry: Optional[StrategyRegistry] = None,
                 ab_test_manager: Optional[ABTestManager] = None,
                 enable_ab_testing: bool = False):

        self.strategy_registry = strategy_registry or StrategyRegistry()
        self.ab_test_manager = ab_test_manager or ABTestManager(strategy_registry=self.strategy_registry)
        self.enable_ab_testing = enable_ab_testing

        self.bankroll_config = BankrollConfig()
        self.session_manager = PaperTradingSessionManager()
        self.is_running = False
        self.session_id = None

        # Get active strategy version
        self.active_strategy = self.strategy_registry.get_active_version()
        if not self.active_strategy:
            raise ValueError("No active strategy version found in registry")

        logger.info(f"Initialized with strategy: {self.active_strategy.version} - {self.active_strategy.name}")

        # A/B testing state
        self.active_test: Optional[ABTest] = None
        if self.enable_ab_testing:
            active_tests = self.ab_test_manager.get_active_tests()
            if active_tests:
                self.active_test = active_tests[0]
                logger.info(f"A/B testing enabled: {self.active_test.test_id}")
                logger.info(f"  Control: {self.active_test.control_version} ({self.active_test.control_allocation_pct}%)")
                logger.info(f"  Variant: {self.active_test.variant_version} ({self.active_test.variant_allocation_pct}%)")

    def get_strategy_for_trade(self) -> StrategyVersion:
        """
        Determine which strategy to use for this trade
        If A/B testing is active, randomly allocate based on test allocation %
        """
        if self.active_test and self.enable_ab_testing:
            if self.active_test.should_allocate_to_variant():
                variant_strategy = self.strategy_registry.get_version(self.active_test.variant_version)
                logger.debug(f"Using variant strategy: {variant_strategy.version}")
                return variant_strategy
            else:
                control_strategy = self.strategy_registry.get_version(self.active_test.control_version)
                logger.debug(f"Using control strategy: {control_strategy.version}")
                return control_strategy
        else:
            # Use active strategy
            return self.active_strategy

    def calculate_kelly_bet(self, prob: float, odds: float, bankroll: float,
                           kelly_fraction: float) -> float:
        """Calculate Kelly bet size"""
        b = odds - 1
        q = 1 - prob
        f = (prob * b - q) / b

        # Apply Kelly fraction
        f = f * kelly_fraction

        if f <= 0:
            return 0

        bet_amount = bankroll * f
        return round(bet_amount, 2)

    def calculate_edge(self, fair_prob: float, market_odds: float) -> float:
        """Calculate betting edge"""
        fair_odds = 1 / fair_prob
        edge = ((market_odds / fair_odds) - 1) * 100
        return edge

    async def find_betting_opportunities(self, strategy: StrategyVersion) -> List[TradingOpportunity]:
        """
        Find betting opportunities using the specified strategy version
        """
        opportunities = []

        try:
            with db_manager.get_db_session() as db:
                # Use strategy's time horizon
                now = datetime.now(timezone.utc)
                future_time = now + timedelta(hours=strategy.time_horizon_hours)

                # Use strategy's market query limit
                markets = db.query(Market).filter(
                    Market.maturity_date > now,
                    Market.maturity_date < future_time,
                    Market.sport.in_(strategy.sports)
                ).limit(strategy.market_query_limit).all()

                logger.info(f"[{strategy.version}] Analyzing {len(markets)} markets " +
                          f"(horizon: {strategy.time_horizon_hours}h, sports: {strategy.sports})")

                for market in markets:
                    # Get odds
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

                        # Apply strategy's minimum edge threshold
                        if edge >= strategy.min_conservative_edge:
                            opportunities.append(TradingOpportunity(
                                market=market,
                                outcome=outcome,
                                odds=odd.decimal_odds,
                                fair_prob=fair_prob,
                                edge=edge,
                                strategy_version=strategy.version,
                                metadata={
                                    'time_horizon': strategy.time_horizon_hours,
                                    'sport': market.sport,
                                    'league': market.league
                                }
                            ))

        except Exception as e:
            logger.error(f"Error finding opportunities: {e}")

        # Sort by edge
        opportunities.sort(key=lambda x: x.edge, reverse=True)
        logger.info(f"[{strategy.version}] Found {len(opportunities)} opportunities (min edge: {strategy.min_conservative_edge}%)")

        return opportunities

    async def place_bet(self, opportunity: TradingOpportunity) -> Optional[Bet]:
        """Place a bet using the specified strategy version"""

        try:
            strategy = self.strategy_registry.get_version(opportunity.strategy_version)
            if not strategy:
                logger.error(f"Strategy {opportunity.strategy_version} not found")
                return None

            bankroll = self.bankroll_config.get_current_bankroll()

            # Calculate bet size using strategy parameters
            bet_amount = self.calculate_kelly_bet(
                opportunity.fair_prob,
                opportunity.odds,
                bankroll,
                strategy.kelly_fraction
            )

            # Apply strategy position limits
            max_position = bankroll * strategy.max_position_pct
            bet_amount = min(bet_amount, max_position)

            # Check if below minimum
            min_bet = 10.0  # Could come from strategy
            if bet_amount < min_bet:
                logger.debug(f"Bet amount ${bet_amount:.2f} below minimum ${min_bet}")
                return None

            # Check portfolio limits
            current_positions = len(self.session_manager.get_current_session().get('positions', {}))
            if current_positions >= strategy.max_open_positions:
                logger.info(f"At max open positions ({current_positions}/{strategy.max_open_positions})")
                return None

            # Get current exposure
            session = self.session_manager.get_current_session()
            current_exposure = sum(
                pos.get('execution_stake', 0)
                for pos in session.get('positions', {}).values()
            )

            # Check total portfolio exposure
            max_exposure = bankroll * strategy.max_portfolio_pct
            if current_exposure + bet_amount > max_exposure:
                logger.info(f"Would exceed max portfolio exposure " +
                          f"({current_exposure + bet_amount:.2f} > {max_exposure:.2f})")
                return None

            # Create bet record
            with db_manager.get_db_session() as db:
                bet = Bet(
                    session_id=self.session_id,
                    source_id=opportunity.market.source_id,
                    unified_market_type='h2h',
                    normalized_outcome=opportunity.outcome,
                    normalized_line=0.0,
                    bet_name=f"{opportunity.market.home_team} vs {opportunity.market.away_team} - {opportunity.outcome}",
                    probability=opportunity.fair_prob,
                    odds=opportunity.odds,
                    stake=bet_amount,
                    execution_stake=bet_amount,
                    fee_amount=0.0,
                    fee_pct=0.0
                )

                db.add(bet)
                db.commit()
                db.refresh(bet)

                logger.info(
                    f"✅ [{strategy.version}] Placed bet: ${bet_amount:.2f} on " +
                    f"{opportunity.market.home_team} vs {opportunity.market.away_team} - " +
                    f"{opportunity.outcome} @ {opportunity.odds:.2f} " +
                    f"(edge: {opportunity.edge:.2f}%)"
                )

                # Update A/B test metrics if applicable
                if self.active_test and self.enable_ab_testing:
                    variant = "variant" if strategy.version == self.active_test.variant_version else "control"
                    # Note: P&L will be updated when bet settles
                    logger.debug(f"Bet assigned to {variant} in A/B test {self.active_test.test_id}")

                # Update strategy performance tracking
                # (Will be updated when bet settles)

                return bet

        except Exception as e:
            logger.error(f"Error placing bet: {e}", exc_info=True)
            return None

    async def settle_bet(self, bet: Bet, actual_result: str):
        """
        Settle a bet and update strategy/test metrics
        """
        try:
            # Calculate P&L
            if bet.normalized_outcome == actual_result:
                # Win
                pnl = bet.execution_stake * (bet.odds - 1)
            else:
                # Loss
                pnl = -bet.execution_stake

            # Update A/B test metrics if applicable
            if self.active_test and self.enable_ab_testing:
                # Determine which variant this bet belongs to
                # (Would need to track this in bet metadata)
                pass

            # Update strategy version performance
            # (Would integrate with StrategyRegistry.update_performance)

            logger.info(f"Settled bet: {bet.bet_name} → {actual_result} (P&L: ${pnl:.2f})")

        except Exception as e:
            logger.error(f"Error settling bet: {e}")

    async def run_trading_loop(self):
        """Main trading loop with strategy versioning"""

        logger.info("🚀 Starting Enhanced Trading Engine")
        logger.info(f"💰 Initial bankroll: ${self.bankroll_config.get_current_bankroll():.2f}")
        logger.info(f"📊 Strategy: {self.active_strategy.version} - {self.active_strategy.name}")

        if self.enable_ab_testing and self.active_test:
            logger.info(f"🧪 A/B Testing Active: {self.active_test.name}")

        # Create trading session
        try:
            with db_manager.get_db_session() as db:
                session = BettingSession(
                    as_of=datetime.now(timezone.utc),
                    session_type='paper',
                    strategy_name=f"{self.active_strategy.version}:{self.active_strategy.name}",
                    kelly_bankroll=self.bankroll_config.get_current_bankroll(),
                    execution_bankroll=self.bankroll_config.get_current_bankroll(),
                    kelly_fraction=self.active_strategy.kelly_fraction,
                    cap_per_game=self.bankroll_config.get_current_bankroll() * 0.1,
                    cap_per_bet=self.bankroll_config.get_current_bankroll() * self.active_strategy.max_position_pct,
                    cap_per_game_market=self.bankroll_config.get_current_bankroll() * 0.05,
                    min_bet_abs=10.0,
                    min_bet_pct=0.001
                )
                db.add(session)
                db.commit()
                self.session_id = session.id
                logger.info(f"Created trading session: {session.id}")
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return

        self.is_running = True

        while self.is_running:
            try:
                # Determine strategy for this cycle
                strategy = self.get_strategy_for_trade()

                # Find opportunities
                opportunities = await self.find_betting_opportunities(strategy)

                if opportunities:
                    logger.info(f"Found {len(opportunities)} opportunities with {strategy.version}")

                    # Calculate how many we can take
                    current_positions = len(self.session_manager.get_current_session().get('positions', {}))
                    max_new_positions = min(
                        strategy.max_open_positions - current_positions,
                        len(opportunities)
                    )

                    if max_new_positions <= 0:
                        logger.info(f"At position limit ({current_positions}/{strategy.max_open_positions})")
                    else:
                        logger.info(f"Processing top {max_new_positions} of {len(opportunities)} opportunities")

                        # Place bets
                        for opp in opportunities[:max_new_positions]:
                            await self.place_bet(opp)
                            await asyncio.sleep(1)

                # Check A/B test status
                if self.active_test and self.enable_ab_testing:
                    # Monitor test
                    status = self.ab_test_manager.monitor_test(self.active_test.test_id)

                    if status['is_complete'] and status['can_analyze']:
                        logger.info("A/B test completed, analyzing results...")
                        analysis = self.ab_test_manager.complete_test(self.active_test.test_id)
                        logger.info(f"Test result: {analysis['result']}")
                        logger.info(f"Recommendation: {analysis.get('recommendation')}")

                        # Optionally promote winner
                        if analysis['result'] == 'variant_wins':
                            logger.info("Variant won! Consider promoting to active.")

                # Log status
                stats = self.bankroll_config.get_performance_stats()
                logger.info(
                    f"📊 Portfolio: ${stats['current_bankroll']:.2f} | " +
                    f"P&L: ${stats['total_pnl']:.2f} | " +
                    f"ROI: {stats['roi']:.2f}%"
                )

                # Wait before next iteration
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Trading loop error: {e}", exc_info=True)
                await asyncio.sleep(30)

    async def stop(self):
        """Stop trading engine"""
        self.is_running = False
        logger.info("Stopping Enhanced Trading Engine...")

        # Final report
        stats = self.bankroll_config.get_performance_stats()
        logger.info("📊 Final Report:")
        logger.info(f"  Strategy: {self.active_strategy.version}")
        logger.info(f"  Final Bankroll: ${stats['current_bankroll']:.2f}")
        logger.info(f"  Total P&L: ${stats['total_pnl']:.2f}")
        logger.info(f"  ROI: {stats['roi']:.2f}%")

        if self.active_test and self.enable_ab_testing:
            logger.info(f"  A/B Test: {self.active_test.name}")
            status = self.ab_test_manager.monitor_test(self.active_test.test_id)
            logger.info(f"    Control trades: {status['control']['trades']}, P&L: ${status['control']['pnl']:.2f}")
            logger.info(f"    Variant trades: {status['variant']['trades']}, P&L: ${status['variant']['pnl']:.2f}")


async def main():
    """Main entry point"""

    # Initialize with strategy versioning
    strategy_registry = StrategyRegistry()
    ab_test_manager = ABTestManager(strategy_registry=strategy_registry)

    # Example: Create an A/B test (optional)
    enable_testing = False  # Set to True to enable A/B testing

    if enable_testing:
        # Check if test already exists
        active_tests = ab_test_manager.get_active_tests()
        if not active_tests:
            # Create a new test
            test = ab_test_manager.create_test(
                name="48h Horizon Expansion Test",
                control_version="1.0.0",
                variant_version="1.1.0",
                description="Testing expanded horizon from 24h to 48h with liquidity checks",
                duration_days=14,
                variant_allocation=20.0,
                created_by="admin"
            )
            logger.info(f"Created A/B test: {test.test_id}")

    # Initialize engine
    engine = EnhancedTradingEngine(
        strategy_registry=strategy_registry,
        ab_test_manager=ab_test_manager,
        enable_ab_testing=enable_testing
    )

    try:
        await engine.run_trading_loop()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal...")
    finally:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
