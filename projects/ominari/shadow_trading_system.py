#!/usr/bin/env python3
"""
Shadow Trading System
Runs strategies in parallel with production to compare performance
without executing real trades.

Logs what trades the shadow strategy WOULD have made alongside
what production ACTUALLY did, enabling side-by-side comparison.
"""

import asyncio
import logging
import os
import sys
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from strategy_versions import StrategyRegistry, StrategyVersion
from paper_trading_sessions import PaperTradingSessionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ShadowTrade:
    """Record of a trade that shadow strategy would have made"""
    timestamp: str
    strategy_version: str
    market_id: str
    market_name: str
    outcome: str
    odds: float
    fair_prob: float
    edge: float
    stake: float
    reason: str  # Why this trade was or wasn't taken
    production_also_took: bool  # Did production also take this trade?

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'ShadowTrade':
        return cls(**data)


@dataclass
class ShadowComparison:
    """Comparison between shadow and production strategies"""
    timestamp: str

    # Opportunity counts
    shadow_opportunities: int
    production_opportunities: int
    overlapping_opportunities: int

    # Trade counts
    shadow_trades: int
    production_trades: int
    both_took: int  # Both strategies took same trade
    shadow_only: int  # Shadow took but production didn't
    production_only: int  # Production took but shadow didn't

    # Divergence reasons
    divergence_reasons: Dict[str, int]  # Count of each reason for divergence

    def to_dict(self) -> Dict:
        return asdict(self)


class ShadowTradingSystem:
    """
    Runs a shadow strategy alongside production
    Tracks what it WOULD do vs what production DOES
    """

    def __init__(self,
                 shadow_strategy_version: str,
                 production_strategy_version: str = "1.0.0",
                 log_dir: str = "shadow_trading_logs"):

        self.registry = StrategyRegistry()
        self.shadow_strategy = self.registry.get_version(shadow_strategy_version)
        self.production_strategy = self.registry.get_version(production_strategy_version)

        if not self.shadow_strategy:
            raise ValueError(f"Shadow strategy {shadow_strategy_version} not found")
        if not self.production_strategy:
            raise ValueError(f"Production strategy {production_strategy_version} not found")

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Session manager to track production trades
        self.session_manager = PaperTradingSessionManager()

        # Shadow trade log
        self.shadow_trades: List[ShadowTrade] = []
        self.comparisons: List[ShadowComparison] = []

        self.is_running = False

        logger.info(f"Shadow Trading: {shadow_strategy_version} vs Production {production_strategy_version}")

    async def start(self):
        """Start shadow trading system"""
        logger.info("Starting Shadow Trading System...")
        self.is_running = True

        while self.is_running:
            try:
                await self._shadow_trading_cycle()
                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Error in shadow trading cycle: {e}")
                await asyncio.sleep(30)

    async def _shadow_trading_cycle(self):
        """Run one cycle of shadow trading"""

        # Get current production state
        production_session = self.session_manager.get_current_session()
        production_positions = production_session.get('positions', {})

        # Find opportunities for shadow strategy
        from enhanced_trading_engine import EnhancedTradingEngine

        # Create temporary engine for opportunity finding
        # (without actually placing trades)
        shadow_opportunities = await self._find_opportunities(self.shadow_strategy)
        production_opportunities = await self._find_opportunities(self.production_strategy)

        # Compare opportunities
        comparison = self._compare_opportunities(shadow_opportunities, production_opportunities)

        # Log shadow trades that would be placed
        for opp in shadow_opportunities:
            # Check if production also took this
            production_also_took = self._check_if_production_took(opp, production_positions)

            # Calculate what shadow would do
            would_take, reason = self._would_shadow_take_trade(opp)

            if would_take:
                shadow_trade = ShadowTrade(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    strategy_version=self.shadow_strategy.version,
                    market_id=opp['market'].source_id,
                    market_name=f"{opp['market'].home_team} vs {opp['market'].away_team}",
                    outcome=opp['outcome'],
                    odds=opp['odds'],
                    fair_prob=opp['fair_prob'],
                    edge=opp['edge'],
                    stake=opp.get('adjusted_bet_size', 0),
                    reason=reason,
                    production_also_took=production_also_took
                )

                self.shadow_trades.append(shadow_trade)

                if not production_also_took:
                    logger.info(f"DIVERGENCE: Shadow would take {shadow_trade.market_name} - {reason}")

        # Save comparison
        self.comparisons.append(comparison)
        self._save_logs()

    async def _find_opportunities(self, strategy: StrategyVersion) -> List[Dict]:
        """Find opportunities for a given strategy"""
        from datetime import timedelta
        from models import Market, Odd
        from sqlalchemy import and_

        opportunities = []

        with db_manager.get_db_session() as db:
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=strategy.time_horizon_hours)

            markets = db.query(Market).filter(
                and_(
                    Market.maturity_date > now,
                    Market.maturity_date < future_time,
                    Market.sport.in_(strategy.sports)
                )
            ).limit(strategy.market_query_limit).all()

            for market in markets:
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
                    edge = ((odd.decimal_odds / (1/fair_prob)) - 1) * 100

                    if edge >= strategy.min_conservative_edge:
                        opportunities.append({
                            'market': market,
                            'outcome': outcome,
                            'odds': odd.decimal_odds,
                            'fair_prob': fair_prob,
                            'edge': edge
                        })

        return opportunities

    def _would_shadow_take_trade(self, opportunity: Dict) -> tuple[bool, str]:
        """Determine if shadow strategy would take this trade"""

        # Check edge threshold
        if opportunity['edge'] < self.shadow_strategy.min_conservative_edge:
            return False, f"Edge too low: {opportunity['edge']:.2f}%"

        # Check position limits
        # (simplified - in real system would check actual exposure)
        current_positions = len(self.shadow_trades)
        if current_positions >= self.shadow_strategy.max_open_positions:
            return False, f"At position limit: {current_positions}/{self.shadow_strategy.max_open_positions}"

        return True, f"Edge: {opportunity['edge']:.2f}%, meets criteria"

    def _check_if_production_took(self, opportunity: Dict, production_positions: Dict) -> bool:
        """Check if production also took this trade"""

        market_id = opportunity['market'].source_id
        outcome = opportunity['outcome']

        # Check if production has this position
        for pos_id, pos in production_positions.items():
            if (pos.get('source_id') == market_id and
                pos.get('normalized_outcome') == outcome):
                return True

        return False

    def _compare_opportunities(self,
                             shadow_opps: List[Dict],
                             production_opps: List[Dict]) -> ShadowComparison:
        """Compare opportunities found by shadow vs production"""

        # Create sets for comparison
        shadow_set = set((o['market'].source_id, o['outcome']) for o in shadow_opps)
        production_set = set((o['market'].source_id, o['outcome']) for o in production_opps)

        overlapping = shadow_set & production_set
        shadow_only = shadow_set - production_set
        production_only = production_set - shadow_set

        comparison = ShadowComparison(
            timestamp=datetime.now(timezone.utc).isoformat(),
            shadow_opportunities=len(shadow_opps),
            production_opportunities=len(production_opps),
            overlapping_opportunities=len(overlapping),
            shadow_trades=len(shadow_opps),  # Simplified
            production_trades=len(production_opps),  # Simplified
            both_took=len(overlapping),
            shadow_only=len(shadow_only),
            production_only=len(production_only),
            divergence_reasons=self._categorize_divergences(shadow_opps, production_opps)
        )

        return comparison

    def _categorize_divergences(self,
                               shadow_opps: List[Dict],
                               production_opps: List[Dict]) -> Dict[str, int]:
        """Categorize reasons for divergences"""

        reasons = {
            'time_horizon_difference': 0,
            'edge_threshold_difference': 0,
            'market_limit_difference': 0,
            'sport_difference': 0
        }

        # Shadow has longer time horizon?
        if self.shadow_strategy.time_horizon_hours > self.production_strategy.time_horizon_hours:
            # Count markets only shadow sees (beyond production horizon)
            reasons['time_horizon_difference'] = len(shadow_opps) - len(production_opps)

        # Different edge thresholds?
        if self.shadow_strategy.min_conservative_edge != self.production_strategy.min_conservative_edge:
            reasons['edge_threshold_difference'] = abs(len(shadow_opps) - len(production_opps))

        return reasons

    def _save_logs(self):
        """Save shadow trading logs"""

        # Save shadow trades
        trades_file = self.log_dir / f"shadow_trades_{self.shadow_strategy.version}.json"
        with open(trades_file, 'w') as f:
            json.dump([t.to_dict() for t in self.shadow_trades], f, indent=2)

        # Save comparisons
        comparisons_file = self.log_dir / f"shadow_comparisons_{self.shadow_strategy.version}.json"
        with open(comparisons_file, 'w') as f:
            json.dump([c.to_dict() for c in self.comparisons], f, indent=2)

        logger.debug(f"Saved shadow logs: {len(self.shadow_trades)} trades, {len(self.comparisons)} comparisons")

    def generate_comparison_report(self) -> str:
        """Generate human-readable comparison report"""

        if not self.comparisons:
            return "No comparison data yet"

        # Aggregate statistics
        total_shadow_opportunities = sum(c.shadow_opportunities for c in self.comparisons)
        total_production_opportunities = sum(c.production_opportunities for c in self.comparisons)
        total_overlapping = sum(c.overlapping_opportunities for c in self.comparisons)

        total_shadow_only = sum(c.shadow_only for c in self.comparisons)
        total_production_only = sum(c.production_only for c in self.comparisons)

        report = []
        report.append("\n" + "="*60)
        report.append("SHADOW TRADING COMPARISON REPORT")
        report.append("="*60)

        report.append(f"\nShadow Strategy: {self.shadow_strategy.version} - {self.shadow_strategy.name}")
        report.append(f"Production Strategy: {self.production_strategy.version} - {self.production_strategy.name}")

        report.append(f"\n📊 Opportunity Comparison")
        report.append(f"  Shadow Opportunities: {total_shadow_opportunities}")
        report.append(f"  Production Opportunities: {total_production_opportunities}")
        report.append(f"  Overlapping: {total_overlapping}")
        report.append(f"  Shadow Only: {total_shadow_only}")
        report.append(f"  Production Only: {total_production_only}")

        report.append(f"\n🔄 Divergence Analysis")
        divergence_rate = (total_shadow_only + total_production_only) / (total_shadow_opportunities + total_production_opportunities) if (total_shadow_opportunities + total_production_opportunities) > 0 else 0
        report.append(f"  Divergence Rate: {divergence_rate:.1%}")

        # Key differences
        report.append(f"\n📋 Strategy Differences")
        report.append(f"  Time Horizon: {self.production_strategy.time_horizon_hours}h → {self.shadow_strategy.time_horizon_hours}h")
        report.append(f"  Market Limit: {self.production_strategy.market_query_limit} → {self.shadow_strategy.market_query_limit}")
        report.append(f"  Min Edge: {self.production_strategy.min_conservative_edge}% → {self.shadow_strategy.min_conservative_edge}%")
        report.append(f"  Sports: {self.production_strategy.sports} → {self.shadow_strategy.sports}")

        report.append("\n" + "="*60)

        return "\n".join(report)

    async def stop(self):
        """Stop shadow trading system"""
        self.is_running = False
        self._save_logs()

        print(self.generate_comparison_report())

        logger.info("Shadow Trading System stopped")


async def main():
    """Example usage"""

    # Run shadow trading for v1.1.0 vs production v1.0.0
    shadow_system = ShadowTradingSystem(
        shadow_strategy_version="1.1.0",
        production_strategy_version="1.0.0"
    )

    try:
        # Run for a limited time (in production, would run continuously)
        await asyncio.wait_for(shadow_system.start(), timeout=300)  # 5 minutes
    except asyncio.TimeoutError:
        await shadow_system.stop()
    except KeyboardInterrupt:
        await shadow_system.stop()


if __name__ == "__main__":
    asyncio.run(main())
