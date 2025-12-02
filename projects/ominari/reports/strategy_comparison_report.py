#!/usr/bin/env python3
"""
Strategy Comparison Report Generator
Creates detailed comparison reports between strategy versions
"""

import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_versions import StrategyRegistry, StrategyVersion
from ab_testing_framework import ABTestManager, ABTest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StrategyComparisonReport:
    """Generates comparison reports between strategy versions"""

    def __init__(self,
                 strategy_registry: Optional[StrategyRegistry] = None,
                 ab_test_manager: Optional[ABTestManager] = None,
                 output_dir: str = "reports/output"):

        self.strategy_registry = strategy_registry or StrategyRegistry()
        self.ab_test_manager = ab_test_manager or ABTestManager(strategy_registry=self.strategy_registry)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def compare_two_strategies(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two strategy versions in detail"""

        v1 = self.strategy_registry.get_version(version1)
        v2 = self.strategy_registry.get_version(version2)

        if not v1 or not v2:
            raise ValueError(f"One or both versions not found: {version1}, {version2}")

        comparison = {
            'version1': version1,
            'version2': version2,
            'v1_name': v1.name,
            'v2_name': v2.name,
            'comparison_date': datetime.now(timezone.utc).isoformat(),

            # Parameter differences
            'parameter_changes': self._compare_parameters(v1, v2),

            # Performance comparison
            'performance': self._compare_performance(v1, v2),

            # Risk comparison
            'risk': self._compare_risk(v1, v2),

            # Feature differences
            'features': self._compare_features(v1, v2),

            # Verdict
            'verdict': self._generate_verdict(v1, v2)
        }

        return comparison

    def _compare_parameters(self, v1: StrategyVersion, v2: StrategyVersion) -> Dict[str, Any]:
        """Compare parameter settings between versions"""

        changes = []

        # Time horizon
        if v1.time_horizon_hours != v2.time_horizon_hours:
            changes.append({
                'parameter': 'time_horizon_hours',
                'v1': v1.time_horizon_hours,
                'v2': v2.time_horizon_hours,
                'change': v2.time_horizon_hours - v1.time_horizon_hours,
                'change_pct': ((v2.time_horizon_hours - v1.time_horizon_hours) / v1.time_horizon_hours * 100)
            })

        # Market query limit
        if v1.market_query_limit != v2.market_query_limit:
            changes.append({
                'parameter': 'market_query_limit',
                'v1': v1.market_query_limit,
                'v2': v2.market_query_limit,
                'change': v2.market_query_limit - v1.market_query_limit,
                'change_pct': ((v2.market_query_limit - v1.market_query_limit) / v1.market_query_limit * 100)
            })

        # Kelly fraction
        if v1.kelly_fraction != v2.kelly_fraction:
            changes.append({
                'parameter': 'kelly_fraction',
                'v1': v1.kelly_fraction,
                'v2': v2.kelly_fraction,
                'change': v2.kelly_fraction - v1.kelly_fraction,
                'change_pct': ((v2.kelly_fraction - v1.kelly_fraction) / v1.kelly_fraction * 100)
            })

        # Min edge
        if v1.min_conservative_edge != v2.min_conservative_edge:
            changes.append({
                'parameter': 'min_conservative_edge',
                'v1': v1.min_conservative_edge,
                'v2': v2.min_conservative_edge,
                'change': v2.min_conservative_edge - v1.min_conservative_edge,
                'change_pct': ((v2.min_conservative_edge - v1.min_conservative_edge) / v1.min_conservative_edge * 100)
            })

        # Sports
        if set(v1.sports) != set(v2.sports):
            changes.append({
                'parameter': 'sports',
                'v1': v1.sports,
                'v2': v2.sports,
                'change': 'Sports list changed',
                'change_pct': None
            })

        return {
            'total_changes': len(changes),
            'changes': changes
        }

    def _compare_performance(self, v1: StrategyVersion, v2: StrategyVersion) -> Dict[str, Any]:
        """Compare performance metrics"""

        # ROI comparison
        roi_v1 = (v1.total_pnl / (v1.total_trades * 100)) * 100 if v1.total_trades > 0 else 0
        roi_v2 = (v2.total_pnl / (v2.total_trades * 100)) * 100 if v2.total_trades > 0 else 0

        return {
            'v1': {
                'total_trades': v1.total_trades,
                'total_pnl': v1.total_pnl,
                'win_rate': v1.win_rate,
                'avg_edge': v1.avg_edge,
                'roi': roi_v1
            },
            'v2': {
                'total_trades': v2.total_trades,
                'total_pnl': v2.total_pnl,
                'win_rate': v2.win_rate,
                'avg_edge': v2.avg_edge,
                'roi': roi_v2
            },
            'differences': {
                'pnl_diff': v2.total_pnl - v1.total_pnl,
                'win_rate_diff': v2.win_rate - v1.win_rate,
                'edge_diff': v2.avg_edge - v1.avg_edge,
                'roi_diff': roi_v2 - roi_v1
            },
            'winner': self._determine_winner(v1, v2)
        }

    def _compare_risk(self, v1: StrategyVersion, v2: StrategyVersion) -> Dict[str, Any]:
        """Compare risk parameters"""

        return {
            'v1': {
                'max_position_pct': v1.max_position_pct,
                'max_portfolio_pct': v1.max_portfolio_pct,
                'max_open_positions': v1.max_open_positions,
                'kelly_fraction': v1.kelly_fraction
            },
            'v2': {
                'max_position_pct': v2.max_position_pct,
                'max_portfolio_pct': v2.max_portfolio_pct,
                'max_open_positions': v2.max_open_positions,
                'kelly_fraction': v2.kelly_fraction
            },
            'risk_assessment': self._assess_risk_changes(v1, v2)
        }

    def _compare_features(self, v1: StrategyVersion, v2: StrategyVersion) -> Dict[str, Any]:
        """Compare feature flags"""

        feature_changes = []

        if v1.enable_liquidity_checks != v2.enable_liquidity_checks:
            feature_changes.append({
                'feature': 'enable_liquidity_checks',
                'v1': v1.enable_liquidity_checks,
                'v2': v2.enable_liquidity_checks
            })

        if v1.enable_multi_signal != v2.enable_multi_signal:
            feature_changes.append({
                'feature': 'enable_multi_signal',
                'v1': v1.enable_multi_signal,
                'v2': v2.enable_multi_signal
            })

        if v1.enable_dynamic_limits != v2.enable_dynamic_limits:
            feature_changes.append({
                'feature': 'enable_dynamic_limits',
                'v1': v1.enable_dynamic_limits,
                'v2': v2.enable_dynamic_limits
            })

        if v1.enable_rebalancing != v2.enable_rebalancing:
            feature_changes.append({
                'feature': 'enable_rebalancing',
                'v1': v1.enable_rebalancing,
                'v2': v2.enable_rebalancing
            })

        if v1.signal_types != v2.signal_types:
            feature_changes.append({
                'feature': 'signal_types',
                'v1': v1.signal_types,
                'v2': v2.signal_types
            })

        return {
            'total_changes': len(feature_changes),
            'changes': feature_changes
        }

    def _determine_winner(self, v1: StrategyVersion, v2: StrategyVersion) -> str:
        """Determine which version performed better"""

        # Need sufficient data
        if v1.total_trades < 50 or v2.total_trades < 50:
            return "insufficient_data"

        # Primary metric: total P&L
        if v2.total_pnl > v1.total_pnl * 1.1:  # 10% better
            return "v2"
        elif v1.total_pnl > v2.total_pnl * 1.1:
            return "v1"

        # Secondary: win rate
        if v2.win_rate > v1.win_rate + 0.05:  # 5% better
            return "v2"
        elif v1.win_rate > v2.win_rate + 0.05:
            return "v1"

        return "tie"

    def _assess_risk_changes(self, v1: StrategyVersion, v2: StrategyVersion) -> str:
        """Assess if risk has increased or decreased"""

        risk_score_v1 = (
            v1.max_position_pct * 100 +
            v1.max_portfolio_pct * 100 +
            v1.kelly_fraction * 100
        )

        risk_score_v2 = (
            v2.max_position_pct * 100 +
            v2.max_portfolio_pct * 100 +
            v2.kelly_fraction * 100
        )

        diff = risk_score_v2 - risk_score_v1

        if diff > 10:
            return "significantly_increased"
        elif diff > 2:
            return "moderately_increased"
        elif diff < -10:
            return "significantly_decreased"
        elif diff < -2:
            return "moderately_decreased"
        else:
            return "unchanged"

    def _generate_verdict(self, v1: StrategyVersion, v2: StrategyVersion) -> Dict[str, Any]:
        """Generate overall verdict and recommendation"""

        winner = self._determine_winner(v1, v2)
        risk_change = self._assess_risk_changes(v1, v2)

        # Build recommendation
        if winner == "insufficient_data":
            recommendation = "Continue testing - insufficient data for conclusive comparison"
            confidence = "low"
        elif winner == "v2":
            if risk_change in ["significantly_increased", "moderately_increased"]:
                recommendation = f"V2 performs better but risk has {risk_change.replace('_', ' ')}. Proceed with caution."
                confidence = "medium"
            else:
                recommendation = "Recommend promoting V2 - better performance with acceptable risk"
                confidence = "high"
        elif winner == "v1":
            recommendation = "Keep V1 - it outperforms V2"
            confidence = "high"
        else:  # tie
            if risk_change in ["significantly_decreased", "moderately_decreased"]:
                recommendation = f"Consider V2 - similar performance with {risk_change.replace('_', ' ')}"
                confidence = "medium"
            else:
                recommendation = "No clear winner - maintain current version or choose based on strategic goals"
                confidence = "low"

        return {
            'winner': winner,
            'risk_change': risk_change,
            'recommendation': recommendation,
            'confidence': confidence
        }

    def generate_markdown_report(self, version1: str, version2: str) -> str:
        """Generate markdown report comparing two versions"""

        comparison = self.compare_two_strategies(version1, version2)

        md = []
        md.append(f"# Strategy Comparison Report\n")
        md.append(f"**Date:** {comparison['comparison_date']}\n")
        md.append(f"**Comparing:** {version1} vs {version2}\n")
        md.append("")

        md.append("## Version Summary\n")
        md.append(f"- **{version1}:** {comparison['v1_name']}")
        md.append(f"- **{version2}:** {comparison['v2_name']}")
        md.append("")

        md.append("## Performance Comparison\n")
        perf = comparison['performance']
        md.append("| Metric | V1 | V2 | Difference |")
        md.append("|--------|----|----|------------|")
        md.append(f"| Total Trades | {perf['v1']['total_trades']} | {perf['v2']['total_trades']} | {perf['v2']['total_trades'] - perf['v1']['total_trades']:+d} |")
        md.append(f"| Total P&L | ${perf['v1']['total_pnl']:.2f} | ${perf['v2']['total_pnl']:.2f} | ${perf['differences']['pnl_diff']:+.2f} |")
        md.append(f"| Win Rate | {perf['v1']['win_rate']:.1%} | {perf['v2']['win_rate']:.1%} | {perf['differences']['win_rate_diff']:+.1%} |")
        md.append(f"| Avg Edge | {perf['v1']['avg_edge']:.2f}% | {perf['v2']['avg_edge']:.2f}% | {perf['differences']['edge_diff']:+.2f}% |")
        md.append(f"| ROI | {perf['v1']['roi']:.2f}% | {perf['v2']['roi']:.2f}% | {perf['differences']['roi_diff']:+.2f}% |")
        md.append("")

        md.append("## Parameter Changes\n")
        params = comparison['parameter_changes']
        if params['total_changes'] > 0:
            md.append("| Parameter | V1 | V2 | Change |")
            md.append("|-----------|----|----|--------|")
            for change in params['changes']:
                v1_val = change['v1']
                v2_val = change['v2']
                change_str = f"{change['change']:+.1f}" if change['change_pct'] else change['change']
                if change['change_pct']:
                    change_str += f" ({change['change_pct']:+.1f}%)"
                md.append(f"| {change['parameter']} | {v1_val} | {v2_val} | {change_str} |")
        else:
            md.append("No parameter changes.")
        md.append("")

        md.append("## Risk Analysis\n")
        risk = comparison['risk']
        md.append(f"**Risk Assessment:** {risk['risk_assessment'].replace('_', ' ').title()}\n")
        md.append("| Parameter | V1 | V2 |")
        md.append("|-----------|----|----|")
        md.append(f"| Max Position % | {risk['v1']['max_position_pct']*100:.1f}% | {risk['v2']['max_position_pct']*100:.1f}% |")
        md.append(f"| Max Portfolio % | {risk['v1']['max_portfolio_pct']*100:.1f}% | {risk['v2']['max_portfolio_pct']*100:.1f}% |")
        md.append(f"| Max Positions | {risk['v1']['max_open_positions']} | {risk['v2']['max_open_positions']} |")
        md.append(f"| Kelly Fraction | {risk['v1']['kelly_fraction']:.2f} | {risk['v2']['kelly_fraction']:.2f} |")
        md.append("")

        md.append("## Feature Changes\n")
        features = comparison['features']
        if features['total_changes'] > 0:
            for change in features['changes']:
                status_v1 = "✓" if change['v1'] else "✗"
                status_v2 = "✓" if change['v2'] else "✗"
                md.append(f"- **{change['feature']}:** {status_v1} → {status_v2}")
        else:
            md.append("No feature changes.")
        md.append("")

        md.append("## Verdict\n")
        verdict = comparison['verdict']
        md.append(f"**Winner:** {verdict['winner'].upper().replace('_', ' ')}")
        md.append(f"**Risk Change:** {verdict['risk_change'].replace('_', ' ').title()}")
        md.append(f"**Confidence:** {verdict['confidence'].upper()}\n")
        md.append(f"**Recommendation:**")
        md.append(f"> {verdict['recommendation']}")
        md.append("")

        return "\n".join(md)

    def generate_all_active_comparisons(self) -> List[str]:
        """Generate comparison reports for all active/testing versions"""

        reports = []

        active = self.strategy_registry.get_active_version()
        testing = self.strategy_registry.get_testing_versions()

        if active and testing:
            for test_version in testing:
                report = self.generate_markdown_report(active.version, test_version.version)

                # Save to file
                filename = f"comparison_{active.version}_vs_{test_version.version}_{datetime.now().strftime('%Y%m%d')}.md"
                filepath = self.output_dir / filename

                with open(filepath, 'w') as f:
                    f.write(report)

                logger.info(f"Generated comparison report: {filepath}")
                reports.append(str(filepath))

        return reports

    def generate_ab_test_report(self, test_id: str) -> str:
        """Generate report for an A/B test"""

        test = self.ab_test_manager.get_test(test_id)
        if not test:
            raise ValueError(f"Test {test_id} not found")

        status = self.ab_test_manager.monitor_test(test_id)

        md = []
        md.append(f"# A/B Test Report\n")
        md.append(f"**Test ID:** {test.test_id}")
        md.append(f"**Test Name:** {test.name}")
        md.append(f"**Description:** {test.description}")
        md.append(f"**Status:** {test.status}")
        md.append(f"**Start Date:** {test.start_date}")
        md.append(f"**Progress:** {status['progress_pct']:.1f}% ({status['elapsed_days']}/{status['planned_days']} days)")
        md.append("")

        md.append("## Test Configuration\n")
        md.append(f"- **Control:** {test.control_version} ({test.control_allocation_pct}% allocation)")
        md.append(f"- **Variant:** {test.variant_version} ({test.variant_allocation_pct}% allocation)")
        md.append(f"- **Minimum Trades Required:** {test.min_trades_required}")
        md.append(f"- **Significance Level:** {test.significance_level}")
        md.append("")

        md.append("## Current Results\n")
        md.append("| Metric | Control | Variant |")
        md.append("|--------|---------|---------|")
        md.append(f"| Trades | {status['control']['trades']} | {status['variant']['trades']} |")
        md.append(f"| Total P&L | ${status['control']['pnl']:.2f} | ${status['variant']['pnl']:.2f} |")
        md.append(f"| Win Rate | {status['control']['win_rate']:.1%} | {status['variant']['win_rate']:.1%} |")
        md.append(f"| Sharpe Ratio | {status['control']['sharpe']:.2f} | {status['variant']['sharpe']:.2f} |")
        md.append("")

        # Analysis if sufficient data
        if status['can_analyze']:
            analysis = self.ab_test_manager.analyze_test(test_id)

            md.append("## Statistical Analysis\n")
            md.append(f"**Result:** {analysis['result'].replace('_', ' ').upper()}")
            if analysis.get('winner'):
                md.append(f"**Winner:** {analysis['winner']}")
            md.append(f"**P-Value:** {analysis.get('p_value', 0):.4f}")
            md.append(f"**Confidence:** {analysis.get('confidence', 0)*100:.1f}%")
            md.append(f"**Effect Size:** {analysis.get('effect_size', 0):.3f}")
            md.append("")

            md.append("## Recommendation\n")
            md.append(f"> {analysis.get('recommendation', 'Continue testing')}")
            md.append("")
        else:
            md.append("## Analysis\n")
            md.append("Insufficient data for statistical analysis. Continue testing.")
            md.append("")

        return "\n".join(md)


if __name__ == "__main__":
    # Example usage
    strategy_registry = StrategyRegistry()
    ab_test_manager = ABTestManager(strategy_registry=strategy_registry)
    report_generator = StrategyComparisonReport(strategy_registry, ab_test_manager)

    print("=== Generating Strategy Comparison Reports ===\n")

    # Compare specific versions
    try:
        report = report_generator.generate_markdown_report("1.0.0", "1.1.0")
        print(report)
        print("\n" + "="*60 + "\n")

        # Save to file
        filename = f"comparison_1.0.0_vs_1.1.0_{datetime.now().strftime('%Y%m%d')}.md"
        filepath = report_generator.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)
        print(f"Report saved to: {filepath}")

    except Exception as e:
        logger.error(f"Error generating report: {e}")

    # Generate all active comparisons
    print("\n=== Generating All Active Comparisons ===\n")
    reports = report_generator.generate_all_active_comparisons()
    print(f"Generated {len(reports)} comparison reports")
