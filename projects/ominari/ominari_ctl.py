#!/usr/bin/env python3
"""
Ominari Control Tool
Simple CLI for managing production vs testing
"""

import json
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Optional pretty printing
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False

# Add to path
sys.path.insert(0, str(Path(__file__).parent))


class OminariControl:
    """Control tool for Ominari trading system"""

    def __init__(self):
        # Lazy imports
        from strategy_versions import StrategyRegistry
        from strategy_testing_pipeline import PipelineManager
        from ab_testing_framework import ABTestManager

        self.registry = StrategyRegistry()
        self.pipeline_manager = PipelineManager()
        self.ab_manager = ABTestManager(strategy_registry=self.registry)
        self.production_lock_file = Path("production_lock.json")

    def get_production_version(self) -> str:
        """Get current production version"""
        if self.production_lock_file.exists():
            with open(self.production_lock_file, 'r') as f:
                data = json.load(f)
                return data.get('production_version', 'Unknown')
        return "Not Set"

    def cmd_status(self):
        """Show system status"""
        print("\n" + "="*70)
        print("OMINARI TRADING SYSTEM STATUS")
        print("="*70)

        # Production
        production_version = self.get_production_version()
        production_strategy = self.registry.get_version(production_version)

        print(f"\n🎯 PRODUCTION")
        print(f"{'─'*70}")
        print(f"Version: {production_version}")
        if production_strategy:
            print(f"Name: {production_strategy.name}")
            print(f"Sports: {', '.join(production_strategy.sports)}")
            print(f"Horizon: {production_strategy.time_horizon_hours}h")
            print(f"Markets: {production_strategy.market_query_limit}")
            print(f"Min Edge: {production_strategy.min_conservative_edge}%")
            if production_strategy.total_trades > 0:
                print(f"Performance: {production_strategy.total_trades} trades, "
                      f"{production_strategy.win_rate:.1%} win rate, "
                      f"${production_strategy.total_pnl:,.2f} P&L")

        # Testing
        print(f"\n🧪 TESTING")
        print(f"{'─'*70}")

        testing_versions = self.registry.get_testing_versions()
        if testing_versions:
            for version in testing_versions:
                pipeline = self.pipeline_manager.get_pipeline(version.version)

                print(f"\n{version.version}: {version.name}")
                if pipeline:
                    print(f"  Stage: {pipeline.current_stage}")
                    print(f"  Status: {pipeline.pipeline_status}")

                    # Show stage results
                    if pipeline.stage_results:
                        latest_stage = pipeline.current_stage
                        if latest_stage in pipeline.stage_results:
                            result = pipeline.stage_results[latest_stage]
                            if result.total_trades > 0:
                                print(f"  Trades: {result.total_trades}")
                                print(f"  Win Rate: {result.win_rate:.1%}")
                                print(f"  ROI: {result.roi:.1%}")
                else:
                    print(f"  Status: Not yet in pipeline")
        else:
            print("  No strategies currently in testing")

        # Active A/B Tests
        active_tests = self.ab_manager.get_active_tests()
        if active_tests:
            print(f"\n⚖️  ACTIVE A/B TESTS")
            print(f"{'─'*70}")
            for test in active_tests:
                print(f"\n{test.name}")
                print(f"  Control: {test.control_version} ({test.control_allocation_pct}%)")
                print(f"  Variant: {test.variant_version} ({test.variant_allocation_pct}%)")
                print(f"  Duration: {test.planned_duration_days} days")
                print(f"  Control Trades: {test.control_metrics.total_trades}")
                print(f"  Variant Trades: {test.variant_metrics.total_trades}")

        print("\n" + "="*70 + "\n")

    def cmd_list_strategies(self):
        """List all strategies"""
        print("\n" + "="*70)
        print("STRATEGY VERSIONS")
        print("="*70 + "\n")

        versions = self.registry.list_versions()

        if not versions:
            print("No strategies registered")
            return

        table_data = []
        for v in versions:
            table_data.append([
                v['version'],
                v['name'][:30],
                v['status'],
                v['total_trades'],
                v['win_rate'],
                v['total_pnl']
            ])

        if HAS_TABULATE:
            print(tabulate(
                table_data,
                headers=['Version', 'Name', 'Status', 'Trades', 'Win Rate', 'P&L'],
                tablefmt='grid'
            ))
        else:
            # Simple fallback
            print(f"{'Version':<12} {'Name':<30} {'Status':<12} {'Trades':<8} {'Win Rate':<10} {'P&L':<12}")
            print("-" * 90)
            for row in table_data:
                print(f"{row[0]:<12} {row[1]:<30} {row[2]:<12} {row[3]:<8} {row[4]:<10} {row[5]:<12}")
        print()

    def cmd_list_pipelines(self):
        """List testing pipelines"""
        print("\n" + "="*70)
        print("TESTING PIPELINES")
        print("="*70 + "\n")

        pipelines = self.pipeline_manager.list_pipelines()

        if not pipelines:
            print("No active pipelines")
            return

        table_data = []
        for p in pipelines:
            table_data.append([
                p['strategy_version'],
                p['current_stage'],
                p['status'],
                p['progress']
            ])

        if HAS_TABULATE:
            print(tabulate(
                table_data,
                headers=['Version', 'Current Stage', 'Status', 'Progress'],
                tablefmt='grid'
            ))
        else:
            # Simple fallback
            print(f"{'Version':<12} {'Current Stage':<25} {'Status':<15} {'Progress':<10}")
            print("-" * 70)
            for row in table_data:
                print(f"{row[0]:<12} {row[1]:<25} {row[2]:<15} {row[3]:<10}")
        print()

    def cmd_production_info(self):
        """Show detailed production info"""
        production_version = self.get_production_version()
        strategy = self.registry.get_version(production_version)

        print("\n" + "="*70)
        print(f"PRODUCTION: {production_version}")
        print("="*70 + "\n")

        if not strategy:
            print(f"Strategy {production_version} not found in registry")
            return

        # Basic info
        print(f"Name: {strategy.name}")
        print(f"Description: {strategy.description}")
        print(f"Status: {strategy.status}")

        # Parameters
        print(f"\nParameters:")
        print(f"  Sports: {', '.join(strategy.sports)}")
        print(f"  Time Horizon: {strategy.time_horizon_hours} hours")
        print(f"  Market Query Limit: {strategy.market_query_limit}")
        print(f"  Min Conservative Edge: {strategy.min_conservative_edge}%")
        print(f"  Kelly Fraction: {strategy.kelly_fraction}")
        print(f"  Max Position: {strategy.max_position_pct*100}%")
        print(f"  Max Portfolio: {strategy.max_portfolio_pct*100}%")
        print(f"  Max Open Positions: {strategy.max_open_positions}")

        # Features
        print(f"\nFeatures:")
        print(f"  Liquidity Checks: {'Yes' if strategy.enable_liquidity_checks else 'No'}")
        print(f"  Multi-Signal: {'Yes' if strategy.enable_multi_signal else 'No'}")
        print(f"  Dynamic Limits: {'Yes' if strategy.enable_dynamic_limits else 'No'}")
        print(f"  Rebalancing: {'Yes' if strategy.enable_rebalancing else 'No'}")

        # Performance
        if strategy.total_trades > 0:
            print(f"\nPerformance:")
            print(f"  Total Trades: {strategy.total_trades}")
            print(f"  Total P&L: ${strategy.total_pnl:,.2f}")
            print(f"  Win Rate: {strategy.win_rate:.1%}")
            print(f"  Avg Edge: {strategy.avg_edge:.2f}%")

        print()

    def cmd_add_strategy(self, args):
        """Add a new strategy to test"""
        from strategy_versions import StrategyVersion

        strategy = StrategyVersion(
            version=args.version,
            name=args.name,
            description=args.description or f"Testing {args.name}",
            sports=args.sports.split(','),
            time_horizon_hours=args.horizon,
            market_query_limit=args.markets,
            min_conservative_edge=args.min_edge,
            kelly_fraction=args.kelly_fraction,
            max_position_pct=args.max_position / 100,
            max_portfolio_pct=args.max_portfolio / 100,
            max_open_positions=args.max_positions,
            status="testing"
        )

        self.registry.register_version(strategy)

        print(f"\n✅ Registered strategy {args.version} for testing")
        print(f"   Name: {args.name}")
        print(f"   The orchestrator will automatically start testing it within 5 minutes")
        print(f"   Monitor progress with: ominari_ctl status\n")

    def cmd_promote(self, args):
        """Manually promote a strategy (EMERGENCY ONLY)"""
        version = args.version

        # Confirm
        print(f"\n⚠️  WARNING: Manual promotion of {version}")
        print(f"   This bypasses automated testing!")
        print(f"   Only use in emergencies.")
        confirm = input(f"\nType '{version}' to confirm: ")

        if confirm != version:
            print("Cancelled")
            return

        # Check if exists
        strategy = self.registry.get_version(version)
        if not strategy:
            print(f"Error: Strategy {version} not found")
            return

        # Promote
        self.registry.promote_version(version)

        # Update lock
        lock_data = {
            'production_version': version,
            'locked_at': datetime.now(timezone.utc).isoformat(),
            'locked_by': 'manual'
        }
        with open(self.production_lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)

        print(f"\n✅ Promoted {version} to production")
        print(f"   Restart production system to take effect\n")

    def cmd_rollback(self, args):
        """Rollback to a previous version (EMERGENCY)"""
        version = args.version

        # Confirm
        print(f"\n🚨 EMERGENCY ROLLBACK to {version}")
        print(f"   This will revert production to {version}")
        confirm = input(f"\nType 'ROLLBACK {version}' to confirm: ")

        if confirm != f"ROLLBACK {version}":
            print("Cancelled")
            return

        # Check if exists
        strategy = self.registry.get_version(version)
        if not strategy:
            print(f"Error: Strategy {version} not found")
            return

        # Rollback
        self.registry.rollback_to_version(version)

        # Update lock
        lock_data = {
            'production_version': version,
            'locked_at': datetime.now(timezone.utc).isoformat(),
            'locked_by': 'emergency_rollback'
        }
        with open(self.production_lock_file, 'w') as f:
            json.dump(lock_data, f, indent=2)

        print(f"\n✅ Rolled back to {version}")
        print(f"   Restart production system to take effect")
        print(f"   Monitor logs carefully\n")


def main():
    parser = argparse.ArgumentParser(
        description='Ominari Control Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Status command
    subparsers.add_parser('status', help='Show system status')

    # List commands
    subparsers.add_parser('strategies', help='List all strategies')
    subparsers.add_parser('pipelines', help='List testing pipelines')
    subparsers.add_parser('production', help='Show production details')

    # Add strategy command
    add_parser = subparsers.add_parser('add', help='Add new strategy to test')
    add_parser.add_argument('version', help='Version (e.g., 1.2.0)')
    add_parser.add_argument('name', help='Strategy name')
    add_parser.add_argument('--description', help='Description')
    add_parser.add_argument('--sports', default='Soccer', help='Comma-separated sports')
    add_parser.add_argument('--horizon', type=int, default=24, help='Time horizon in hours')
    add_parser.add_argument('--markets', type=int, default=50, help='Market query limit')
    add_parser.add_argument('--min-edge', type=float, default=2.0, help='Min edge %')
    add_parser.add_argument('--kelly-fraction', type=float, default=0.25, help='Kelly fraction')
    add_parser.add_argument('--max-position', type=float, default=2.0, help='Max position %')
    add_parser.add_argument('--max-portfolio', type=float, default=20.0, help='Max portfolio %')
    add_parser.add_argument('--max-positions', type=int, default=50, help='Max open positions')

    # Promote command (emergency)
    promote_parser = subparsers.add_parser('promote', help='Manually promote strategy (EMERGENCY)')
    promote_parser.add_argument('version', help='Version to promote')

    # Rollback command (emergency)
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to version (EMERGENCY)')
    rollback_parser.add_argument('version', help='Version to rollback to')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    ctl = OminariControl()

    # Execute command
    if args.command == 'status':
        ctl.cmd_status()
    elif args.command == 'strategies':
        ctl.cmd_list_strategies()
    elif args.command == 'pipelines':
        ctl.cmd_list_pipelines()
    elif args.command == 'production':
        ctl.cmd_production_info()
    elif args.command == 'add':
        ctl.cmd_add_strategy(args)
    elif args.command == 'promote':
        ctl.cmd_promote(args)
    elif args.command == 'rollback':
        ctl.cmd_rollback(args)


if __name__ == "__main__":
    main()
