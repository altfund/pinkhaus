#!/usr/bin/env python3
"""
Strategy Versioning System
Manages different versions of trading strategies with feature flags and parameters.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyVersion:
    """Represents a specific version of a trading strategy."""

    version: str  # Semantic versioning: "1.0.0"
    name: str  # Human-readable name
    description: str  # What's new/different

    # Core parameters
    sports: List[str]
    time_horizon_hours: int
    market_query_limit: int
    min_conservative_edge: float
    kelly_fraction: float

    # Position limits
    max_position_pct: float
    max_portfolio_pct: float
    max_open_positions: int

    # Feature flags
    enable_liquidity_checks: bool = False
    enable_multi_signal: bool = False
    enable_dynamic_limits: bool = True
    enable_rebalancing: bool = True

    # Advanced features
    signal_types: List[str] = field(default_factory=lambda: ["intrinsic"])
    cost_model: str = "realistic"  # "realistic" or "simple"

    # Metadata
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "active"  # "active", "testing", "retired", "deprecated"
    parent_version: Optional[str] = None  # Version this was derived from

    # Performance tracking
    total_trades: int = 0
    total_pnl: float = 0.0
    win_rate: float = 0.0
    avg_edge: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'StrategyVersion':
        """Create from dictionary."""
        return cls(**data)

    def is_compatible_with(self, other: 'StrategyVersion') -> bool:
        """Check if two versions can be A/B tested together."""
        # Should test only one major change at a time
        changes = 0

        if self.time_horizon_hours != other.time_horizon_hours:
            changes += 1
        if self.market_query_limit != other.market_query_limit:
            changes += 1
        if self.sports != other.sports:
            changes += 1
        if self.enable_multi_signal != other.enable_multi_signal:
            changes += 1

        # Compatible if only 1-2 changes
        return changes <= 2


class StrategyRegistry:
    """Manages strategy versions and their lifecycle."""

    def __init__(self, registry_file: str = "strategy_registry.json"):
        self.registry_file = Path(registry_file)
        self.versions: Dict[str, StrategyVersion] = {}
        self._load_registry()

    def _load_registry(self):
        """Load existing registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, 'r') as f:
                    data = json.load(f)

                for version_str, version_data in data.items():
                    self.versions[version_str] = StrategyVersion.from_dict(version_data)

                logger.info(f"Loaded {len(self.versions)} strategy versions from registry")
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self.versions = {}
        else:
            logger.info("No existing registry found, starting fresh")
            self._create_baseline_version()

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {
                version: strat.to_dict()
                for version, strat in self.versions.items()
            }

            with open(self.registry_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved registry with {len(self.versions)} versions")
        except Exception as e:
            logger.error(f"Error saving registry: {e}")

    def _create_baseline_version(self):
        """Create the baseline v1.0.0 from current production system."""
        baseline = StrategyVersion(
            version="1.0.0",
            name="Production Baseline",
            description="Current production system - Soccer 24h conservative",
            sports=["Soccer"],
            time_horizon_hours=24,
            market_query_limit=20,
            min_conservative_edge=2.0,
            kelly_fraction=0.25,
            max_position_pct=0.02,
            max_portfolio_pct=0.20,
            max_open_positions=50,
            enable_liquidity_checks=False,
            enable_multi_signal=False,
            enable_dynamic_limits=True,
            enable_rebalancing=True,
            signal_types=["intrinsic"],
            cost_model="realistic",
            status="active"
        )

        self.register_version(baseline)
        logger.info("Created baseline version 1.0.0")

    def register_version(self, version: StrategyVersion):
        """Register a new strategy version."""
        if version.version in self.versions:
            logger.warning(f"Version {version.version} already exists, will overwrite")

        self.versions[version.version] = version
        self._save_registry()
        logger.info(f"Registered strategy version {version.version}: {version.name}")

    def get_version(self, version: str) -> Optional[StrategyVersion]:
        """Get a specific version."""
        return self.versions.get(version)

    def get_active_version(self) -> Optional[StrategyVersion]:
        """Get the currently active production version."""
        active_versions = [
            v for v in self.versions.values()
            if v.status == "active"
        ]

        if not active_versions:
            logger.warning("No active version found!")
            return None

        # Return the highest version number
        return max(active_versions, key=lambda v: v.version)

    def get_testing_versions(self) -> List[StrategyVersion]:
        """Get all versions currently in testing."""
        return [
            v for v in self.versions.values()
            if v.status == "testing"
        ]

    def update_performance(self, version: str, metrics: Dict[str, float]):
        """Update performance metrics for a version."""
        if version not in self.versions:
            logger.error(f"Version {version} not found")
            return

        strategy = self.versions[version]
        strategy.total_trades = metrics.get('total_trades', strategy.total_trades)
        strategy.total_pnl = metrics.get('total_pnl', strategy.total_pnl)
        strategy.win_rate = metrics.get('win_rate', strategy.win_rate)
        strategy.avg_edge = metrics.get('avg_edge', strategy.avg_edge)

        self._save_registry()
        logger.info(f"Updated performance for version {version}")

    def compare_versions(self, v1: str, v2: str) -> Dict[str, Any]:
        """Compare two strategy versions."""
        if v1 not in self.versions or v2 not in self.versions:
            raise ValueError(f"One or both versions not found: {v1}, {v2}")

        version1 = self.versions[v1]
        version2 = self.versions[v2]

        comparison = {
            "v1": {
                "version": v1,
                "name": version1.name,
                "performance": {
                    "total_trades": version1.total_trades,
                    "total_pnl": version1.total_pnl,
                    "win_rate": version1.win_rate,
                    "avg_edge": version1.avg_edge
                }
            },
            "v2": {
                "version": v2,
                "name": version2.name,
                "performance": {
                    "total_trades": version2.total_trades,
                    "total_pnl": version2.total_pnl,
                    "win_rate": version2.win_rate,
                    "avg_edge": version2.avg_edge
                }
            },
            "differences": {
                "time_horizon": f"{version1.time_horizon_hours}h → {version2.time_horizon_hours}h",
                "market_limit": f"{version1.market_query_limit} → {version2.market_query_limit}",
                "sports": f"{version1.sports} → {version2.sports}",
                "multi_signal": f"{version1.enable_multi_signal} → {version2.enable_multi_signal}",
            },
            "performance_delta": {
                "pnl_diff": version2.total_pnl - version1.total_pnl,
                "win_rate_diff": version2.win_rate - version1.win_rate,
                "edge_diff": version2.avg_edge - version1.avg_edge
            }
        }

        return comparison

    def promote_version(self, version: str):
        """Promote a testing version to active (production)."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        # Retire current active version
        current_active = self.get_active_version()
        if current_active:
            current_active.status = "retired"
            logger.info(f"Retired version {current_active.version}")

        # Promote new version
        self.versions[version].status = "active"
        self._save_registry()
        logger.info(f"Promoted version {version} to active")

    def rollback_to_version(self, version: str):
        """Rollback to a previous version (emergency)."""
        if version not in self.versions:
            raise ValueError(f"Version {version} not found")

        logger.warning(f"ROLLBACK: Reverting to version {version}")

        # Set all versions to retired except the rollback target
        for v in self.versions.values():
            if v.version == version:
                v.status = "active"
            else:
                v.status = "retired"

        self._save_registry()
        logger.warning(f"Rolled back to version {version}")

    def list_versions(self) -> List[Dict[str, Any]]:
        """List all versions with summary info."""
        return [
            {
                "version": v.version,
                "name": v.name,
                "status": v.status,
                "created_at": v.created_at,
                "win_rate": f"{v.win_rate:.1%}" if v.total_trades > 0 else "N/A",
                "total_trades": v.total_trades,
                "total_pnl": f"${v.total_pnl:,.2f}" if v.total_trades > 0 else "N/A"
            }
            for v in sorted(self.versions.values(), key=lambda x: x.version, reverse=True)
        ]


def create_enhanced_version() -> StrategyVersion:
    """Create v1.1.0 - Enhanced with 48h horizon and 50 markets."""
    return StrategyVersion(
        version="1.1.0",
        name="Enhanced 48h",
        description="Expanded to 48h horizon and 50 markets with liquidity checks",
        sports=["Soccer"],
        time_horizon_hours=48,  # Expanded from 24h
        market_query_limit=50,  # Expanded from 20
        min_conservative_edge=2.0,
        kelly_fraction=0.25,
        max_position_pct=0.02,
        max_portfolio_pct=0.20,
        max_open_positions=50,
        enable_liquidity_checks=True,  # NEW
        enable_multi_signal=False,
        enable_dynamic_limits=True,
        enable_rebalancing=True,
        signal_types=["intrinsic"],
        cost_model="realistic",
        status="testing",
        parent_version="1.0.0"
    )


def create_experimental_version() -> StrategyVersion:
    """Create v2.0.0 - Experimental with 72h, 200 markets, multi-signal."""
    return StrategyVersion(
        version="2.0.0",
        name="Experimental Multi-Signal",
        description="72h horizon, 200 markets, multi-signal combination",
        sports=["Soccer"],
        time_horizon_hours=72,  # Expanded from 48h
        market_query_limit=200,  # Expanded from 50
        min_conservative_edge=2.0,
        kelly_fraction=0.25,
        max_position_pct=0.02,
        max_portfolio_pct=0.20,
        max_open_positions=50,
        enable_liquidity_checks=True,
        enable_multi_signal=True,  # NEW
        enable_dynamic_limits=True,
        enable_rebalancing=True,
        signal_types=["intrinsic", "momentum", "value"],  # Multi-signal
        cost_model="realistic",
        status="testing",
        parent_version="1.1.0"
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    registry = StrategyRegistry()

    # Register enhanced version
    enhanced = create_enhanced_version()
    registry.register_version(enhanced)

    # Register experimental version
    experimental = create_experimental_version()
    registry.register_version(experimental)

    # List all versions
    print("\n=== Strategy Registry ===")
    for v in registry.list_versions():
        print(f"{v['version']:8} {v['name']:30} [{v['status']:10}] Trades: {v['total_trades']}")

    # Get active version
    active = registry.get_active_version()
    if active:
        print(f"\nActive Version: {active.version} - {active.name}")
