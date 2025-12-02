#!/usr/bin/env python3
"""
Trading Configuration Profiles
Pre-configured risk and strategy profiles for different trading objectives
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_versions import StrategyVersion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Risk profile levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXPERIMENTAL = "experimental"


@dataclass
class TradingProfile:
    """Complete trading profile with strategy and risk settings"""

    profile_name: str
    risk_level: str  # conservative, moderate, aggressive, experimental
    description: str

    # Strategy parameters
    sports: List[str]
    time_horizon_hours: int
    market_query_limit: int
    min_conservative_edge: float

    # Position sizing
    kelly_fraction: float
    max_position_pct: float  # As decimal (0.02 = 2%)
    max_portfolio_pct: float  # As decimal (0.20 = 20%)
    max_open_positions: int

    # Feature flags
    enable_liquidity_checks: bool
    enable_multi_signal: bool
    enable_dynamic_limits: bool
    enable_rebalancing: bool

    # Advanced settings
    signal_types: List[str]
    cost_model: str  # "realistic" or "simple"

    # Safety limits (circuit breakers)
    max_daily_loss_pct: float = 5.0
    max_drawdown_from_peak_pct: float = 15.0
    max_consecutive_losses: int = 10
    require_daily_confirmation: bool = True

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'TradingProfile':
        """Create from dictionary"""
        return cls(**data)

    def to_strategy_version(self, version: str, name: str) -> StrategyVersion:
        """Convert profile to StrategyVersion"""
        return StrategyVersion(
            version=version,
            name=name,
            description=self.description,
            sports=self.sports,
            time_horizon_hours=self.time_horizon_hours,
            market_query_limit=self.market_query_limit,
            min_conservative_edge=self.min_conservative_edge,
            kelly_fraction=self.kelly_fraction,
            max_position_pct=self.max_position_pct,
            max_portfolio_pct=self.max_portfolio_pct,
            max_open_positions=self.max_open_positions,
            enable_liquidity_checks=self.enable_liquidity_checks,
            enable_multi_signal=self.enable_multi_signal,
            enable_dynamic_limits=self.enable_dynamic_limits,
            enable_rebalancing=self.enable_rebalancing,
            signal_types=self.signal_types,
            cost_model=self.cost_model
        )


class TradingProfileManager:
    """Manages trading configuration profiles"""

    def __init__(self, profiles_file: str = "config/trading_profiles.json"):
        self.profiles_file = Path(profiles_file)
        self.profiles: Dict[str, TradingProfile] = {}
        self._initialize_default_profiles()
        self._load_profiles()

    def _initialize_default_profiles(self):
        """Create default profiles if they don't exist"""

        # CONSERVATIVE: Production-ready, proven strategy
        self.profiles['conservative'] = TradingProfile(
            profile_name='conservative',
            risk_level=RiskProfile.CONSERVATIVE.value,
            description='Conservative production strategy - Soccer only, 24h horizon, proven edge',
            sports=['Soccer'],
            time_horizon_hours=24,
            market_query_limit=20,
            min_conservative_edge=2.0,  # Only bet when we have 2%+ edge
            kelly_fraction=0.25,  # Very conservative Kelly
            max_position_pct=0.02,  # 2% max per position
            max_portfolio_pct=0.20,  # 20% max total exposure
            max_open_positions=50,
            enable_liquidity_checks=False,  # Not required for conservative
            enable_multi_signal=False,  # Single signal only
            enable_dynamic_limits=True,
            enable_rebalancing=True,
            signal_types=['intrinsic'],
            cost_model='realistic',
            max_daily_loss_pct=5.0,
            max_drawdown_from_peak_pct=15.0,
            max_consecutive_losses=10,
            require_daily_confirmation=True
        )

        # MODERATE: Expanded but still safe
        self.profiles['moderate'] = TradingProfile(
            profile_name='moderate',
            risk_level=RiskProfile.MODERATE.value,
            description='Moderate strategy - Soccer 48h horizon, liquidity-aware, more markets',
            sports=['Soccer'],
            time_horizon_hours=48,  # Expanded horizon
            market_query_limit=50,  # More markets
            min_conservative_edge=2.0,
            kelly_fraction=0.25,
            max_position_pct=0.02,
            max_portfolio_pct=0.20,
            max_open_positions=50,
            enable_liquidity_checks=True,  # Add liquidity awareness
            enable_multi_signal=False,
            enable_dynamic_limits=True,
            enable_rebalancing=True,
            signal_types=['intrinsic'],
            cost_model='realistic',
            max_daily_loss_pct=7.0,  # Slightly higher tolerance
            max_drawdown_from_peak_pct=20.0,
            max_consecutive_losses=12,
            require_daily_confirmation=True
        )

        # AGGRESSIVE: Multi-sport, longer horizon, higher limits
        self.profiles['aggressive'] = TradingProfile(
            profile_name='aggressive',
            risk_level=RiskProfile.AGGRESSIVE.value,
            description='Aggressive strategy - Multi-sport, 72h horizon, higher exposure',
            sports=['Soccer', 'Tennis', 'Basketball', 'American Football'],
            time_horizon_hours=72,
            market_query_limit=100,
            min_conservative_edge=1.5,  # Lower edge threshold
            kelly_fraction=0.30,  # Higher Kelly fraction
            max_position_pct=0.03,  # 3% per position
            max_portfolio_pct=0.30,  # 30% total exposure
            max_open_positions=75,
            enable_liquidity_checks=True,
            enable_multi_signal=False,  # Still single signal for safety
            enable_dynamic_limits=True,
            enable_rebalancing=True,
            signal_types=['intrinsic'],
            cost_model='realistic',
            max_daily_loss_pct=10.0,
            max_drawdown_from_peak_pct=25.0,
            max_consecutive_losses=15,
            require_daily_confirmation=True
        )

        # EXPERIMENTAL: For testing new features
        self.profiles['experimental'] = TradingProfile(
            profile_name='experimental',
            risk_level=RiskProfile.EXPERIMENTAL.value,
            description='Experimental - Multi-signal, all features enabled, higher limits',
            sports=['Soccer', 'Tennis', 'Basketball', 'American Football', 'Baseball', 'Hockey'],
            time_horizon_hours=72,
            market_query_limit=200,
            min_conservative_edge=1.0,  # Lower threshold for testing
            kelly_fraction=0.20,  # Conservative despite being experimental
            max_position_pct=0.015,  # 1.5% - lower due to experimental nature
            max_portfolio_pct=0.15,  # 15% - lower total exposure
            max_open_positions=50,  # Limit number of experimental bets
            enable_liquidity_checks=True,
            enable_multi_signal=True,  # TEST multi-signal
            enable_dynamic_limits=True,
            enable_rebalancing=True,
            signal_types=['intrinsic', 'momentum', 'value'],  # Multiple signals
            cost_model='realistic',
            max_daily_loss_pct=5.0,  # Tight limits for experimental
            max_drawdown_from_peak_pct=10.0,
            max_consecutive_losses=8,
            require_daily_confirmation=True
        )

    def _load_profiles(self):
        """Load custom profiles from disk"""
        if self.profiles_file.exists():
            try:
                with open(self.profiles_file, 'r') as f:
                    data = json.load(f)

                # Merge custom profiles with defaults
                for profile_name, profile_data in data.items():
                    if profile_name not in self.profiles:  # Don't override defaults
                        self.profiles[profile_name] = TradingProfile.from_dict(profile_data)

                logger.info(f"Loaded {len(data)} custom profiles from {self.profiles_file}")
            except Exception as e:
                logger.error(f"Error loading profiles: {e}")
        else:
            logger.info("No custom profiles file found, using defaults only")
            self._save_profiles()  # Create file with defaults

    def _save_profiles(self):
        """Save profiles to disk"""
        try:
            self.profiles_file.parent.mkdir(parents=True, exist_ok=True)

            data = {
                name: profile.to_dict()
                for name, profile in self.profiles.items()
            }

            with open(self.profiles_file, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.profiles)} profiles to {self.profiles_file}")
        except Exception as e:
            logger.error(f"Error saving profiles: {e}")

    def get_profile(self, profile_name: str) -> Optional[TradingProfile]:
        """Get a trading profile by name"""
        return self.profiles.get(profile_name)

    def list_profiles(self) -> List[Dict[str, Any]]:
        """List all available profiles"""
        return [
            {
                'name': name,
                'risk_level': profile.risk_level,
                'description': profile.description,
                'sports': profile.sports,
                'horizon': f"{profile.time_horizon_hours}h",
                'markets': profile.market_query_limit,
                'min_edge': f"{profile.min_conservative_edge}%",
                'kelly_fraction': profile.kelly_fraction,
                'max_position': f"{profile.max_position_pct*100:.1f}%",
                'max_exposure': f"{profile.max_portfolio_pct*100:.1f}%"
            }
            for name, profile in sorted(self.profiles.items())
        ]

    def create_custom_profile(self, profile_name: str, base_profile: str,
                            **overrides) -> TradingProfile:
        """Create custom profile based on existing profile with overrides"""

        base = self.profiles.get(base_profile)
        if not base:
            raise ValueError(f"Base profile '{base_profile}' not found")

        # Start with base profile data
        profile_data = base.to_dict()

        # Apply overrides
        profile_data['profile_name'] = profile_name
        for key, value in overrides.items():
            if key in profile_data:
                profile_data[key] = value
            else:
                logger.warning(f"Unknown parameter: {key}")

        # Create new profile
        custom_profile = TradingProfile.from_dict(profile_data)
        self.profiles[profile_name] = custom_profile
        self._save_profiles()

        logger.info(f"Created custom profile: {profile_name} (based on {base_profile})")
        return custom_profile

    def compare_profiles(self, profile1: str, profile2: str) -> Dict[str, Any]:
        """Compare two profiles side by side"""

        p1 = self.profiles.get(profile1)
        p2 = self.profiles.get(profile2)

        if not p1 or not p2:
            raise ValueError(f"One or both profiles not found: {profile1}, {profile2}")

        differences = []

        # Check key parameters
        params_to_compare = [
            'sports', 'time_horizon_hours', 'market_query_limit',
            'min_conservative_edge', 'kelly_fraction', 'max_position_pct',
            'max_portfolio_pct', 'max_open_positions'
        ]

        for param in params_to_compare:
            v1 = getattr(p1, param)
            v2 = getattr(p2, param)
            if v1 != v2:
                differences.append({
                    'parameter': param,
                    'profile1': v1,
                    'profile2': v2
                })

        return {
            'profile1': profile1,
            'profile2': profile2,
            'risk_level_1': p1.risk_level,
            'risk_level_2': p2.risk_level,
            'differences': differences,
            'total_differences': len(differences)
        }

    def validate_profile(self, profile_name: str) -> tuple[bool, List[str]]:
        """Validate profile settings"""

        profile = self.profiles.get(profile_name)
        if not profile:
            return False, [f"Profile '{profile_name}' not found"]

        errors = []

        # Validate ranges
        if not 0 < profile.kelly_fraction <= 1.0:
            errors.append(f"Kelly fraction must be between 0 and 1.0, got {profile.kelly_fraction}")

        if not 0 < profile.max_position_pct <= 0.10:
            errors.append(f"Max position % should be <= 10%, got {profile.max_position_pct*100:.1f}%")

        if not 0 < profile.max_portfolio_pct <= 1.0:
            errors.append(f"Max portfolio % must be <= 100%, got {profile.max_portfolio_pct*100:.1f}%")

        if profile.max_position_pct > profile.max_portfolio_pct:
            errors.append("Max position % cannot exceed max portfolio %")

        if profile.time_horizon_hours < 1:
            errors.append(f"Time horizon must be at least 1 hour, got {profile.time_horizon_hours}")

        if profile.market_query_limit < 1:
            errors.append(f"Market query limit must be at least 1, got {profile.market_query_limit}")

        if profile.min_conservative_edge < 0:
            errors.append(f"Min edge cannot be negative, got {profile.min_conservative_edge}%")

        if not profile.sports:
            errors.append("At least one sport must be specified")

        if errors:
            return False, errors
        else:
            return True, []


if __name__ == "__main__":
    # Example usage
    manager = TradingProfileManager()

    print("\n=== Available Trading Profiles ===\n")
    for profile in manager.list_profiles():
        print(f"{profile['name'].upper()} ({profile['risk_level']})")
        print(f"  {profile['description']}")
        print(f"  Sports: {', '.join(profile['sports'])}")
        print(f"  Horizon: {profile['horizon']} | Markets: {profile['markets']} | Min Edge: {profile['min_edge']}")
        print(f"  Kelly: {profile['kelly_fraction']} | Max Position: {profile['max_position']} | Max Exposure: {profile['max_exposure']}")
        print()

    # Compare profiles
    print("\n=== Comparing Conservative vs Aggressive ===\n")
    comparison = manager.compare_profiles('conservative', 'aggressive')
    print(f"Risk Levels: {comparison['risk_level_1']} vs {comparison['risk_level_2']}")
    print(f"Total Differences: {comparison['total_differences']}")
    print("\nKey Differences:")
    for diff in comparison['differences']:
        print(f"  {diff['parameter']}: {diff['profile1']} → {diff['profile2']}")

    # Create custom profile
    print("\n=== Creating Custom Profile ===\n")
    custom = manager.create_custom_profile(
        profile_name='custom_moderate_plus',
        base_profile='moderate',
        time_horizon_hours=60,  # Between moderate and aggressive
        market_query_limit=75,
        description='Custom moderate-plus profile with 60h horizon'
    )
    print(f"Created: {custom.profile_name}")
    print(f"  Horizon: {custom.time_horizon_hours}h")
    print(f"  Markets: {custom.market_query_limit}")

    # Validate profiles
    print("\n=== Validating Profiles ===\n")
    for profile_name in ['conservative', 'moderate', 'aggressive', 'experimental']:
        valid, errors = manager.validate_profile(profile_name)
        status = "✓ VALID" if valid else "✗ INVALID"
        print(f"{profile_name}: {status}")
        if errors:
            for error in errors:
                print(f"    - {error}")

    # Convert profile to strategy version
    print("\n=== Converting Profile to Strategy Version ===\n")
    conservative = manager.get_profile('conservative')
    strategy = conservative.to_strategy_version("1.0.0", "Production Conservative")
    print(f"Strategy Version: {strategy.version}")
    print(f"  Name: {strategy.name}")
    print(f"  Sports: {strategy.sports}")
    print(f"  Time Horizon: {strategy.time_horizon_hours}h")
