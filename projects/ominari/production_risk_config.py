#!/usr/bin/env python3
"""
Production Risk Management Configuration for Ominari Trading System.
Implements comprehensive risk limits and safety controls.
"""

import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk tolerance levels."""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


@dataclass
class PositionLimits:
    """Position-level risk limits."""
    max_single_bet_pct: float = 0.02  # Max 2% of bankroll per bet
    max_single_bet_abs: float = 100.0  # Max $100 per bet
    min_bet_abs: float = 1.0  # Min $1 per bet
    min_bet_pct: float = 0.001  # Min 0.1% of bankroll
    max_odds: float = 10.0  # Max decimal odds (avoid longshots)
    min_odds: float = 1.1  # Min decimal odds (avoid heavy favorites)
    min_edge: float = 0.01  # Min 1% edge required
    max_positions_per_match: int = 3  # Max bets per match
    max_correlated_exposure: float = 0.05  # Max 5% on correlated bets


@dataclass
class PortfolioLimits:
    """Portfolio-level risk limits."""
    max_total_exposure_pct: float = 0.25  # Max 25% of bankroll at risk
    max_daily_loss_pct: float = 0.05  # Max 5% daily loss
    max_drawdown_pct: float = 0.15  # Max 15% drawdown
    max_concurrent_positions: int = 50  # Max open positions
    max_leverage: float = 1.0  # No leverage by default
    concentration_limit_pct: float = 0.15  # Max 15% in single sport/league
    correlation_threshold: float = 0.7  # Correlation limit for positions


@dataclass
class TimeLimits:
    """Time-based risk controls."""
    min_time_to_event: int = 300  # Min 5 minutes before event
    max_time_to_event: int = 86400  # Max 24 hours before event
    cool_off_period_loss: int = 3600  # 1 hour cooldown after big loss
    max_bets_per_hour: int = 20  # Rate limiting
    max_bets_per_day: int = 100
    session_timeout_minutes: int = 360  # 6 hour max session
    break_after_consecutive_losses: int = 5  # Force break after 5 losses


@dataclass
class KellyLimits:
    """Kelly criterion controls."""
    kelly_fraction: float = 0.25  # 25% Kelly (conservative)
    max_kelly_fraction: float = 0.5  # Never exceed 50% Kelly
    min_kelly_fraction: float = 0.1  # Minimum 10% Kelly
    kelly_cap_per_game: float = 0.1  # Max 10% per game
    kelly_cap_per_bet: float = 0.05  # Max 5% per bet
    kelly_cap_per_market: float = 0.07  # Max 7% per market type


@dataclass
class SignalLimits:
    """Signal and prediction limits."""
    min_signal_confidence: float = 0.55  # Min 55% confidence
    max_signal_staleness_seconds: int = 300  # Max 5 min old signals
    min_signals_required: int = 1  # Min signals for bet
    max_signal_disagreement: float = 0.2  # Max 20% disagreement
    signal_weight_floor: float = 0.1  # Min 10% weight per signal
    signal_weight_ceiling: float = 0.5  # Max 50% weight per signal


@dataclass
class RiskMonitoring:
    """Risk monitoring configuration."""
    alert_on_drawdown_pct: float = 0.10  # Alert at 10% drawdown
    alert_on_daily_loss_pct: float = 0.03  # Alert at 3% daily loss
    alert_on_correlation_breach: bool = True
    alert_on_concentration_breach: bool = True
    enable_kill_switch: bool = True  # Emergency stop
    kill_switch_loss_pct: float = 0.08  # Stop at 8% loss
    log_all_violations: bool = True
    webhook_url: Optional[str] = None  # For alerts


@dataclass
class ComplianceLimits:
    """Regulatory and compliance limits."""
    max_bankroll: float = 10000.0  # Max $10k bankroll
    kyc_required_above: float = 1000.0  # KYC above $1k
    restricted_markets: List[str] = field(default_factory=list)
    restricted_leagues: List[str] = field(default_factory=list)
    allowed_bookmakers: List[str] = field(default_factory=lambda: ["overtime_markets"])
    geo_restrictions: List[str] = field(default_factory=list)


@dataclass
class ProductionRiskConfig:
    """Complete production risk configuration."""
    risk_level: RiskLevel = RiskLevel.MODERATE
    position_limits: PositionLimits = field(default_factory=PositionLimits)
    portfolio_limits: PortfolioLimits = field(default_factory=PortfolioLimits)
    time_limits: TimeLimits = field(default_factory=TimeLimits)
    kelly_limits: KellyLimits = field(default_factory=KellyLimits)
    signal_limits: SignalLimits = field(default_factory=SignalLimits)
    monitoring: RiskMonitoring = field(default_factory=RiskMonitoring)
    compliance: ComplianceLimits = field(default_factory=ComplianceLimits)
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    environment: str = "production"
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        data = asdict(self)
        # Convert datetime objects to ISO format
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['risk_level'] = self.risk_level.value
        return json.dumps(data, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'ProductionRiskConfig':
        """Deserialize from JSON."""
        data = json.loads(json_str)
        # Convert ISO strings back to datetime
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        data['risk_level'] = RiskLevel(data['risk_level'])
        
        # Convert nested dicts to dataclasses
        data['position_limits'] = PositionLimits(**data['position_limits'])
        data['portfolio_limits'] = PortfolioLimits(**data['portfolio_limits'])
        data['time_limits'] = TimeLimits(**data['time_limits'])
        data['kelly_limits'] = KellyLimits(**data['kelly_limits'])
        data['signal_limits'] = SignalLimits(**data['signal_limits'])
        data['monitoring'] = RiskMonitoring(**data['monitoring'])
        data['compliance'] = ComplianceLimits(**data['compliance'])
        
        return cls(**data)
    
    def validate(self) -> List[str]:
        """Validate configuration consistency."""
        errors = []
        
        # Position limits validation
        if self.position_limits.max_single_bet_pct > self.portfolio_limits.max_total_exposure_pct:
            errors.append("Single bet limit exceeds total exposure limit")
        
        if self.position_limits.min_bet_abs > self.position_limits.max_single_bet_abs:
            errors.append("Minimum bet exceeds maximum bet")
        
        # Kelly limits validation
        if self.kelly_limits.max_kelly_fraction > 1.0:
            errors.append("Kelly fraction cannot exceed 100%")
        
        if self.kelly_limits.kelly_cap_per_bet > self.kelly_limits.kelly_cap_per_game:
            errors.append("Per-bet Kelly cap exceeds per-game cap")
        
        # Time limits validation
        if self.time_limits.min_time_to_event > self.time_limits.max_time_to_event:
            errors.append("Minimum time to event exceeds maximum")
        
        # Monitoring validation
        if self.monitoring.kill_switch_loss_pct < self.monitoring.alert_on_daily_loss_pct:
            errors.append("Kill switch triggers before alert threshold")
        
        return errors


def get_preset_config(risk_level: RiskLevel) -> ProductionRiskConfig:
    """Get preset configuration for risk level."""
    
    if risk_level == RiskLevel.CONSERVATIVE:
        return ProductionRiskConfig(
            risk_level=RiskLevel.CONSERVATIVE,
            position_limits=PositionLimits(
                max_single_bet_pct=0.01,  # 1%
                max_single_bet_abs=50.0,
                min_edge=0.02,  # 2% edge required
                max_odds=5.0
            ),
            portfolio_limits=PortfolioLimits(
                max_total_exposure_pct=0.1,  # 10%
                max_daily_loss_pct=0.02,  # 2%
                max_drawdown_pct=0.08,  # 8%
                max_concurrent_positions=20
            ),
            kelly_limits=KellyLimits(
                kelly_fraction=0.1,  # 10% Kelly
                max_kelly_fraction=0.25
            ),
            signal_limits=SignalLimits(
                min_signal_confidence=0.6,  # 60%
                min_signals_required=2
            )
        )
    
    elif risk_level == RiskLevel.AGGRESSIVE:
        return ProductionRiskConfig(
            risk_level=RiskLevel.AGGRESSIVE,
            position_limits=PositionLimits(
                max_single_bet_pct=0.05,  # 5%
                max_single_bet_abs=500.0,
                min_edge=0.005,  # 0.5% edge
                max_odds=20.0
            ),
            portfolio_limits=PortfolioLimits(
                max_total_exposure_pct=0.5,  # 50%
                max_daily_loss_pct=0.1,  # 10%
                max_drawdown_pct=0.25,  # 25%
                max_concurrent_positions=100
            ),
            kelly_limits=KellyLimits(
                kelly_fraction=0.5,  # 50% Kelly
                max_kelly_fraction=1.0  # Full Kelly
            ),
            signal_limits=SignalLimits(
                min_signal_confidence=0.52,  # 52%
                min_signals_required=1
            )
        )
    
    else:  # MODERATE (default)
        return ProductionRiskConfig(risk_level=RiskLevel.MODERATE)


def save_config(config: ProductionRiskConfig, filepath: str = "risk_config.json"):
    """Save configuration to file."""
    with open(filepath, 'w') as f:
        f.write(config.to_json())
    logger.info(f"Saved risk configuration to {filepath}")


def load_config(filepath: str = "risk_config.json") -> ProductionRiskConfig:
    """Load configuration from file."""
    with open(filepath, 'r') as f:
        config = ProductionRiskConfig.from_json(f.read())
    logger.info(f"Loaded risk configuration from {filepath}")
    return config


def validate_bet_against_limits(
    bet: Dict,
    config: ProductionRiskConfig,
    current_portfolio: Dict
) -> tuple[bool, List[str]]:
    """
    Validate a proposed bet against risk limits.
    
    Returns:
        (is_valid, list_of_violations)
    """
    violations = []
    
    # Position size checks
    if bet['stake'] > config.position_limits.max_single_bet_abs:
        violations.append(f"Stake ${bet['stake']} exceeds max ${config.position_limits.max_single_bet_abs}")
    
    if bet['stake_pct'] > config.position_limits.max_single_bet_pct:
        violations.append(f"Stake {bet['stake_pct']:.1%} exceeds max {config.position_limits.max_single_bet_pct:.1%}")
    
    # Odds checks
    if bet['odds'] > config.position_limits.max_odds:
        violations.append(f"Odds {bet['odds']} exceed max {config.position_limits.max_odds}")
    
    if bet['odds'] < config.position_limits.min_odds:
        violations.append(f"Odds {bet['odds']} below min {config.position_limits.min_odds}")
    
    # Edge check
    if bet.get('edge', 0) < config.position_limits.min_edge:
        violations.append(f"Edge {bet.get('edge', 0):.2%} below min {config.position_limits.min_edge:.2%}")
    
    # Portfolio exposure check
    new_exposure = current_portfolio.get('total_exposure', 0) + bet['stake']
    if new_exposure > current_portfolio['bankroll'] * config.portfolio_limits.max_total_exposure_pct:
        violations.append(f"Would exceed max portfolio exposure")
    
    # Time checks
    time_to_event = (bet['event_time'] - datetime.now(timezone.utc)).total_seconds()
    if time_to_event < config.time_limits.min_time_to_event:
        violations.append(f"Event starts in {time_to_event/60:.0f} min, below min")
    
    return len(violations) == 0, violations


def demonstrate_risk_configs():
    """Demonstrate different risk configurations."""
    print("=== Production Risk Configurations ===\n")
    
    for risk_level in [RiskLevel.CONSERVATIVE, RiskLevel.MODERATE, RiskLevel.AGGRESSIVE]:
        config = get_preset_config(risk_level)
        print(f"\n{risk_level.value.upper()} Configuration:")
        print(f"  Single bet limit: {config.position_limits.max_single_bet_pct:.1%}")
        print(f"  Total exposure: {config.portfolio_limits.max_total_exposure_pct:.1%}")
        print(f"  Kelly fraction: {config.kelly_limits.kelly_fraction:.1%}")
        print(f"  Min confidence: {config.signal_limits.min_signal_confidence:.1%}")
        
        # Validate
        errors = config.validate()
        if errors:
            print(f"  ⚠️  Validation errors: {errors}")
        else:
            print(f"  ✅ Configuration valid")
    
    # Save example config
    config = get_preset_config(RiskLevel.MODERATE)
    save_config(config, "risk_config_moderate.json")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_risk_configs()