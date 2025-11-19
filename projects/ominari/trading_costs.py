#!/usr/bin/env python3
"""
Trading Costs Module
Implements realistic fees, slippage, and gas costs for paper trading.
"""

import random
import math
from dataclasses import dataclass
from typing import Dict, Any, Tuple
from datetime import datetime, timezone


@dataclass
class TradingCosts:
    """Comprehensive trading cost breakdown."""
    platform_fee: float = 0.0        # Platform commission (e.g., 2%)
    gas_fee: float = 0.0              # Blockchain gas fee (fixed USD)
    slippage_cost: float = 0.0        # Price slippage cost
    timing_fee: float = 0.0           # Time-to-event premium
    total_fee: float = 0.0            # Total cost in USD
    effective_odds: float = 0.0       # Odds after slippage
    execution_stake: float = 0.0      # Stake including all fees


class RealisticTradingCostCalculator:
    """Calculate realistic trading costs for sports betting."""
    
    def __init__(self):
        # Platform fee structures (based on Overtime exchange specifics)
        self.platform_fee_rate = 0.02   # 2% platform fee (lower than generic)
        self.gas_fee_base = 1.50         # Lower base gas on Arbitrum (L2)
        self.gas_fee_per_100 = 0.10      # Reduced per-$100 gas cost
        
        # Slippage parameters (Overtime AMM characteristics)
        self.base_slippage_bps = 8       # 8 bps base slippage (tighter AMM)
        self.size_impact_factor = 0.0008 # 0.08% per $100 (better depth)
        self.liquidity_bonus = 0.7       # 30% better execution for popular markets
        
        # Overtime-specific timing costs
        self.time_premium_factors = {
            'immediate': 0.008,    # <2 hours: 0.8% extra (worse fills close to game)
            'short': 0.003,       # 2-12 hours: 0.3% extra cost  
            'medium': 0.001,      # 12-48 hours: 0.1% extra cost
            'long': 0.0           # >48 hours: no premium
        }
        
        # Market-specific adjustments for Overtime
        self.market_adjustments = {
            'soccer': {'liquidity_multiplier': 0.9, 'spread_bonus': 0.95},
            'football': {'liquidity_multiplier': 1.1, 'spread_bonus': 1.0},
            'basketball': {'liquidity_multiplier': 0.85, 'spread_bonus': 0.92},
            'default': {'liquidity_multiplier': 1.0, 'spread_bonus': 1.0}
        }
        
    def calculate_time_category(self, time_to_event_hours: float) -> str:
        """Categorize trade based on time to event (Overtime-specific)."""
        if time_to_event_hours < 2:
            return 'immediate'
        elif time_to_event_hours < 12:
            return 'short'
        elif time_to_event_hours < 48:
            return 'medium'
        else:
            return 'long'
    
    def calculate_slippage(self, stake: float, odds: float, sport: str = 'soccer', market_depth: str = 'medium') -> Tuple[float, float]:
        """
        Calculate price slippage based on stake size and Overtime market conditions.
        
        Returns:
            (slippage_cost, effective_odds)
        """
        # Get sport-specific adjustments
        sport_config = self.market_adjustments.get(sport.lower(), self.market_adjustments['default'])
        
        # Base slippage (improved for Overtime AMM)
        base_slippage = self.base_slippage_bps / 10000  # Convert bps to decimal
        base_slippage *= sport_config['spread_bonus']  # Sport-specific spread improvement
        
        # Size impact with Overtime's better liquidity
        size_impact = (stake / 100) * self.size_impact_factor * sport_config['liquidity_multiplier']
        
        # Market depth adjustment (AMM has more consistent depth)
        depth_multipliers = {'thin': 1.5, 'medium': 1.0, 'thick': 0.6}
        depth_multiplier = depth_multipliers.get(market_depth, 1.0)
        
        # Reduced random component for AMM (more predictable than orderbook)
        random_component = random.uniform(-0.0002, 0.0008)  # Less variance, still biased
        
        # Total slippage rate
        total_slippage = (base_slippage + size_impact) * depth_multiplier + random_component
        total_slippage = max(0, total_slippage)  # Can't be negative
        
        # Calculate effective odds (worse for bettor)
        effective_odds = odds * (1 - total_slippage)
        effective_odds = max(1.01, effective_odds)  # Minimum odds
        
        # Slippage cost
        slippage_cost = stake * total_slippage
        
        return slippage_cost, effective_odds
    
    def calculate_gas_fee(self, stake: float) -> float:
        """Calculate blockchain gas fees."""
        base_fee = self.gas_fee_base
        size_fee = (stake / 100) * self.gas_fee_per_100
        
        # Add some randomness to simulate gas price volatility
        volatility_factor = random.uniform(0.8, 1.4)
        
        total_gas = (base_fee + size_fee) * volatility_factor
        return round(total_gas, 2)
    
    def calculate_timing_premium(self, stake: float, time_to_event_hours: float) -> float:
        """Calculate premium for trades close to event time."""
        time_category = self.calculate_time_category(time_to_event_hours)
        premium_rate = self.time_premium_factors[time_category]
        return stake * premium_rate
    
    def calculate_total_costs(self, trade: Dict[str, Any], 
                            market_conditions: Dict[str, Any] = None) -> TradingCosts:
        """
        Calculate comprehensive trading costs for a trade.
        
        Args:
            trade: Trade dict with stake, odds, maturity_date etc.
            market_conditions: Optional market condition overrides
        """
        stake = trade.get('stake', 0)
        odds = trade.get('odds', 2.0)
        maturity_date = trade.get('maturity_date')
        
        # Default market conditions
        if market_conditions is None:
            market_conditions = {}
        
        market_depth = market_conditions.get('depth', 'medium')
        
        # Calculate time to event
        time_to_event_hours = 24  # Default
        if maturity_date:
            try:
                if isinstance(maturity_date, str):
                    maturity_dt = datetime.fromisoformat(maturity_date.replace('Z', '+00:00'))
                else:
                    maturity_dt = maturity_date
                    
                time_delta = maturity_dt - datetime.now(timezone.utc)
                time_to_event_hours = max(0.1, time_delta.total_seconds() / 3600)
            except:
                time_to_event_hours = 24
        
        # Calculate individual cost components
        platform_fee = stake * self.platform_fee_rate
        gas_fee = self.calculate_gas_fee(stake)
        slippage_cost, effective_odds = self.calculate_slippage(stake, odds, market_depth)
        timing_fee = self.calculate_timing_premium(stake, time_to_event_hours)
        
        # Total costs
        total_fee = platform_fee + gas_fee + slippage_cost + timing_fee
        execution_stake = stake + total_fee
        
        return TradingCosts(
            platform_fee=platform_fee,
            gas_fee=gas_fee,
            slippage_cost=slippage_cost,
            timing_fee=timing_fee,
            total_fee=total_fee,
            effective_odds=effective_odds,
            execution_stake=execution_stake
        )
    
    def apply_costs_to_trade(self, trade: Dict[str, Any], 
                           market_conditions: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Apply realistic trading costs to a trade and return enhanced trade dict.
        """
        costs = self.calculate_total_costs(trade, market_conditions)
        
        # Create enhanced trade with cost information
        enhanced_trade = trade.copy()
        enhanced_trade.update({
            'original_odds': trade.get('odds'),
            'effective_odds': costs.effective_odds,
            'odds': costs.effective_odds,  # Use slipped odds
            'fee_info': {
                'platform_fee': costs.platform_fee,
                'gas_fee': costs.gas_fee,
                'slippage_cost': costs.slippage_cost,
                'timing_fee': costs.timing_fee,
                'total_fee': costs.total_fee,
                'execution_stake': costs.execution_stake,
                'fee_breakdown': {
                    'platform_fee': f"${costs.platform_fee:.2f} (2.0%)",
                    'gas_fee': f"${costs.gas_fee:.2f}",
                    'slippage': f"${costs.slippage_cost:.2f}",
                    'timing': f"${costs.timing_fee:.2f}"
                }
            }
        })
        
        return enhanced_trade


def test_trading_costs():
    """Test the trading costs calculator."""
    print("💰 Testing Trading Costs Calculator")
    print("=" * 50)
    
    calculator = RealisticTradingCostCalculator()
    
    # Test different trade scenarios
    test_trades = [
        {
            'stake': 100.0,
            'odds': 2.5,
            'maturity_date': '2025-11-19T18:00:00+00:00',
            'scenario': 'Standard bet'
        },
        {
            'stake': 500.0,
            'odds': 1.8,
            'maturity_date': '2025-11-19T12:00:00+00:00',
            'scenario': 'Large bet, short time'
        },
        {
            'stake': 50.0,
            'odds': 4.5,
            'maturity_date': '2025-11-20T20:00:00+00:00',
            'scenario': 'Small bet, long odds'
        }
    ]
    
    for trade in test_trades:
        scenario = trade.pop('scenario')
        print(f"\n📊 {scenario}")
        print(f"   Original: ${trade['stake']:.0f} @ {trade['odds']:.2f} odds")
        
        costs = calculator.calculate_total_costs(trade)
        
        print(f"   Costs Breakdown:")
        print(f"     Platform Fee: ${costs.platform_fee:.2f}")
        print(f"     Gas Fee: ${costs.gas_fee:.2f}")
        print(f"     Slippage: ${costs.slippage_cost:.2f}")
        print(f"     Timing Premium: ${costs.timing_fee:.2f}")
        print(f"     TOTAL COST: ${costs.total_fee:.2f}")
        print(f"   Result: ${costs.execution_stake:.2f} @ {costs.effective_odds:.3f} odds")
        print(f"   Cost Impact: {(costs.total_fee/trade['stake'])*100:.1f}% of stake")


if __name__ == "__main__":
    test_trading_costs()