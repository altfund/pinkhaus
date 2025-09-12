#!/usr/bin/env python3
"""
Integration between Ominari signals and Carver's systematic trading framework.
Converts signal providers into trading rules and applies systematic position sizing.
"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Any
from carver_framework import SystematicFramework, TradingRule
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
from database_v2 import db_manager
from models import Market, Odd
import logging

logger = logging.getLogger(__name__)

class BettingTradingRule(TradingRule):
    """Extends TradingRule for betting-specific forecasting."""
    
    def __init__(self, signal_provider, forecast_scalar: float = 10.0, **kwargs):
        super().__init__(
            name=signal_provider.name,
            forecast_scalar=forecast_scalar,
            **kwargs
        )
        self.signal_provider = signal_provider
        
    def calculate_forecast(self, market_data: pd.DataFrame) -> pd.Series:
        """Convert signal provider probabilities to Carver-style forecasts."""
        try:
            # Get probabilities from signal provider
            probs = self.signal_provider.get_probs(market_data)
            
            # Convert probabilities to forecasts
            # Forecast = (predicted_prob - implied_prob) * scalar
            implied_probs = 1 / market_data['decimal_odds']
            forecast = (probs - implied_probs) * self.forecast_scalar
            
            # Cap at [-20, +20] range per Carver methodology
            forecast = forecast.clip(self.forecast_floor, self.forecast_cap)
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error calculating forecast for {self.name}: {e}")
            return pd.Series(0, index=market_data.index)


class BettingSystematicFramework(SystematicFramework):
    """Systematic framework adapted specifically for sports betting."""
    
    def __init__(self, capital: float = 10000, **kwargs):
        super().__init__(capital=capital, **kwargs)
        self.commission_rate = 0.02  # 2% commission
        self.spread_cost = 0.01     # 1% spread
        
    def prepare_market_data(self, markets: List[Market]) -> Dict[str, pd.DataFrame]:
        """Convert market data to format expected by framework."""
        market_data = {}
        
        for market in markets:
            # Get latest odds for this market
            with db_manager.get_db_session() as db:
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                if not odds:
                    continue
                    
                # Build DataFrame for this market
                odds_df = pd.DataFrame([{
                    'source_id': odd.source_id,
                    'outcome': odd.outcome,
                    'decimal_odds': odd.decimal_odds,
                    'implied_raw': (1/odd.decimal_odds * 100) if odd.decimal_odds > 0 else 0,
                    'updated_at': odd.updated_at,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'league_name': market.league_name,
                    'maturity_date': market.maturity_date,
                    'normalized_outcome': odd.outcome,
                    'time': odd.updated_at
                } for odd in odds])
                
                if not odds_df.empty:
                    # Use updated_at as index
                    odds_df.set_index('updated_at', inplace=True)
                    
                    # Add required columns for Carver framework
                    odds_df['close'] = odds_df['decimal_odds']
                    odds_df['volume'] = 1000  # Placeholder
                    
                    market_data[market.source_id] = odds_df
                    
        return market_data
        
    def calculate_all_forecasts(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Calculate forecasts for all markets using all trading rules."""
        all_forecasts = {}
        
        for market_id, data in market_data.items():
            forecasts_df = pd.DataFrame(index=data.index)
            
            # Apply each trading rule
            for rule_name, rule in self.trading_rules.items():
                if isinstance(rule, BettingTradingRule):
                    forecast = rule.calculate_forecast(data)
                    forecasts_df[rule_name] = forecast
                    
            all_forecasts[market_id] = forecasts_df
            
        return all_forecasts
        
    def calculate_expected_edge(self, 
                               market_data: pd.DataFrame, 
                               forecast: float,
                               odds: float) -> float:
        """Calculate expected edge including trading costs."""
        if odds <= 1.0 or forecast == 0:
            return 0
            
        # Probability implied by forecast (scaled from [-20, 20] to probability)
        signal_prob = 0.5 + (forecast / 40.0)  # Convert to [0, 1] range
        implied_prob = 1 / odds
        
        # Expected value before costs
        raw_edge = signal_prob * (odds - 1) - (1 - signal_prob)
        
        # Subtract trading costs
        total_costs = self.commission_rate + self.spread_cost
        net_edge = raw_edge - total_costs
        
        return net_edge
        
    def evaluate_all_opportunities(self) -> pd.DataFrame:
        """Evaluate all betting opportunities using systematic approach."""
        opportunities = []
        
        with db_manager.get_db_session() as db:
            # Get all active markets
            active_markets = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).limit(200).all()
            
            # Prepare market data
            market_data = self.prepare_market_data(active_markets)
            
            # Calculate forecasts for all markets
            all_forecasts = self.calculate_all_forecasts(market_data)
            
            # Evaluate each opportunity
            for market_id, forecasts_df in all_forecasts.items():
                data = market_data[market_id]
                
                for index, row in forecasts_df.iterrows():
                    for rule_name in forecasts_df.columns:
                        forecast = row[rule_name]
                        
                        if abs(forecast) > 0.1:  # Only consider non-zero forecasts
                            market_row = data.loc[index]
                            odds = market_row['decimal_odds']
                            
                            # Calculate expected edge
                            edge = self.calculate_expected_edge(
                                data.loc[[index]], forecast, odds
                            )
                            
                            opportunities.append({
                                'market_id': market_id,
                                'rule_name': rule_name,
                                'forecast': forecast,
                                'odds': odds,
                                'expected_edge': edge,
                                'signal_strength': abs(forecast) / 20.0,  # Normalize to [0, 1]
                                'outcome': market_row['outcome'],
                                'home_team': market_row.get('home_team', ''),
                                'away_team': market_row.get('away_team', ''),
                                'margin_adjusted': edge > -0.01,  # Small tolerance for costs
                                'recommended_action': 'Back' if forecast > 0 else 'Lay'
                            })
                            
        return pd.DataFrame(opportunities)


def create_betting_systematic_system() -> BettingSystematicFramework:
    """Create systematic framework using existing signal providers."""
    system = BettingSystematicFramework(capital=10000)
    
    # Convert each signal provider to trading rule
    for provider in SIGNAL_PROVIDERS:
        # Get weight for this provider
        weight = SIGNAL_WEIGHTS.get(provider.name, 1.0)
        
        # Create trading rule with appropriate scalar
        rule = BettingTradingRule(
            signal_provider=provider,
            forecast_scalar=weight * 10.0,  # Scale by weight
            turnover=2.0  # Low turnover assumption
        )
        
        system.add_trading_rule(rule)
        
    logger.info(f"Created systematic framework with {len(system.trading_rules)} trading rules")
    return system


def analyze_signal_performance() -> Dict[str, Any]:
    """Analyze current signal performance without cutoffs."""
    system = create_betting_systematic_system()
    opportunities = system.evaluate_all_opportunities()
    
    if opportunities.empty:
        return {
            'total_opportunities': 0,
            'positive_edge_count': 0,
            'negative_edge_count': 0,
            'signal_breakdown': {}
        }
    
    # Analyze by signal provider
    signal_breakdown = {}
    for rule_name in opportunities['rule_name'].unique():
        rule_ops = opportunities[opportunities['rule_name'] == rule_name]
        
        signal_breakdown[rule_name] = {
            'total_opportunities': len(rule_ops),
            'positive_edge': len(rule_ops[rule_ops['expected_edge'] > 0]),
            'negative_edge': len(rule_ops[rule_ops['expected_edge'] <= 0]),
            'avg_edge': rule_ops['expected_edge'].mean(),
            'avg_forecast_strength': rule_ops['signal_strength'].mean(),
            'margin_adjusted_count': len(rule_ops[rule_ops['margin_adjusted']])
        }
    
    return {
        'total_opportunities': len(opportunities),
        'positive_edge_count': len(opportunities[opportunities['expected_edge'] > 0]),
        'negative_edge_count': len(opportunities[opportunities['expected_edge'] <= 0]),
        'margin_adjusted_count': len(opportunities[opportunities['margin_adjusted']]),
        'avg_edge': opportunities['expected_edge'].mean(),
        'signal_breakdown': signal_breakdown,
        'opportunities': opportunities.to_dict('records')[:20]  # Top 20 for display
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Analyzing Signal Performance with Carver Framework ===")
    analysis = analyze_signal_performance()
    
    print(f"\nTotal opportunities found: {analysis['total_opportunities']}")
    print(f"Positive edge count: {analysis['positive_edge_count']}")
    print(f"Negative edge count: {analysis['negative_edge_count']}")
    print(f"Margin-adjusted viable: {analysis['margin_adjusted_count']}")
    print(f"Average edge: {analysis['avg_edge']:.3f}")
    
    print("\n=== Signal Provider Breakdown ===")
    for signal_name, stats in analysis['signal_breakdown'].items():
        print(f"\n{signal_name}:")
        print(f"  Opportunities: {stats['total_opportunities']}")
        print(f"  Positive edge: {stats['positive_edge']}")
        print(f"  Negative edge: {stats['negative_edge']}")
        print(f"  Average edge: {stats['avg_edge']:.3f}")
        print(f"  Signal strength: {stats['avg_forecast_strength']:.2f}")
        print(f"  Viable after costs: {stats['margin_adjusted_count']}")