#!/usr/bin/env python3
"""
Enhanced Kelly System: Combines Ominari's soccer-specific Kelly logic 
with Carver's forecast combination methodology.

Keeps the proven Kelly mutual exclusivity and correlation handling
while adding systematic forecast scaling and combination.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
from carver_framework import ForecastCombiner
import logging

logger = logging.getLogger(__name__)

class EnhancedSignalCombiner:
    """
    Combines multiple signals using Carver's methodology but feeds to Kelly system.
    """
    
    def __init__(self):
        self.forecast_combiner = ForecastCombiner()
        
    def convert_signals_to_forecasts(self, 
                                   market_data: pd.DataFrame,
                                   signal_providers: List) -> pd.DataFrame:
        """
        Convert signal provider probabilities to Carver-style forecasts.
        Forecasts represent predicted edge over fair value.
        """
        forecasts = pd.DataFrame(index=market_data.index)
        
        for provider in signal_providers:
            try:
                # Get probabilities from signal provider
                probs = provider.get_probs(market_data)
                
                # Handle gRPC fallback values (-1.0)
                if isinstance(probs, pd.Series):
                    probs = probs.replace(-1.0, np.nan)
                    
                # Convert to forecasts: difference from implied probability
                implied_probs = 1 / market_data['decimal_odds']
                forecast = (probs - implied_probs) * 20  # Scale to [-20, 20] range
                
                # Handle NaN values
                forecast = forecast.fillna(0.0)
                
                # Cap forecasts
                forecast = forecast.clip(-20, 20)
                
                forecasts[provider.name] = forecast
                
            except Exception as e:
                logger.error(f"Error getting forecast from {provider.name}: {e}")
                forecasts[provider.name] = pd.Series(0, index=market_data.index)
                
        return forecasts
        
    def combine_forecasts(self, 
                         forecasts: pd.DataFrame,
                         weights: Dict[str, float]) -> pd.Series:
        """Combine forecasts using Carver's methodology."""
        if forecasts.empty:
            return pd.Series([], dtype=float)
            
        # Calculate forecast weights and diversification multiplier
        forecast_weights, div_mult = self.forecast_combiner.calculate_forecast_weights(
            forecasts, method='equal'  # Start with equal weights
        )
        
        # Apply user-defined weights if available
        for signal_name, weight in weights.items():
            if signal_name in forecast_weights.index:
                forecast_weights[signal_name] *= weight
                
        # Normalize weights
        forecast_weights = forecast_weights / forecast_weights.sum()
        
        # Combine forecasts
        combined = self.forecast_combiner.combine_forecasts(
            forecasts, forecast_weights, div_mult
        )
        
        return combined
        
    def convert_forecasts_to_probabilities(self, 
                                         forecasts: pd.Series,
                                         market_data: pd.DataFrame) -> pd.Series:
        """Convert combined forecasts back to probabilities for Kelly system."""
        # Convert forecasts back to probability adjustments
        prob_adjustments = forecasts / 20.0  # Scale from [-20, 20] to [-1, 1]
        
        # Start with implied probabilities
        implied_probs = 1 / market_data['decimal_odds']
        
        # Apply adjustments
        adjusted_probs = implied_probs + prob_adjustments
        
        # Ensure probabilities stay in [0, 1] range
        adjusted_probs = adjusted_probs.clip(0.0, 1.0)
        
        return adjusted_probs


class EnhancedKellySystem:
    """
    Enhanced Kelly system that maintains soccer-specific logic
    while using Carver's forecast combination.
    """
    
    def __init__(self, 
                 bankroll: float = 10000,
                 max_stake_per_bet: float = None,
                 min_edge_for_betting: float = 0.0):  # No arbitrary cutoff
        self.bankroll = bankroll
        self.max_stake_per_bet = max_stake_per_bet
        self.min_edge_for_betting = min_edge_for_betting
        self.signal_combiner = EnhancedSignalCombiner()
        
    def evaluate_market_opportunities(self, 
                                    market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate ALL betting opportunities without arbitrary cutoffs.
        Let Kelly system decide what to bet on.
        """
        if market_data.empty:
            return pd.DataFrame()
            
        # Get forecasts from all signal providers
        forecasts = self.signal_combiner.convert_signals_to_forecasts(
            market_data, SIGNAL_PROVIDERS
        )
        
        # Combine forecasts using Carver methodology
        combined_forecast = self.signal_combiner.combine_forecasts(
            forecasts, SIGNAL_WEIGHTS
        )
        
        # Convert back to probabilities for Kelly
        enhanced_probabilities = self.signal_combiner.convert_forecasts_to_probabilities(
            combined_forecast, market_data
        )
        
        # Prepare bets DataFrame for Kelly system
        bets_df = pd.DataFrame({
            'source_id': market_data['source_id'],
            'match_id': market_data.get('match_id', market_data['source_id']),
            'unified_market_type': market_data.get('market_type', 'winner'),
            'normalized_outcome': market_data['normalized_outcome'],
            'normalized_line': market_data.get('normalized_line', 0.0),
            'bet_name': market_data.get('bet_name', market_data['normalized_outcome']),
            'market_name': market_data.get('home_team', '') + ' vs ' + market_data.get('away_team', ''),
            'league_name': market_data.get('league_name', 'Unknown'),
            'bookmaker': 'Overtime',
            'odds': market_data['decimal_odds'],
            'probability': enhanced_probabilities
        })
        
        # Remove any rows with invalid data
        bets_df = bets_df.dropna()
        bets_df = bets_df[bets_df['odds'] > 1.0]
        
        if bets_df.empty:
            return pd.DataFrame()
            
        # Apply Kelly optimization with soccer-specific constraints
        try:
            kelly_results = calculate_kelly_stakes_with_exclusivity(
                bets_df,
                bankroll=self.bankroll,
                correlation_matrix=None,  # Let Kelly calculate optimal correlation
                risk_adjusted=True,
                max_stake_per_bet=self.max_stake_per_bet
            )
            
            # Add edge analysis for all opportunities (not just positive ones)
            kelly_results['implied_prob'] = 1 / kelly_results['odds']
            kelly_results['expected_edge'] = (
                kelly_results['probability'] - kelly_results['implied_prob']
            )
            kelly_results['expected_return'] = (
                kelly_results['probability'] * (kelly_results['odds'] - 1) - 
                (1 - kelly_results['probability'])
            )
            
            # Add forecast information
            kelly_results['combined_forecast'] = combined_forecast
            kelly_results['individual_forecasts'] = forecasts.to_dict('index')
            
            return kelly_results
            
        except Exception as e:
            logger.error(f"Kelly optimization failed: {e}")
            return pd.DataFrame()
            
    def generate_trading_report(self, opportunities: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive trading report."""
        if opportunities.empty:
            return {
                'summary': 'No opportunities found',
                'total_stake': 0,
                'expected_return': 0,
                'signal_analysis': {}
            }
            
        # Overall summary
        total_stake = opportunities['stake'].sum()
        total_expected = (opportunities['expected_return'] * opportunities['stake']).sum()
        
        # Break down by signal contribution
        signal_analysis = {}
        
        # Analyze forecast contributions
        for idx, row in opportunities.iterrows():
            forecasts = row.get('individual_forecasts', {})
            for signal_name, forecast_val in forecasts.items():
                if signal_name not in signal_analysis:
                    signal_analysis[signal_name] = {
                        'total_forecast_sum': 0,
                        'opportunities_count': 0,
                        'stake_allocated': 0,
                        'expected_contribution': 0
                    }
                    
                signal_analysis[signal_name]['total_forecast_sum'] += forecast_val
                signal_analysis[signal_name]['opportunities_count'] += 1
                signal_analysis[signal_name]['stake_allocated'] += row['stake']
                signal_analysis[signal_name]['expected_contribution'] += (
                    row['expected_return'] * row['stake'] * 
                    abs(forecast_val) / 20.0  # Weight by forecast strength
                )
        
        return {
            'summary': f"{len(opportunities)} opportunities, ${total_stake:.2f} total stake",
            'total_stake': total_stake,
            'expected_return': total_expected,
            'expected_roi': (total_expected / total_stake * 100) if total_stake > 0 else 0,
            'opportunities': opportunities[['market_name', 'bet_name', 'odds', 
                                         'probability', 'expected_edge', 'stake']].to_dict('records'),
            'signal_analysis': signal_analysis
        }


def analyze_current_system_with_kelly() -> Dict[str, Any]:
    """
    Analyze current system performance using enhanced Kelly 
    without arbitrary cutoffs.
    """
    from database_v2 import db_manager
    from models import Market, Odd
    
    kelly_system = EnhancedKellySystem()
    
    with db_manager.get_db_session() as db:
        # Get active soccer markets
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).limit(50).all()  # Reasonable limit for analysis
        
        all_opportunities = []
        
        for market in active_markets:
            # Get odds for this market
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            if len(odds) < 3:  # Need home/draw/away
                continue
                
            # Build market data DataFrame
            market_df = pd.DataFrame([{
                'source_id': odd.source_id,
                'market_type': 'winner',
                'normalized_outcome': odd.outcome,
                'normalized_line': 0.0,
                'bet_name': odd.outcome,
                'decimal_odds': odd.decimal_odds,
                'implied_raw': (1/odd.decimal_odds * 100) if odd.decimal_odds > 0 else 0,
                'home_team': market.home_team,
                'away_team': market.away_team,
                'league_name': market.league_name,
                'time': odd.updated_at,
                'maturity_date': market.maturity_date
            } for odd in odds])
            
            # Evaluate opportunities for this market
            market_opps = kelly_system.evaluate_market_opportunities(market_df)
            
            if not market_opps.empty:
                all_opportunities.append(market_opps)
                
        # Combine all opportunities
        if all_opportunities:
            combined_opps = pd.concat(all_opportunities, ignore_index=True)
            report = kelly_system.generate_trading_report(combined_opps)
        else:
            report = kelly_system.generate_trading_report(pd.DataFrame())
            
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=== Enhanced Kelly System Analysis ===")
    print("Preserving soccer-specific Kelly logic with Carver forecast combination")
    print("Removing arbitrary edge cutoffs\n")
    
    analysis = analyze_current_system_with_kelly()
    
    print(f"System Summary: {analysis['summary']}")
    print(f"Expected ROI: {analysis['expected_roi']:.2f}%")
    print(f"Total Expected Return: ${analysis['expected_return']:.2f}")
    
    print("\n=== Signal Provider Analysis ===")
    for signal_name, stats in analysis['signal_analysis'].items():
        avg_forecast = stats['total_forecast_sum'] / max(stats['opportunities_count'], 1)
        print(f"\n{signal_name}:")
        print(f"  Average forecast: {avg_forecast:.2f}")
        print(f"  Opportunities: {stats['opportunities_count']}")
        print(f"  Stake allocated: ${stats['stake_allocated']:.2f}")
        print(f"  Expected contribution: ${stats['expected_contribution']:.2f}")
        
    print("\n=== Top Opportunities (All Edges) ===")
    for i, opp in enumerate(analysis['opportunities'][:10]):
        edge_pct = opp['expected_edge'] * 100
        print(f"{i+1}. {opp['market_name']} - {opp['bet_name']}")
        print(f"    Edge: {edge_pct:+.2f}% | Stake: ${opp['stake']:.2f} | Odds: {opp['odds']:.2f}")