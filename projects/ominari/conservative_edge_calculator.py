#!/usr/bin/env python3
"""
Conservative Edge Calculator
Implements more realistic edge calculations by adding market efficiency factors,
uncertainty adjustments, and dynamic confidence scoring.
"""

import os
import sys
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intrinsic_probability_models import IntrinsicProbabilityCalculator, IntrinsicProbabilities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ConservativeEdgeMetrics:
    """Conservative edge calculation with uncertainty adjustments."""
    raw_edge: float
    market_efficiency_factor: float
    uncertainty_discount: float
    information_asymmetry: float
    competition_factor: float
    final_conservative_edge: float
    confidence_score: float
    method: str

class ConservativeEdgeCalculator:
    """Calculate more realistic edges with market efficiency and uncertainty factors."""
    
    def __init__(self):
        self.base_calculator = IntrinsicProbabilityCalculator()
        
        # Market efficiency factors (reduce edges by these amounts)
        self.efficiency_factors = {
            'major_league_discount': 0.85,    # Major leagues are 15% more efficient
            'liquidity_discount': 0.90,       # High liquidity markets are 10% more efficient  
            'recency_discount': 0.92,         # Recent odds changes indicate efficient pricing
            'public_attention_discount': 0.88, # High-profile matches have tighter spreads
        }
        
        # Uncertainty multipliers (further reduce edges when uncertain)
        self.uncertainty_factors = {
            'team_strength_unknown': 0.80,    # When team relative strength unclear
            'limited_data': 0.85,             # When historical data is sparse
            'volatile_odds': 0.90,            # When odds have been moving significantly
            'injury_uncertainty': 0.85,       # When key player availability unclear
        }
        
        # Competition factors (more competitors = tighter edges)
        self.competition_adjustments = {
            'mainstream_market': 0.75,        # Many bettors watching
            'arbitrage_monitored': 0.70,      # Professional arbitrage scrutiny
            'algorithmic_competition': 0.65,  # Competing with other algorithms
        }
    
    def assess_market_efficiency(self, market, odds_data: Dict[str, float], 
                               historical_odds: Optional[list] = None) -> float:
        """
        Assess how efficient this specific market is.
        More efficient = lower edges available.
        """
        efficiency_score = 1.0
        
        # Major league adjustment
        league = getattr(market, 'league_name', 'Other')
        if league in ['Premier League', 'La Liga', 'Bundesliga', 'Serie A', 'Champions League']:
            efficiency_score *= self.efficiency_factors['major_league_discount']
        
        # Market depth/liquidity proxy (tight spreads = high liquidity = more efficient)
        if len(odds_data) >= 3:
            implied_probs = [1/odds for odds in odds_data.values() if odds > 0]
            market_margin = (sum(implied_probs) - 1.0) * 100
            
            if market_margin < 3.0:  # Very tight margin
                efficiency_score *= self.efficiency_factors['liquidity_discount']
            elif market_margin < 5.0:  # Moderate margin
                efficiency_score *= 0.95
        
        # Public attention proxy (round odds numbers indicate retail attention)
        round_odds_count = sum(1 for odds in odds_data.values() if abs(odds - round(odds, 1)) < 0.05)
        if round_odds_count >= 2:  # Multiple round odds
            efficiency_score *= self.efficiency_factors['public_attention_discount']
        
        # Recent odds movement (if we had historical odds data)
        if historical_odds and len(historical_odds) > 3:
            # Calculate volatility in odds
            odds_volatility = self._calculate_odds_volatility(historical_odds)
            if odds_volatility > 0.1:  # High volatility indicates uncertainty
                efficiency_score *= 0.95  # Slight discount for volatility
        
        return efficiency_score
    
    def assess_information_uncertainty(self, market, odds_data: Dict[str, float]) -> float:
        """
        Assess how much uncertainty exists about the true probabilities.
        Higher uncertainty = larger discounts to edges.
        """
        uncertainty_score = 1.0
        
        # Team strength assessment uncertainty
        home_odds = odds_data.get('home', odds_data.get('Home', 3.0))
        away_odds = odds_data.get('away', odds_data.get('Away', 3.0))
        
        # Very uneven matches have high uncertainty in exact probabilities
        strength_ratio = max(home_odds, away_odds) / min(home_odds, away_odds)
        if strength_ratio > 3.0:  # Very lopsided
            uncertainty_score *= self.uncertainty_factors['team_strength_unknown']
        elif strength_ratio > 2.0:  # Moderately lopsided  
            uncertainty_score *= 0.90
        
        # Time to match (closer = more information available)
        if hasattr(market, 'maturity_date'):
            time_to_match = market.maturity_date - datetime.now(timezone.utc)
            if time_to_match < timedelta(hours=2):  # Very close to kickoff
                uncertainty_score *= 1.05  # Slightly less uncertain (more info available)
            elif time_to_match > timedelta(days=3):  # Far future
                uncertainty_score *= self.uncertainty_factors['limited_data']
        
        # League familiarity
        league = getattr(market, 'league_name', 'Other')
        if league == 'Other' or 'Cup' in league or 'Youth' in league:
            uncertainty_score *= self.uncertainty_factors['limited_data']
        
        return uncertainty_score
    
    def assess_competition_level(self, market, odds_data: Dict[str, float]) -> float:
        """
        Assess how much competition exists in this market.
        More competition = tighter edges.
        """
        competition_score = 1.0
        
        # Major leagues have more professional competition
        league = getattr(market, 'league_name', 'Other')
        if league in ['Premier League', 'La Liga', 'Champions League']:
            competition_score *= self.competition_adjustments['algorithmic_competition']
        elif league in ['Bundesliga', 'Serie A', 'Ligue 1']:
            competition_score *= self.competition_adjustments['arbitrage_monitored']
        elif league not in ['Other']:
            competition_score *= self.competition_adjustments['mainstream_market']
        
        # Popular bet types (home/away) have more competition than draws
        home_odds = odds_data.get('home', odds_data.get('Home', 3.0))
        if home_odds < 2.5:  # Favorite (popular bet)
            competition_score *= 0.90
        
        return competition_score
    
    def calculate_conservative_edge(self, market, odds_data: Dict[str, float], 
                                  historical_odds: Optional[list] = None) -> Dict[str, ConservativeEdgeMetrics]:
        """
        Calculate conservative edges with multiple discount factors applied.
        """
        # Start with base intrinsic probabilities
        intrinsic_probs = self.base_calculator.calculate_intrinsic_probabilities(market, odds_data)
        base_edges = self.base_calculator.calculate_vig_arbitrage_edge(intrinsic_probs, odds_data)
        
        # Apply conservative adjustments
        market_efficiency = self.assess_market_efficiency(market, odds_data, historical_odds)
        uncertainty_factor = self.assess_information_uncertainty(market, odds_data)
        competition_factor = self.assess_competition_level(market, odds_data)
        
        conservative_edges = {}
        
        for outcome, raw_edge in base_edges.items():
            # Apply all conservative factors
            conservative_edge = raw_edge * market_efficiency * uncertainty_factor * competition_factor
            
            # Additional reality checks
            if abs(conservative_edge) > 15:  # Cap edges at 15%
                conservative_edge = 15 * (1 if conservative_edge > 0 else -1)
            
            if abs(conservative_edge) > 10:  # Strong discount for very high claimed edges
                conservative_edge *= 0.75
            
            # Calculate final confidence score
            confidence = intrinsic_probs.confidence * market_efficiency * uncertainty_factor
            
            conservative_edges[outcome] = ConservativeEdgeMetrics(
                raw_edge=raw_edge,
                market_efficiency_factor=market_efficiency,
                uncertainty_discount=uncertainty_factor,
                information_asymmetry=1.0 - intrinsic_probs.confidence,
                competition_factor=competition_factor,
                final_conservative_edge=conservative_edge,
                confidence_score=confidence,
                method="conservative_multi_factor"
            )
        
        return conservative_edges
    
    def _calculate_odds_volatility(self, historical_odds: list) -> float:
        """Calculate volatility in odds over time."""
        if len(historical_odds) < 2:
            return 0.0
        
        odds_changes = []
        for i in range(1, len(historical_odds)):
            if historical_odds[i-1] > 0 and historical_odds[i] > 0:
                change = abs(historical_odds[i] - historical_odds[i-1]) / historical_odds[i-1]
                odds_changes.append(change)
        
        if not odds_changes:
            return 0.0
        
        # Return average absolute change
        return sum(odds_changes) / len(odds_changes)

def test_conservative_edge_calculation():
    """Test the conservative edge calculator."""
    print("🧪 Testing Conservative Edge Calculator")
    
    calculator = ConservativeEdgeCalculator()
    
    # Create mock market data
    class MockMarket:
        def __init__(self):
            self.league_name = "Premier League"
            self.home_team = "Arsenal"
            self.away_team = "Chelsea"
            self.maturity_date = datetime(2024, 3, 15, 20, 0, tzinfo=timezone.utc)
    
    market = MockMarket()
    
    test_cases = [
        {
            'odds': {'home': 2.45, 'away': 3.10, 'draw': 3.20},
            'scenario': 'Major League: Premier League match'
        },
        {
            'odds': {'home': 1.80, 'away': 4.50, 'draw': 3.80},
            'scenario': 'Heavy Favorite: Low competition expected'
        },
        {
            'odds': {'home': 2.95, 'away': 2.85, 'draw': 3.10},
            'scenario': 'Even Match: High uncertainty'
        }
    ]
    
    for i, test_case in enumerate(test_cases):
        print(f"\n📊 Test {i+1}: {test_case['scenario']}")
        print(f"  Market Odds: {test_case['odds']}")
        
        # Calculate conservative edges
        conservative_results = calculator.calculate_conservative_edge(market, test_case['odds'])
        
        print("\n  Conservative Edge Analysis:")
        for outcome, metrics in conservative_results.items():
            print(f"    {outcome.upper()}:")
            print(f"      Raw Edge: {metrics.raw_edge:+.2f}%")
            print(f"      Market Efficiency: {metrics.market_efficiency_factor:.3f}")
            print(f"      Uncertainty Discount: {metrics.uncertainty_discount:.3f}")
            print(f"      Competition Factor: {metrics.competition_factor:.3f}")
            print(f"      Final Conservative Edge: {metrics.final_conservative_edge:+.2f}%")
            print(f"      Confidence Score: {metrics.confidence_score:.3f}")
        
        # Show the most conservative positive edge
        positive_edges = {k: v for k, v in conservative_results.items() 
                         if v.final_conservative_edge > 0.5}
        if positive_edges:
            best_conservative = max(positive_edges.items(), key=lambda x: x[1].final_conservative_edge)
            reduction = ((best_conservative[1].raw_edge - best_conservative[1].final_conservative_edge) / 
                        best_conservative[1].raw_edge) * 100
            print(f"  Best Conservative Edge: {best_conservative[0]} at {best_conservative[1].final_conservative_edge:+.2f}%")
            print(f"  Edge Reduction: {reduction:.1f}% from raw calculation")
        else:
            print("  No positive conservative edges found")

if __name__ == "__main__":
    test_conservative_edge_calculation()