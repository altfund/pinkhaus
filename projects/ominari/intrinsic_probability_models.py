#!/usr/bin/env python3
"""
Intrinsic Probability Models for Edge Calculation
Uses structural market patterns, home/away bias, league data, and statistical models
rather than normalized market odds.
"""

import os
import sys
import logging
from typing import Dict, Tuple
from datetime import datetime
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntrinsicProbabilities:
    """Intrinsic probability estimates from structural data."""
    home: float
    away: float
    draw: float
    confidence: float
    method: str

class IntrinsicProbabilityCalculator:
    """Calculate fair probabilities using intrinsic market data patterns."""
    
    def __init__(self):
        # Historical soccer statistics from large datasets
        self.soccer_base_rates = {
            'home_win_rate': 0.46,  # Home teams win ~46% historically
            'away_win_rate': 0.27,  # Away teams win ~27% historically  
            'draw_rate': 0.27,      # Draws occur ~27% historically
        }
        
        # League-specific adjustments (from public data)
        self.league_adjustments = {
            'Premier League': {'home_advantage': 0.03, 'draw_rate_adj': -0.02},
            'La Liga': {'home_advantage': 0.02, 'draw_rate_adj': 0.01},
            'Bundesliga': {'home_advantage': 0.025, 'draw_rate_adj': -0.01},
            'Serie A': {'home_advantage': 0.02, 'draw_rate_adj': 0.005},
            'Ligue 1': {'home_advantage': 0.025, 'draw_rate_adj': 0.0},
            # Default for other leagues
            'Other': {'home_advantage': 0.025, 'draw_rate_adj': 0.0}
        }
        
        # Time-based adjustments
        self.time_factors = {
            'weekend_home_boost': 0.01,  # Stronger home support on weekends
            'evening_home_boost': 0.005,  # Better atmosphere evening games
        }
    
    def calculate_home_field_advantage(self, league: str, match_time: datetime = None) -> float:
        """Calculate home field advantage for this specific context."""
        base_advantage = self.league_adjustments.get(
            league, self.league_adjustments['Other']
        )['home_advantage']
        
        adjustments = 0
        if match_time:
            # Weekend boost (Saturday/Sunday)
            if match_time.weekday() >= 5:
                adjustments += self.time_factors['weekend_home_boost']
            
            # Evening game boost (after 6pm local time)
            if match_time.hour >= 18:
                adjustments += self.time_factors['evening_home_boost']
        
        return base_advantage + adjustments
    
    def calculate_draw_probability_adjustment(self, league: str) -> float:
        """Calculate league-specific draw rate adjustments."""
        return self.league_adjustments.get(
            league, self.league_adjustments['Other']
        )['draw_rate_adj']
    
    def estimate_team_strength_from_odds(self, odds_data: Dict[str, float]) -> Tuple[float, float]:
        """
        Extract relative team strength from market odds structure.
        This uses the INFORMATION in odds without using them as fair probabilities.
        """
        home_odds = odds_data.get('home', odds_data.get('Home', 3.0))
        away_odds = odds_data.get('away', odds_data.get('Away', 3.0))
        
        # Use odds ratio to estimate relative strength
        # Lower odds = market thinks team is stronger
        strength_ratio = away_odds / home_odds
        
        if strength_ratio > 1.5:
            # Home team heavily favored
            home_strength = 0.65
            away_strength = 0.35
        elif strength_ratio > 1.2:
            # Home team favored
            home_strength = 0.58
            away_strength = 0.42
        elif strength_ratio > 0.8:
            # Relatively even match
            home_strength = 0.52
            away_strength = 0.48
        elif strength_ratio > 0.6:
            # Away team favored
            home_strength = 0.42
            away_strength = 0.58
        else:
            # Away team heavily favored
            home_strength = 0.35
            away_strength = 0.65
        
        return home_strength, away_strength
    
    def calculate_intrinsic_probabilities(self, market, odds_data: Dict[str, float]) -> IntrinsicProbabilities:
        """
        Calculate intrinsic probabilities using structural market data.
        This is our independent estimate vs market odds.
        """
        # Start with historical base rates
        base_home = self.soccer_base_rates['home_win_rate']
        base_away = self.soccer_base_rates['away_win_rate'] 
        base_draw = self.soccer_base_rates['draw_rate']
        
        # Adjust for league-specific patterns
        league = getattr(market, 'league_name', 'Other')
        home_advantage = self.calculate_home_field_advantage(league, market.maturity_date)
        draw_adjustment = self.calculate_draw_probability_adjustment(league)
        
        # Extract team strength information from odds structure
        home_strength, away_strength = self.estimate_team_strength_from_odds(odds_data)
        
        # Blend base rates with team strength estimates
        strength_weight = 0.4  # 40% weight to market-derived strength, 60% to base rates
        
        adjusted_home = (base_home * (1 - strength_weight) + 
                        home_strength * strength_weight + 
                        home_advantage)
        
        adjusted_away = (base_away * (1 - strength_weight) + 
                        away_strength * strength_weight)
        
        adjusted_draw = base_draw + draw_adjustment
        
        # Normalize to sum to 1.0 (this is our independent probability estimate)
        total = adjusted_home + adjusted_away + adjusted_draw
        
        home_prob = adjusted_home / total
        away_prob = adjusted_away / total
        draw_prob = adjusted_draw / total
        
        # Calculate confidence based on how much information we have
        confidence = 0.6  # Base confidence for structural model
        
        # Higher confidence for well-known leagues
        if league in ['Premier League', 'La Liga', 'Bundesliga', 'Serie A']:
            confidence += 0.1
            
        # Lower confidence for very uneven matches (higher uncertainty)
        home_away_ratio = max(home_prob/away_prob, away_prob/home_prob)
        if home_away_ratio > 2.0:
            confidence -= 0.1
        
        return IntrinsicProbabilities(
            home=home_prob,
            away=away_prob, 
            draw=draw_prob,
            confidence=min(confidence, 0.8),  # Cap at 80%
            method="structural_home_away_league"
        )

    def calculate_vig_arbitrage_edge(self, intrinsic_probs: IntrinsicProbabilities, 
                                   odds_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate edge using intrinsic probabilities vs market odds.
        This is technically correct Kelly edge calculation.
        """
        edges = {}
        
        # Map outcomes to probabilities
        prob_map = {
            'home': intrinsic_probs.home,
            'Home': intrinsic_probs.home,
            '1': intrinsic_probs.home,
            'away': intrinsic_probs.away,
            'Away': intrinsic_probs.away,
            '2': intrinsic_probs.away,
            'draw': intrinsic_probs.draw,
            'Draw': intrinsic_probs.draw,
            'X': intrinsic_probs.draw
        }
        
        for outcome, odds in odds_data.items():
            if outcome in prob_map and odds > 1.0:
                fair_prob = prob_map[outcome]
                
                # Kelly edge: (p * odds - 1) / (odds - 1) 
                kelly_edge = (fair_prob * odds - 1) / (odds - 1)
                
                # Convert to percentage
                edge_pct = kelly_edge * 100
                
                edges[outcome] = edge_pct
        
        return edges

def test_intrinsic_model():
    """Test the intrinsic probability model."""
    print("🧪 Testing Intrinsic Probability Model")
    
    calculator = IntrinsicProbabilityCalculator()
    
    # Create mock market data
    class MockMarket:
        def __init__(self):
            self.league_name = "Premier League"
            self.home_team = "Arsenal"
            self.away_team = "Chelsea"
            self.maturity_date = datetime(2024, 3, 15, 20, 0)  # Weekend evening
    
    market = MockMarket()
    
    # Test with different odds scenarios
    test_cases = [
        {'home': 2.45, 'away': 3.10, 'draw': 3.20, 'scenario': 'Home favored'},
        {'home': 3.20, 'away': 2.45, 'draw': 3.00, 'scenario': 'Away favored'},
        {'home': 2.95, 'away': 2.85, 'draw': 3.10, 'scenario': 'Even match'},
    ]
    
    for i, test_case in enumerate(test_cases):
        scenario = test_case.pop('scenario')
        odds_data = test_case
        
        print(f"\n📊 Test {i+1}: {scenario}")
        print(f"  Market Odds: {odds_data}")
        
        # Calculate market margin for reference
        implied_total = sum(1/odds for odds in odds_data.values())
        margin = (implied_total - 1) * 100
        print(f"  Market Margin: {margin:.2f}%")
        
        # Calculate intrinsic probabilities
        intrinsic = calculator.calculate_intrinsic_probabilities(market, odds_data)
        print(f"  Intrinsic Probs: home={intrinsic.home:.3f}, away={intrinsic.away:.3f}, draw={intrinsic.draw:.3f}")
        print(f"  Confidence: {intrinsic.confidence:.2f}")
        print(f"  Method: {intrinsic.method}")
        
        # Calculate edges
        edges = calculator.calculate_vig_arbitrage_edge(intrinsic, odds_data)
        print(f"  Calculated Edges: {edges}")
        
        avg_edge = sum(edges.values()) / len(edges) if edges else 0
        print(f"  Average Edge: {avg_edge:+.2f}%")
        
        # Identify best opportunities
        positive_edges = {k: v for k, v in edges.items() if v > 1.0}
        if positive_edges:
            best_bet = max(positive_edges.items(), key=lambda x: x[1])
            print(f"  Best Opportunity: {best_bet[0]} at {best_bet[1]:+.2f}% edge")
        else:
            print(f"  No positive edge opportunities")

if __name__ == "__main__":
    test_intrinsic_model()