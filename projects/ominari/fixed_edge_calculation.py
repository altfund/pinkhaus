#!/usr/bin/env python3
"""
Fixed Edge Calculation using Intrinsic Probability Models
Replaces flawed normalized-odds approach with structural market analysis.
"""

import os
import sys
import logging
from typing import Dict, Optional
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from intrinsic_probability_models import IntrinsicProbabilityCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedEdgeSignalProvider:
    """Fixed edge signal provider using intrinsic probability models."""
    
    def __init__(self):
        self.intrinsic_calculator = IntrinsicProbabilityCalculator()
        self.min_edge_threshold = 1.0  # 1% minimum edge
        self.min_confidence_threshold = 0.5  # 50% minimum confidence
        
    def generate_signals_for_market(self, market, odds_data: Dict[str, float]) -> Dict[str, Dict]:
        """
        Generate edge signals using intrinsic probability models.
        This replaces the flawed normalized-odds approach.
        """
        if market.sport != 'Soccer':
            return {}
        
        try:
            # Calculate intrinsic probabilities using structural data
            intrinsic_probs = self.intrinsic_calculator.calculate_intrinsic_probabilities(market, odds_data)
            
            # Skip if confidence too low
            if intrinsic_probs.confidence < self.min_confidence_threshold:
                logger.debug(f"Skipping {market.home_team} vs {market.away_team}: low confidence {intrinsic_probs.confidence:.2f}")
                return {}
            
            # Calculate edges using intrinsic vs market odds
            edges = self.intrinsic_calculator.calculate_vig_arbitrage_edge(intrinsic_probs, odds_data)
            
            # Generate signals for positive edges above threshold
            signals = {}
            
            for outcome, edge in edges.items():
                if edge > self.min_edge_threshold:
                    # Map outcome to standardized format
                    outcome_map = {
                        'home': 'home', 'Home': 'home', '1': 'home',
                        'away': 'away', 'Away': 'away', '2': 'away',
                        'draw': 'draw', 'Draw': 'draw', 'X': 'draw'
                    }
                    
                    standard_outcome = outcome_map.get(outcome, outcome)
                    signal_key = f"fixed_edge_{standard_outcome}"
                    
                    # Get the probability for this outcome
                    prob_map = {
                        'home': intrinsic_probs.home,
                        'away': intrinsic_probs.away,
                        'draw': intrinsic_probs.draw
                    }
                    
                    fair_prob = prob_map.get(standard_outcome, 0.33)
                    market_odds = odds_data.get(outcome, 3.0)
                    
                    signals[signal_key] = {
                        'forecast': fair_prob,  # Our intrinsic probability estimate
                        'edge': edge,  # Kelly edge percentage
                        'confidence': intrinsic_probs.confidence,
                        'market_odds': market_odds,
                        'outcome': standard_outcome,
                        'method': intrinsic_probs.method,
                        'signal_type': 'intrinsic_structural'
                    }
                    
                    logger.info(f"✅ Signal: {market.home_team} vs {market.away_team} - {standard_outcome} @ {market_odds:.2f} = {edge:+.2f}% edge")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating fixed edge signals: {e}")
            return {}

def test_fixed_edge_calculation():
    """Test the fixed edge calculation approach."""
    print("🔧 Testing Fixed Edge Calculation")
    
    provider = FixedEdgeSignalProvider()
    
    # Create mock market
    class MockMarket:
        def __init__(self):
            self.sport = "Soccer"
            self.league_name = "Premier League"
            self.home_team = "Arsenal"
            self.away_team = "Chelsea"
            self.maturity_date = datetime(2024, 3, 15, 20, 0)
    
    market = MockMarket()
    
    # Test with the same odds that were showing -4.15% average edge before
    odds_data = {'home': 2.45, 'away': 3.10, 'draw': 3.20}
    
    print(f"Testing market: {market.home_team} vs {market.away_team}")
    print(f"Odds: {odds_data}")
    
    # Calculate market margin for reference
    implied_total = sum(1/odds for odds in odds_data.values())
    margin = (implied_total - 1) * 100
    print(f"Market Margin: {margin:.2f}%")
    
    # Generate signals
    signals = provider.generate_signals_for_market(market, odds_data)
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal_key, signal_data in signals.items():
        print(f"  {signal_key}:")
        print(f"    Edge: {signal_data['edge']:+.2f}%")
        print(f"    Confidence: {signal_data['confidence']:.2f}")
        print(f"    Fair Prob: {signal_data['forecast']:.3f}")
        print(f"    Market Odds: {signal_data['market_odds']:.2f}")
        print(f"    Method: {signal_data['method']}")
    
    if signals:
        avg_edge = sum(s['edge'] for s in signals.values()) / len(signals)
        print(f"\nAverage Edge of Tradeable Signals: {avg_edge:+.2f}%")
    else:
        print(f"\nNo signals above threshold")

if __name__ == "__main__":
    test_fixed_edge_calculation()