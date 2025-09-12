#!/usr/bin/env python3
"""
Blockchain-Based Signal Provider

Uses blockchain market data and enrichment to generate trading signals,
replacing external API dependency with direct chain access.
"""

import logging
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from signals import SignalProvider
from market_enrichment import MarketEnrichmentService
from enhanced_tag_mappings import tag_mapper
from team_metadata_service import TeamMetadataService
from blockchain_reader import BlockchainReader

logger = logging.getLogger(__name__)


class BlockchainEnhancedSignal(SignalProvider):
    """
    Signal provider that uses blockchain market data with team metadata
    and venue information to generate enhanced predictions.
    """
    name = "blockchain_enhanced_signal"
    
    def __init__(self, networks: List[str] = None):
        self.networks = networks or ['optimism', 'arbitrum']
        self.enrichment_service = MarketEnrichmentService()
        self.metadata_service = TeamMetadataService()
        self.readers = {}
        
        # Initialize blockchain readers
        for network in self.networks:
            try:
                self.readers[network] = BlockchainReader(network)
                logger.info(f"✅ Initialized {network} reader for signals")
            except Exception as e:
                logger.warning(f"Could not initialize {network} reader: {e}")
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate probabilities using blockchain data and metadata.
        
        Args:
            df: DataFrame with market data
            
        Returns:
            Series of probabilities for home team winning
        """
        probs = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            try:
                prob = self._calculate_enhanced_probability(row)
                probs.loc[idx] = prob
            except Exception as e:
                logger.error(f"Error calculating probability for {idx}: {e}")
                probs.loc[idx] = 0.5  # Default to neutral probability
        
        return probs
    
    def _calculate_enhanced_probability(self, market_row) -> float:
        """Calculate probability using blockchain data and metadata."""
        
        # Get market source info
        source = getattr(market_row, 'source', '')
        if not source.startswith('blockchain_'):
            # Fallback for non-blockchain markets
            return self._basic_implied_probability(market_row)
        
        # Extract network and get enriched data
        network = source.replace('blockchain_', '')
        market_id = getattr(market_row, 'source_id', '')
        
        # Get enriched market data
        enriched = self.enrichment_service.enrich_market(
            market_id.replace('blockchain_', ''), network
        )
        
        if not enriched:
            logger.warning(f"Could not enrich market {market_id}")
            return self._basic_implied_probability(market_row)
        
        # Start with basic odds probability
        base_prob = self._extract_odds_probability(enriched)
        
        # Apply home field advantage
        home_advantage = self._calculate_home_advantage(enriched)
        
        # Apply team strength differential
        team_strength = self._calculate_team_strength_factor(enriched)
        
        # Apply venue-specific factors
        venue_factor = self._calculate_venue_factor(enriched)
        
        # Apply sport-specific factors
        sport_factor = self._calculate_sport_factor(enriched)
        
        # Combine all factors
        enhanced_prob = base_prob
        enhanced_prob = self._apply_factor(enhanced_prob, home_advantage, 0.1)
        enhanced_prob = self._apply_factor(enhanced_prob, team_strength, 0.15)
        enhanced_prob = self._apply_factor(enhanced_prob, venue_factor, 0.05)
        enhanced_prob = self._apply_factor(enhanced_prob, sport_factor, 0.08)
        
        # Ensure probability is within valid range
        enhanced_prob = max(0.1, min(0.9, enhanced_prob))
        
        logger.debug(f"Enhanced probability for {enriched.home_team.short_name} vs "
                    f"{enriched.away_team.short_name}: {enhanced_prob:.3f} "
                    f"(base: {base_prob:.3f})")
        
        return enhanced_prob
    
    def _extract_odds_probability(self, enriched) -> float:
        """Extract implied probability from current odds."""
        home_odds = enriched.current_odds.get('home', 0)
        away_odds = enriched.current_odds.get('away', 0)
        draw_odds = enriched.current_odds.get('draw', 0)
        
        if home_odds <= 0:
            return 0.5  # Default if no odds available
        
        # Convert decimal odds to implied probability
        home_implied = 1.0 / home_odds if home_odds > 0 else 0.33
        away_implied = 1.0 / away_odds if away_odds > 0 else 0.33
        draw_implied = 1.0 / draw_odds if draw_odds > 0 and enriched.has_draw else 0
        
        # Normalize probabilities (remove bookmaker margin)
        total_implied = home_implied + away_implied + draw_implied
        if total_implied > 0:
            normalized_home_prob = home_implied / total_implied
        else:
            normalized_home_prob = 0.5
        
        return normalized_home_prob
    
    def _calculate_home_advantage(self, enriched) -> float:
        """Calculate home field advantage factor."""
        # Different sports have different home advantages
        sport_advantages = {
            'Soccer': 0.15,      # Strong home advantage
            'Basketball': 0.12,   # Moderate home advantage  
            'American Football': 0.08,  # Some home advantage
            'Baseball': 0.06,     # Slight home advantage
            'Hockey': 0.10,       # Moderate home advantage
            'Tennis': 0.02,       # Minimal home advantage
            'MMA': 0.03,          # Slight advantage
            'Boxing': 0.03,       # Slight advantage
        }
        
        base_advantage = sport_advantages.get(enriched.sport.name, 0.05)
        
        # Enhance based on venue capacity (larger venues = more advantage)
        venue_multiplier = 1.0
        if enriched.home_team.venue:
            # Extract capacity if available in metadata
            venue_data = enriched.home_team.venue
            if isinstance(venue_data, dict) and 'capacity' in venue_data:
                capacity = venue_data['capacity']
                if capacity > 70000:
                    venue_multiplier = 1.3  # Large stadium advantage
                elif capacity > 40000:
                    venue_multiplier = 1.2  # Medium stadium advantage
                elif capacity > 20000:
                    venue_multiplier = 1.1  # Small stadium advantage
        
        return base_advantage * venue_multiplier
    
    def _calculate_team_strength_factor(self, enriched) -> float:
        """
        Calculate relative team strength factor.
        
        This would ideally use historical performance data,
        but for now uses team recognition and league tier.
        """
        # Premier teams get slight advantage
        premier_teams = {
            'Liverpool', 'Manchester City', 'Real Madrid', 'Barcelona',
            'Lakers', 'Warriors', 'Cowboys', 'Patriots', 'Knicks'
        }
        
        home_is_premier = any(name in enriched.home_team.full_name 
                             for name in premier_teams)
        away_is_premier = any(name in enriched.away_team.full_name 
                             for name in premier_teams)
        
        if home_is_premier and not away_is_premier:
            return 0.1  # Home team advantage
        elif away_is_premier and not home_is_premier:
            return -0.1  # Away team advantage
        else:
            return 0.0  # Equal strength
    
    def _calculate_venue_factor(self, enriched) -> float:
        """Calculate venue-specific factors."""
        # Famous difficult venues
        difficult_venues = {
            'Anfield': 0.08,           # Liverpool's fortress
            'Old Trafford': 0.06,      # Manchester United
            'Santiago Bernabéu': 0.07, # Real Madrid
            'Camp Nou': 0.08,          # Barcelona
            'AT&T Stadium': 0.05,      # Cowboys
        }
        
        if enriched.home_team.venue:
            venue_name = enriched.home_team.venue
            if isinstance(venue_name, dict):
                venue_name = venue_name.get('name', '')
            elif not isinstance(venue_name, str):
                venue_name = str(venue_name)
            
            return difficult_venues.get(venue_name, 0.0)
        
        return 0.0
    
    def _calculate_sport_factor(self, enriched) -> float:
        """Calculate sport-specific factors."""
        sport_factors = {
            'Soccer': self._soccer_specific_factors(enriched),
            'Basketball': self._basketball_specific_factors(enriched),
            'American Football': self._football_specific_factors(enriched),
            'Baseball': self._baseball_specific_factors(enriched),
        }
        
        return sport_factors.get(enriched.sport.name, 0.0)
    
    def _soccer_specific_factors(self, enriched) -> float:
        """Soccer-specific factors."""
        factor = 0.0
        
        # Derby matches (same city teams) are more unpredictable
        if self._is_derby_match(enriched):
            factor -= 0.05  # Reduce home advantage for derbies
        
        # Champions League/Europa League venues
        european_venues = ['Camp Nou', 'Santiago Bernabéu', 'Anfield']
        if enriched.home_team.venue in european_venues:
            factor += 0.03
        
        return factor
    
    def _basketball_specific_factors(self, enriched) -> float:
        """Basketball-specific factors."""
        # NBA altitude advantage (Denver)
        if 'Denver' in enriched.home_team.full_name:
            return 0.04
        
        # Back-to-back games would reduce advantage (not implemented)
        return 0.0
    
    def _football_specific_factors(self, enriched) -> float:
        """American Football-specific factors."""
        # Weather factors for outdoor stadiums
        if enriched.home_team.venue and 'Stadium' in str(enriched.home_team.venue):
            return 0.02  # Outdoor stadium advantage
        
        return 0.0
    
    def _baseball_specific_factors(self, enriched) -> float:
        """Baseball-specific factors."""
        # Pitcher handedness, wind, etc. would go here
        return 0.0
    
    def _is_derby_match(self, enriched) -> bool:
        """Check if this is a derby match (same city teams)."""
        derby_pairs = [
            ('Liverpool', 'Everton'),
            ('Manchester City', 'Manchester United'),
            ('Real Madrid', 'Atlético Madrid'),
            ('Barcelona', 'Espanyol'),
            ('Lakers', 'Clippers'),
        ]
        
        for team1, team2 in derby_pairs:
            if ((team1 in enriched.home_team.full_name and 
                 team2 in enriched.away_team.full_name) or
                (team2 in enriched.home_team.full_name and 
                 team1 in enriched.away_team.full_name)):
                return True
        
        return False
    
    def _apply_factor(self, base_prob: float, factor: float, weight: float) -> float:
        """Apply a factor with given weight to the base probability."""
        # Convert probability to log-odds, apply factor, convert back
        if base_prob <= 0:
            base_prob = 0.01
        elif base_prob >= 1:
            base_prob = 0.99
        
        log_odds = np.log(base_prob / (1 - base_prob))
        adjusted_log_odds = log_odds + (factor * weight)
        adjusted_prob = 1 / (1 + np.exp(-adjusted_log_odds))
        
        return adjusted_prob
    
    def _basic_implied_probability(self, market_row) -> float:
        """Fallback probability calculation for non-blockchain markets."""
        # This would use the existing implied probability logic
        return 0.5  # Neutral probability as fallback


class BlockchainMarketScout(SignalProvider):
    """
    Signal provider that scouts for arbitrage opportunities
    and value bets using real-time blockchain data.
    """
    name = "blockchain_market_scout"
    
    def __init__(self, networks: List[str] = None):
        self.networks = networks or ['optimism', 'arbitrum']
        self.enrichment_service = MarketEnrichmentService()
        self.readers = {}
        
        for network in self.networks:
            try:
                self.readers[network] = BlockchainReader(network)
                logger.info(f"✅ Initialized {network} scout reader")
            except Exception as e:
                logger.warning(f"Could not initialize {network} reader: {e}")
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """
        Scout for value bets by comparing odds across positions
        and identifying mispriced markets.
        """
        probs = pd.Series(index=df.index, dtype=float)
        
        for idx, row in df.iterrows():
            try:
                prob = self._scout_market_value(row)
                probs.loc[idx] = prob
            except Exception as e:
                logger.error(f"Error scouting market {idx}: {e}")
                probs.loc[idx] = 0.5
        
        return probs
    
    def _scout_market_value(self, market_row) -> float:
        """Scout for value in the market."""
        source = getattr(market_row, 'source', '')
        if not source.startswith('blockchain_'):
            return 0.5
        
        network = source.replace('blockchain_', '')
        market_id = getattr(market_row, 'source_id', '')
        
        # Get current odds from blockchain
        reader = self.readers.get(network)
        if not reader:
            return 0.5
        
        market_address = market_id.replace('blockchain_', '')
        current_odds = reader.get_current_odds(market_address)
        
        if not current_odds:
            return 0.5
        
        # Look for value opportunities
        value_score = self._calculate_value_score(current_odds)
        
        # Convert value score to probability (higher value = higher probability)
        prob = 0.5 + (value_score * 0.3)  # Scale factor
        return max(0.1, min(0.9, prob))
    
    def _calculate_value_score(self, odds_data: Dict) -> float:
        """Calculate value score based on odds analysis."""
        if not odds_data or len(odds_data) < 2:
            return 0.0
        
        # Get buy odds for home and away
        home_odds = odds_data.get(0, {}).get('buy', 0)
        away_odds = odds_data.get(1, {}).get('buy', 0)
        
        if home_odds <= 0 or away_odds <= 0:
            return 0.0
        
        # Calculate implied probabilities
        home_implied = 1.0 / home_odds
        away_implied = 1.0 / away_odds
        total_implied = home_implied + away_implied
        
        # Look for markets with low overround (good value)
        overround = total_implied - 1.0
        
        # Lower overround = better value
        value_score = max(0, 0.2 - overround) / 0.2  # Scale 0-1
        
        # Bonus for extreme odds (potential value)
        if home_odds > 3.0 or home_odds < 1.5:
            value_score += 0.1
        
        return min(1.0, value_score)


def test_blockchain_signals():
    """Test the blockchain signal providers."""
    logger.info("Testing Blockchain Signal Providers")
    logger.info("=" * 50)
    
    # Create test data
    test_data = pd.DataFrame([
        {
            'source': 'blockchain_optimism',
            'source_id': 'blockchain_demo_soccer_001',
            'sport': 'Soccer',
            'league_name': 'English Premier League',
            'home_team': 'Liverpool',
            'away_team': 'Manchester City'
        },
        {
            'source': 'blockchain_optimism', 
            'source_id': 'blockchain_demo_nba_001',
            'sport': 'Basketball',
            'league_name': 'National Basketball Association',
            'home_team': 'Lakers',
            'away_team': 'Warriors'
        }
    ])
    
    # Test enhanced signal provider
    logger.info("Testing BlockchainEnhancedSignal...")
    enhanced_signal = BlockchainEnhancedSignal()
    enhanced_probs = enhanced_signal.get_probs(test_data)
    
    logger.info("Enhanced Signal Results:")
    for idx, prob in enhanced_probs.items():
        row = test_data.loc[idx]
        logger.info(f"  {row['home_team']} vs {row['away_team']}: {prob:.3f}")
    
    # Test market scout
    logger.info("\nTesting BlockchainMarketScout...")
    scout_signal = BlockchainMarketScout()
    scout_probs = scout_signal.get_probs(test_data)
    
    logger.info("Market Scout Results:")
    for idx, prob in scout_probs.items():
        row = test_data.loc[idx]
        logger.info(f"  {row['home_team']} vs {row['away_team']}: {prob:.3f}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_blockchain_signals()