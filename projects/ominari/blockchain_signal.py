#!/usr/bin/env python3
"""
Blockchain Signal Provider

Uses real-time blockchain oracle data to generate trading signals.
Integrates with the signal registry for paper trading.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from signals import SignalProvider
from blockchain_reader import BlockchainReader
from database_v2 import db_manager
from models import Market, Odd

logger = logging.getLogger(__name__)


class BlockchainOracleSignal(SignalProvider):
    """Signal provider using blockchain oracle data."""
    
    name = "blockchain_oracle"
    
    def __init__(self, networks: List[str] = None):
        """
        Initialize blockchain signal provider.
        
        Args:
            networks: List of networks to use (default: ['optimism', 'arbitrum'])
        """
        self.networks = networks or ['optimism', 'arbitrum']
        self.readers = {}
        self.oracle_data_cache = {}
        self.cache_ttl = 60  # Cache for 1 minute
        
        # Initialize blockchain readers
        self._init_readers()
    
    def _init_readers(self):
        """Initialize blockchain readers for each network."""
        contracts = {
            'optimism': {
                'sports_amm': '0x170a5714112daEfF20E798B6e92e25B86Ea603C1',
                'market_manager': '0x5ed98Ebb66A929758C7Fe5Ac60c979aDF0F4040a'
            },
            'arbitrum': {
                'sports_amm': '0xd9aB397Fb7B3849A010f0e9a516Ab333feC52891',
                'market_manager': '0x3E10355b57e7eFEbB6F405beFf6C2dDb078d8769'
            }
        }
        
        for network in self.networks:
            try:
                reader = BlockchainReader(
                    network=network,
                    contracts=contracts.get(network, {})
                )
                self.readers[network] = reader
                logger.info(f"Initialized {network} blockchain reader")
            except Exception as e:
                logger.error(f"Failed to initialize {network} reader: {e}")
    
    def get_oracle_prices(self, market_id: str) -> Optional[Dict[str, float]]:
        """
        Get current oracle prices from blockchain.
        
        Returns:
            Dict with home_prob, away_prob, draw_prob (if applicable)
        """
        # Check cache first
        cache_key = f"oracle_{market_id}"
        if cache_key in self.oracle_data_cache:
            cached_data, timestamp = self.oracle_data_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
        
        # Try each network
        for network, reader in self.readers.items():
            try:
                # Get market data from blockchain
                market_data = reader.get_market_data(market_id)
                
                if market_data and 'oracle_prices' in market_data:
                    prices = market_data['oracle_prices']
                    
                    # Convert to probabilities
                    total = sum(prices.values())
                    probabilities = {
                        outcome: price / total 
                        for outcome, price in prices.items()
                    }
                    
                    # Cache the result
                    self.oracle_data_cache[cache_key] = (probabilities, datetime.now())
                    
                    return probabilities
                    
            except Exception as e:
                logger.debug(f"Failed to get oracle data from {network}: {e}")
        
        return None
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculate probabilities using blockchain oracle data.
        
        Args:
            df: DataFrame with columns:
                - market_id: Unique market identifier
                - home_odds, away_odds, draw_odds: Current betting odds
                - is_finished: Whether market is closed
                
        Returns:
            Series of probabilities indexed by 'outcome_market_id'
        """
        probs = {}
        
        for idx, row in df.iterrows():
            if row['is_finished']:
                continue
            
            market_id = row['market_id']
            
            # Get oracle probabilities
            oracle_probs = self.get_oracle_prices(market_id)
            
            if oracle_probs:
                # Use oracle data directly
                for outcome, prob in oracle_probs.items():
                    key = f"{outcome}_{market_id}"
                    probs[key] = prob
            else:
                # Fallback to implied probabilities with adjustment
                # This gives us a baseline when oracle data isn't available
                
                # Calculate implied probabilities
                impl_home = 1 / row['home_odds'] if row['home_odds'] > 0 else 0
                impl_away = 1 / row['away_odds'] if row['away_odds'] > 0 else 0
                impl_draw = 1 / row['draw_odds'] if row.get('draw_odds', 0) > 0 else 0
                
                # Remove overround
                total_impl = impl_home + impl_away + impl_draw
                if total_impl > 0:
                    # Apply a small edge adjustment based on historical accuracy
                    # This represents our confidence in the market efficiency
                    edge_factor = 0.98  # 2% skepticism about market efficiency
                    
                    probs[f"home_{market_id}"] = (impl_home / total_impl) * edge_factor
                    probs[f"away_{market_id}"] = (impl_away / total_impl) * edge_factor
                    
                    if impl_draw > 0:
                        probs[f"draw_{market_id}"] = (impl_draw / total_impl) * edge_factor
        
        return pd.Series(probs)
    
    def get_confidence(self, market_id: str) -> float:
        """
        Get confidence score for predictions on this market.
        
        Higher confidence when:
        - Oracle data is available
        - Multiple oracles agree
        - Recent data updates
        """
        oracle_data = self.get_oracle_prices(market_id)
        
        if oracle_data:
            # High confidence with oracle data
            return 0.9
        else:
            # Lower confidence without oracle data
            return 0.6


class HybridBlockchainSignal(SignalProvider):
    """
    Hybrid signal combining blockchain oracle with historical performance.
    """
    
    name = "hybrid_blockchain"
    
    def __init__(self):
        self.blockchain_signal = BlockchainOracleSignal()
        self.historical_weight = 0.3
        self.oracle_weight = 0.7
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """
        Combine blockchain oracle with historical data analysis.
        """
        # Get blockchain probabilities
        blockchain_probs = self.blockchain_signal.get_probs(df)
        
        # Calculate historical adjustments
        probs = {}
        
        for idx, row in df.iterrows():
            if row['is_finished']:
                continue
                
            market_id = row['market_id']
            
            # Get historical performance for similar matches
            historical_edge = self._calculate_historical_edge(row)
            
            # Combine blockchain and historical signals
            for outcome in ['home', 'away', 'draw']:
                key = f"{outcome}_{market_id}"
                
                if key in blockchain_probs:
                    blockchain_prob = blockchain_probs[key]
                    
                    # Adjust based on historical data
                    if outcome in historical_edge:
                        hist_adjustment = historical_edge[outcome]
                        combined_prob = (
                            self.oracle_weight * blockchain_prob + 
                            self.historical_weight * hist_adjustment
                        )
                        probs[key] = np.clip(combined_prob, 0.01, 0.99)
                    else:
                        probs[key] = blockchain_prob
        
        return pd.Series(probs)
    
    def _calculate_historical_edge(self, market_row) -> Dict[str, float]:
        """
        Calculate historical edge based on past performance.
        """
        # This is a simplified version - in production you'd want
        # more sophisticated historical analysis
        
        edges = {}
        
        # Example: home team advantage in certain leagues
        if 'league' in market_row:
            league = market_row['league']
            
            # Some leagues have stronger home advantage
            home_advantage_leagues = ['Premier League', 'La Liga', 'Serie A']
            if any(league_name in league for league_name in home_advantage_leagues):
                edges['home'] = 0.52  # 52% historical home win rate
                edges['away'] = 0.30  # 30% away win rate
                edges['draw'] = 0.18  # 18% draw rate
            else:
                # Default probabilities
                edges['home'] = 0.45
                edges['away'] = 0.35
                edges['draw'] = 0.20
        
        return edges


def test_blockchain_signal():
    """Test the blockchain signal provider."""
    print("=== Testing Blockchain Signal Provider ===\n")
    
    # Create signal
    signal = BlockchainOracleSignal()
    
    # Get some test markets
    with db_manager.get_db_session() as db:
        markets = db.query(Market).filter(
            Market.is_finished == False
        ).limit(5).all()
        
        if not markets:
            print("No active markets found!")
            return
        
        # Create DataFrame
        df_data = []
        for market in markets:
            # Get latest odds
            odd = db.query(Odd).filter(
                Odd.market_id == market.market_id
            ).order_by(Odd.updated_at.desc()).first()
            
            if odd:
                df_data.append({
                    'market_id': market.market_id,
                    'home_odds': odd.home_odds,
                    'away_odds': odd.away_odds,
                    'draw_odds': odd.draw_odds,
                    'is_finished': market.is_finished,
                    'sport': market.sport,
                    'home_team': market.home_team,
                    'away_team': market.away_team
                })
        
        df = pd.DataFrame(df_data)
        
        # Get probabilities
        print("Calculating probabilities...\n")
        probs = signal.get_probs(df)
        
        # Display results
        for idx, row in df.iterrows():
            print(f"{row['home_team']} vs {row['away_team']}")
            market_id = row['market_id']
            
            # Show oracle probabilities
            oracle_data = signal.get_oracle_prices(market_id)
            if oracle_data:
                print("  Oracle probabilities:")
                for outcome, prob in oracle_data.items():
                    print(f"    {outcome}: {prob:.2%}")
            
            # Show signal probabilities
            print("  Signal probabilities:")
            for outcome in ['home', 'away', 'draw']:
                key = f"{outcome}_{market_id}"
                if key in probs:
                    print(f"    {outcome}: {probs[key]:.2%}")
            
            print()


if __name__ == "__main__":
    # Test the signal
    test_blockchain_signal()