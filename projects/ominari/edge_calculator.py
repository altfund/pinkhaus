#!/usr/bin/env python3
"""Enhanced edge calculation using real signal providers"""

import pandas as pd
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional
import numpy as np
from signals import get_signal_providers, SIGNAL_WEIGHTS

class EdgeCalculator:
    """Calculate edge using multiple signal providers"""
    
    def __init__(self):
        self.signal_providers = get_signal_providers()
        self.weights = SIGNAL_WEIGHTS
        print(f"EdgeCalculator initialized with {len(self.signal_providers)} signal providers")
    
    def calculate_edges(self, markets: List[Dict]) -> List[Dict]:
        """Calculate edges for a list of markets"""
        if not markets:
            return []
        
        # Convert to DataFrame for signal providers
        df = pd.DataFrame(markets)
        
        # Prepare data for signal providers
        if 'source_id' not in df.columns and 'market_id' in df.columns:
            df['source_id'] = df['market_id']
        
        # Add required columns for signals
        if 'normalized_outcome' not in df.columns:
            # Infer from position type (home/draw/away)
            df['normalized_outcome'] = df.apply(self._get_normalized_outcome, axis=1)
        
        if 'time' not in df.columns:
            df['time'] = datetime.now(timezone.utc)
        
        # Calculate implied probabilities from odds
        if 'odds' not in df.columns:
            # Map from home_odds/draw_odds/away_odds based on position
            df['odds'] = df.apply(self._get_odds_for_position, axis=1)
        
        if 'implied_raw' not in df.columns and 'odds' in df.columns:
            df['implied_raw'] = (1.0 / df['odds'].replace(0, float('inf')) * 100).fillna(50)
        
        # Get probabilities from each signal provider
        all_probs = {}
        for provider in self.signal_providers:
            try:
                probs = provider.get_probs(df)
                all_probs[provider.name] = probs
                print(f"Got {len(probs)} probabilities from {provider.name}")
            except Exception as e:
                print(f"Error with provider {provider.name}: {e}")
                # Fallback to implied probabilities
                all_probs[provider.name] = df['implied_raw'] / 100.0
        
        # Combine probabilities using weights
        combined_probs = self._combine_probabilities(all_probs, df)
        
        # Calculate edges
        edges = []
        for idx, row in df.iterrows():
            market_id = row.get('market_id', row.get('source_id', ''))
            
            # Get odds for this market
            home_odds = float(row.get('home_odds', 0))
            draw_odds = float(row.get('draw_odds', 0))
            away_odds = float(row.get('away_odds', 0))
            
            # Get combined probability for this outcome
            prob = combined_probs[idx] if idx < len(combined_probs) else 0.5
            
            # Calculate edge based on position
            position = row.get('position', '').lower()
            if position == 'home':
                edge = self._calculate_single_edge(prob, home_odds)
            elif position == 'draw':
                edge = self._calculate_single_edge(prob, draw_odds)
            elif position == 'away':
                edge = self._calculate_single_edge(prob, away_odds)
            else:
                # Calculate edges for all positions
                home_prob = prob if 'home' in str(row.get('normalized_outcome', '')).lower() else 0.33
                draw_prob = prob if 'draw' in str(row.get('normalized_outcome', '')).lower() else 0.33
                away_prob = prob if 'away' in str(row.get('normalized_outcome', '')).lower() else 0.33
                
                edge = {
                    'home': self._calculate_single_edge(home_prob, home_odds),
                    'draw': self._calculate_single_edge(draw_prob, draw_odds),
                    'away': self._calculate_single_edge(away_prob, away_odds)
                }
            
            edges.append({
                'market_id': market_id,
                'edge': edge,
                'probability': prob,
                'confidence': self._calculate_confidence(all_probs, idx)
            })
        
        return edges
    
    def _get_normalized_outcome(self, row):
        """Get normalized outcome from row data"""
        position = str(row.get('position', '')).lower()
        if position == 'home':
            return f"{row.get('home_team', 'Home')} Win"
        elif position == 'draw':
            return "Draw"
        elif position == 'away':
            return f"{row.get('away_team', 'Away')} Win"
        return "Unknown"
    
    def _get_odds_for_position(self, row):
        """Get odds based on position"""
        position = str(row.get('position', '')).lower()
        if position == 'home':
            return float(row.get('home_odds', 0))
        elif position == 'draw':
            return float(row.get('draw_odds', 0))
        elif position == 'away':
            return float(row.get('away_odds', 0))
        return 0
    
    def _combine_probabilities(self, all_probs: Dict[str, pd.Series], df: pd.DataFrame) -> pd.Series:
        """Combine probabilities from multiple providers using weights"""
        if not all_probs:
            return pd.Series([0.5] * len(df), index=df.index)
        
        # Initialize combined probabilities
        combined = pd.Series([0.0] * len(df), index=df.index)
        total_weight = 0.0
        
        for provider_name, probs in all_probs.items():
            weight = self.weights.get(provider_name, 1.0)
            # Handle any invalid probabilities
            valid_probs = probs.fillna(0.5).clip(0.01, 0.99)
            combined += valid_probs * weight
            total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined = combined / total_weight
        
        return combined.clip(0.01, 0.99)
    
    def _calculate_single_edge(self, probability: float, odds: float) -> float:
        """Calculate edge for a single outcome"""
        if odds <= 1 or probability <= 0:
            return 0.0
        
        # Edge = (probability * odds - 1) * 100
        # Positive edge means expected value is positive
        edge = (probability * odds - 1) * 100
        
        # Apply confidence adjustment based on odds range
        if odds < 2.0:  # Heavy favorite
            edge *= 0.9  # Reduce edge by 10% due to favorite-longshot bias
        elif odds > 5.0:  # Longshot
            edge *= 1.1  # Increase edge by 10% for longshots
        
        return round(edge, 2)
    
    def _calculate_confidence(self, all_probs: Dict[str, pd.Series], idx: int) -> float:
        """Calculate confidence score based on agreement between providers"""
        if not all_probs or len(all_probs) == 1:
            return 0.5
        
        # Get all probabilities for this index
        probs = []
        for provider_probs in all_probs.values():
            if idx in provider_probs.index:
                probs.append(provider_probs[idx])
        
        if len(probs) < 2:
            return 0.5
        
        # Calculate standard deviation - lower std = higher confidence
        std_dev = np.std(probs)
        
        # Convert to confidence score (0-1)
        # std_dev of 0 = perfect agreement = confidence 1.0
        # std_dev of 0.5 = max disagreement = confidence 0.0
        confidence = max(0, 1 - (std_dev * 2))
        
        return round(confidence, 3)
    
    def calculate_kelly_stake(self, edge: float, odds: float, bankroll: float, 
                            kelly_fraction: float = 0.25, min_bet: float = 10, 
                            cap_per_bet: float = 0.02) -> float:
        """Calculate Kelly stake based on edge and odds"""
        if edge <= 0 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where f = fraction to bet, b = odds - 1, p = probability, q = 1-p
        # edge = (p * odds - 1) * 100, so p = (edge/100 + 1) / odds
        probability = (edge / 100 + 1) / odds
        b = odds - 1
        kelly_stake = (probability * b - (1 - probability)) / b * kelly_fraction * bankroll
        
        # Apply constraints
        if kelly_stake < min_bet:
            return 0
        
        return min(kelly_stake, bankroll * cap_per_bet)