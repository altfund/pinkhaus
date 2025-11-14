#!/usr/bin/env python3
"""
Advanced Trading Strategies for Ominari
Implements multiple strategies with dynamic weight adjustment
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sqlalchemy import func

from database_v2 import db_manager
from models import Market, Odd, Bet


class TradingStrategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str, min_edge: float = 2.0):
        self.name = name
        self.min_edge = min_edge
        self.performance_history = []
        self.weight = 1.0  # Dynamic weight based on performance
        
    @abstractmethod
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Evaluate a trading opportunity and return signal if favorable"""
        pass
        
    @abstractmethod
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Calculate optimal bet size based on edge and bankroll"""
        pass
        
    def update_performance(self, bet_result: Dict):
        """Update strategy performance based on bet result"""
        self.performance_history.append({
            'timestamp': datetime.now(),
            'won': bet_result['won'],
            'stake': bet_result['stake'],
            'payout': bet_result['payout'],
            'edge': bet_result['edge']
        })
        
        # Adjust weight based on recent performance
        if len(self.performance_history) >= 10:
            recent_results = self.performance_history[-20:]
            win_rate = sum(1 for r in recent_results if r['won']) / len(recent_results)
            avg_edge = np.mean([r['edge'] for r in recent_results])
            
            # Adjust weight: higher for strategies with good win rate and edge
            self.weight = 0.5 + (win_rate * 0.5) + (avg_edge / 100)
            self.weight = max(0.1, min(2.0, self.weight))  # Cap between 0.1 and 2.0


class ValueBettingStrategy(TradingStrategy):
    """Classic value betting - bet when odds offer positive expected value"""
    
    def __init__(self):
        super().__init__("value_betting", min_edge=3.0)
        
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Look for value in mispriced odds"""
        if len(odds) < 2:
            return None
            
        # Calculate fair probabilities
        total_implied = sum(1/odd.decimal_odds for odd in odds.values())
        
        best_value = None
        best_edge = 0
        
        for outcome, odd in odds.items():
            # Fair probability after removing margin
            fair_prob = (1/odd.decimal_odds) / total_implied
            
            # Expected value
            ev = (odd.decimal_odds * fair_prob) - 1
            edge_pct = ev * 100
            
            if edge_pct > best_edge and edge_pct >= self.min_edge:
                best_edge = edge_pct
                best_value = {
                    'outcome': outcome,
                    'odds': odd.decimal_odds,
                    'edge': edge_pct,
                    'fair_prob': fair_prob,
                    'strategy': self.name
                }
                
        return best_value
        
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Kelly criterion with conservative fraction"""
        edge_decimal = edge / 100
        win_prob = 1 / odds + edge_decimal
        
        kelly_fraction = (win_prob * odds - 1) / (odds - 1)
        kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25%
        
        # Apply conservative factor
        bet_fraction = kelly_fraction * 0.3  # 30% of Kelly
        bet_amount = bankroll * bet_fraction
        
        # Min/max constraints
        return max(10, min(bet_amount, bankroll * 0.05))  # 0.5-5% of bankroll


class MomentumStrategy(TradingStrategy):
    """Bet on markets with strong odds movement in one direction"""
    
    def __init__(self):
        super().__init__("momentum", min_edge=2.5)
        self.lookback_hours = 4
        
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Look for momentum in odds movement"""
        with db_manager.get_db_session() as db:
            # Get historical odds
            cutoff_time = datetime.now() - timedelta(hours=self.lookback_hours)
            
            momentum_signals = []
            
            for outcome, current_odd in odds.items():
                historical_odds = db.query(Odd).filter(
                    Odd.market_id == market.id,
                    Odd.outcome == outcome,
                    Odd.created_at >= cutoff_time
                ).order_by(Odd.created_at).all()
                
                if len(historical_odds) < 5:
                    continue
                    
                # Calculate momentum
                odds_series = [o.decimal_odds for o in historical_odds]
                
                # Linear regression slope
                x = np.arange(len(odds_series))
                slope = np.polyfit(x, odds_series, 1)[0]
                
                # Positive slope = odds increasing = probability decreasing
                # We want to bet against the momentum (contrarian)
                if slope > 0.01:  # Odds drifting up
                    # Market thinks this outcome is less likely
                    # We bet if we still see value
                    implied_prob = 1 / current_odd.decimal_odds
                    momentum_edge = (slope * 10)  # Scale momentum to edge
                    
                    momentum_signals.append({
                        'outcome': outcome,
                        'odds': current_odd.decimal_odds,
                        'edge': momentum_edge,
                        'momentum': slope,
                        'strategy': self.name
                    })
                    
        # Return best momentum signal
        if momentum_signals:
            return max(momentum_signals, key=lambda x: x['edge'])
            
        return None
        
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Smaller bets for momentum strategy due to higher uncertainty"""
        bet_fraction = (edge / 100) * 0.2  # Very conservative
        bet_amount = bankroll * bet_fraction
        
        return max(10, min(bet_amount, bankroll * 0.03))  # 0.5-3% of bankroll


class MarketTimingStrategy(TradingStrategy):
    """Bet on markets at optimal times based on liquidity and information"""
    
    def __init__(self):
        super().__init__("market_timing", min_edge=2.0)
        
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Evaluate based on market timing factors"""
        now = datetime.now()
        time_to_start = (market.start_time - now).total_seconds() / 3600  # Hours
        
        # Skip if too far or too close to start
        if time_to_start > 24 or time_to_start < 0.5:
            return None
            
        # Check market liquidity (number of odds updates)
        with db_manager.get_db_session() as db:
            odds_count = db.query(func.count(Odd.id)).filter(
                Odd.market_id == market.id,
                Odd.created_at >= now - timedelta(hours=1)
            ).scalar()
            
        # Low liquidity = potential inefficiency
        liquidity_score = min(odds_count / 10, 1.0)  # Normalize to 0-1
        inefficiency_bonus = (1 - liquidity_score) * 2  # Up to 2% bonus
        
        # Time-based edge (markets are less efficient 2-6 hours before)
        if 2 <= time_to_start <= 6:
            time_bonus = 1.5
        else:
            time_bonus = 0
            
        # Find best value with timing bonus
        best_opportunity = None
        best_score = 0
        
        for outcome, odd in odds.items():
            implied_prob = 1 / odd.decimal_odds
            base_edge = (1 - implied_prob) * inefficiency_bonus
            total_edge = base_edge + time_bonus
            
            if total_edge >= self.min_edge and total_edge > best_score:
                best_score = total_edge
                best_opportunity = {
                    'outcome': outcome,
                    'odds': odd.decimal_odds,
                    'edge': total_edge,
                    'liquidity_score': liquidity_score,
                    'time_to_start': time_to_start,
                    'strategy': self.name
                }
                
        return best_opportunity
        
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Moderate bet sizing for timing strategy"""
        bet_fraction = (edge / 100) * 0.25
        bet_amount = bankroll * bet_fraction
        
        return max(10, min(bet_amount, bankroll * 0.04))


class ArbitrageStrategy(TradingStrategy):
    """Look for arbitrage opportunities across outcomes"""
    
    def __init__(self):
        super().__init__("arbitrage", min_edge=0.5)  # Lower threshold for arb
        
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Find arbitrage opportunities"""
        if len(odds) < 2:
            return None
            
        # Calculate total implied probability
        total_implied = sum(1/odd.decimal_odds for odd in odds.values())
        
        # Arbitrage exists if total implied < 1
        if total_implied < 0.99:  # 1% margin for fees
            arb_pct = (1 - total_implied) * 100
            
            if arb_pct >= self.min_edge:
                # Return all outcomes for hedged betting
                return {
                    'outcomes': {
                        outcome: {
                            'odds': odd.decimal_odds,
                            'stake_pct': (1/odd.decimal_odds) / total_implied
                        }
                        for outcome, odd in odds.items()
                    },
                    'edge': arb_pct,
                    'strategy': self.name,
                    'is_arbitrage': True
                }
                
        return None
        
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """For arbitrage, use larger portion of bankroll"""
        # This is risk-free, so we can be more aggressive
        bet_fraction = min(0.2, edge / 100 * 5)  # Up to 20% for strong arbs
        return bankroll * bet_fraction


class EnsembleStrategy(TradingStrategy):
    """Combine multiple strategies with dynamic weighting"""
    
    def __init__(self, strategies: List[TradingStrategy]):
        super().__init__("ensemble", min_edge=2.0)
        self.strategies = strategies
        
    def evaluate_opportunity(self, market: Market, odds: Dict[str, Odd]) -> Optional[Dict]:
        """Combine signals from multiple strategies"""
        signals = []
        
        for strategy in self.strategies:
            signal = strategy.evaluate_opportunity(market, odds)
            if signal:
                signal['weight'] = strategy.weight
                signals.append(signal)
                
        if not signals:
            return None
            
        # Weighted voting
        outcome_scores = {}
        
        for signal in signals:
            outcome = signal.get('outcome', 'multi')  # Handle arb case
            weight = signal['weight']
            edge = signal['edge']
            
            if outcome not in outcome_scores:
                outcome_scores[outcome] = 0
                
            outcome_scores[outcome] += weight * edge
            
        # Get best outcome
        best_outcome = max(outcome_scores, key=outcome_scores.get)
        weighted_edge = outcome_scores[best_outcome] / sum(s['weight'] for s in signals)
        
        if weighted_edge >= self.min_edge:
            # Find the signal for this outcome
            outcome_signals = [s for s in signals if s.get('outcome') == best_outcome]
            if outcome_signals:
                best_signal = max(outcome_signals, key=lambda x: x['edge'])
                best_signal['edge'] = weighted_edge
                best_signal['strategy'] = 'ensemble'
                best_signal['sub_strategies'] = [s['strategy'] for s in signals]
                return best_signal
                
        return None
        
    def calculate_bet_size(self, edge: float, odds: float, bankroll: float) -> float:
        """Conservative sizing for ensemble"""
        bet_fraction = (edge / 100) * 0.3
        bet_amount = bankroll * bet_fraction
        
        return max(10, min(bet_amount, bankroll * 0.05))


# Strategy factory
def create_trading_strategies() -> Dict[str, TradingStrategy]:
    """Create and return all trading strategies"""
    value_strategy = ValueBettingStrategy()
    momentum_strategy = MomentumStrategy()
    timing_strategy = MarketTimingStrategy()
    arb_strategy = ArbitrageStrategy()
    
    # Ensemble combines all strategies
    ensemble_strategy = EnsembleStrategy([
        value_strategy,
        momentum_strategy,
        timing_strategy,
        arb_strategy
    ])
    
    return {
        'value_betting': value_strategy,
        'momentum': momentum_strategy,
        'market_timing': timing_strategy,
        'arbitrage': arb_strategy,
        'ensemble': ensemble_strategy
    }