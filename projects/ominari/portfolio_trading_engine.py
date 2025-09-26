#!/usr/bin/env python3
"""Portfolio-based trading engine using Kelly optimization"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime, timezone

# Import portfolio optimization functions
try:
    from kelly_multimarket import (
        calculate_kelly_stakes_with_exclusivity, 
        infer_exclusivity_groups
    )
except ImportError:
    # Fallback if module structure is different
    from .kelly_multimarket import (
        calculate_kelly_stakes_with_exclusivity, 
        infer_exclusivity_groups
    )

try:
    from evaluate_open_markets import (
        build_structural_correlation_matrix,
        trim_kelly_results
    )
except ImportError:
    # If these functions don't exist, we'll implement minimal versions
    def build_structural_correlation_matrix(markets_df):
        """Build correlation matrix based on market structure"""
        n = len(markets_df)
        corr_matrix = np.eye(n)
        
        # Add correlation for same match different outcomes
        for i in range(n):
            for j in range(i+1, n):
                if markets_df.iloc[i]['match_id'] == markets_df.iloc[j]['match_id']:
                    # Same match, different outcomes are negatively correlated
                    corr_matrix[i, j] = -0.5
                    corr_matrix[j, i] = -0.5
        
        return corr_matrix
    
    def trim_kelly_results(kelly_df, kelly_fraction=0.25, bankroll=1000, 
                          cap_per_game=0.02, cap_per_bet=0.01, **kwargs):
        """Apply position limits to Kelly results"""
        df = kelly_df.copy()
        
        # Apply Kelly fraction
        if 'stake' not in df.columns and 'stake_fraction' in df.columns:
            df['stake'] = df['stake_fraction'] * bankroll * kelly_fraction
        else:
            df['stake'] = df.get('stake', 0) * kelly_fraction
            
        # Apply per-bet cap
        max_bet = bankroll * cap_per_bet
        df['stake'] = df['stake'].clip(upper=max_bet)
        
        # Apply per-game cap
        game_stakes = df.groupby('match_id')['stake'].sum()
        max_per_game = bankroll * cap_per_game
        
        for match_id in game_stakes[game_stakes > max_per_game].index:
            mask = df['match_id'] == match_id
            scale = max_per_game / game_stakes[match_id]
            df.loc[mask, 'stake'] *= scale
        
        return df

logger = logging.getLogger(__name__)

class PortfolioTradingEngine:
    """Manages portfolio-based trading using Kelly optimization"""
    
    def __init__(self, session_manager, edge_calculator, strategy_config):
        self.session_manager = session_manager
        self.edge_calculator = edge_calculator
        self.strategy_config = strategy_config
        
    def get_current_portfolio(self, session_id: str) -> pd.DataFrame:
        """Get current portfolio as DataFrame"""
        positions = self.session_manager.get_positions(session_id)
        open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
        
        if not open_positions:
            return pd.DataFrame()
            
        df = pd.DataFrame(open_positions)
        df['position_key'] = df['match_id'] + '_' + df['bet_on']
        return df
    
    def prepare_markets_for_kelly(self, markets: List[Dict], signals: List[Dict]) -> pd.DataFrame:
        """Convert markets and signals to format needed for Kelly optimization"""
        rows = []
        
        for i, market in enumerate(markets):
            signal = signals[i]
            market_id = market.get('market_id', market.get('source_id', ''))
            
            # Create rows for each outcome
            for outcome in ['home', 'draw', 'away']:
                edge = signal.get(f'{outcome}_edge', 0)
                odds = signal.get(f'{outcome}_odds', 0)
                
                # Only include positive edge bets or bets needed for portfolio balance
                if odds > 1:  # Valid odds
                    probability = signal.get(f'{outcome}_implied_prob', 1.0 / odds if odds > 0 else 0)
                    
                    # Adjust probability based on edge
                    # edge = (probability * odds - 1) * 100
                    # So: probability = (edge/100 + 1) / odds
                    if edge != 0:
                        adjusted_prob = (edge / 100 + 1) / odds
                        adjusted_prob = max(0.01, min(0.99, adjusted_prob))  # Clamp to valid range
                    else:
                        adjusted_prob = probability
                    
                    rows.append({
                        'source_id': market_id,
                        'match_id': market_id,
                        'sport': market.get('sport', 'Soccer'),
                        'home_team': market.get('home_team', ''),
                        'away_team': market.get('away_team', ''),
                        'unified_market_type': 'winner',
                        'normalized_line': 0,
                        'bet_name': f"{market.get('home_team', '')} vs {market.get('away_team', '')} - {outcome}",
                        'outcome': outcome,
                        'odds': odds,
                        'probability': adjusted_prob,
                        'edge': edge,
                        'confidence': signal.get(f'{outcome}_confidence', 0.5),
                        'maturity_date': market.get('maturity_date'),
                        'position_key': f"{market_id}_{outcome}"
                    })
        
        return pd.DataFrame(rows)
    
    def calculate_target_portfolio(self, markets_df: pd.DataFrame, bankroll: float) -> pd.DataFrame:
        """Calculate optimal portfolio using Kelly criterion"""
        if markets_df.empty:
            return pd.DataFrame()
            
        # Build correlation matrix
        correlation_matrix = build_structural_correlation_matrix(markets_df)
        
        # Run Kelly optimization
        kelly_results = calculate_kelly_stakes_with_exclusivity(
            markets_df,
            bankroll=bankroll,
            correlation_matrix=correlation_matrix,
            risk_adjusted=True
        )
        
        # Apply position limits
        trimmed = trim_kelly_results(
            kelly_results,
            kelly_fraction=self.strategy_config['kelly_fraction'],
            bankroll=bankroll,
            cap_per_game=self.strategy_config['cap_per_game'],
            cap_per_bet=self.strategy_config['cap_per_bet'],
            cap_per_game_market=self.strategy_config.get('cap_per_game_market', 0.005),
            min_bet_abs=self.strategy_config['min_bet'],
            min_bet_pct=self.strategy_config.get('min_bet_pct', 0.001),
            min_break_minutes=self.strategy_config.get('min_break_minutes', 240)
        )
        
        # Only keep positions with stake > 0
        trimmed = trimmed[trimmed['stake'] > 0].copy()
        
        return trimmed
    
    def calculate_portfolio_diff(self, current_portfolio: pd.DataFrame, 
                               target_portfolio: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Calculate differences between current and target portfolios"""
        
        # Get position keys
        current_positions = set(current_portfolio['position_key'].tolist()) if not current_portfolio.empty else set()
        target_positions = set(target_portfolio['position_key'].tolist()) if not target_portfolio.empty else set()
        
        trades_to_execute = []
        positions_to_close = []
        positions_to_adjust = []
        
        # Find new positions to open
        new_positions = target_positions - current_positions
        for pos_key in new_positions:
            target_row = target_portfolio[target_portfolio['position_key'] == pos_key].iloc[0]
            trades_to_execute.append({
                'action': 'open',
                'match_id': target_row['match_id'],
                'sport': target_row.get('sport', 'Soccer'),
                'home_team': target_row.get('home_team', ''),
                'away_team': target_row.get('away_team', ''),
                'bet_type': 'moneyline',
                'bet_on': target_row['outcome'],
                'odds': target_row['odds'],
                'stake': target_row['stake'],
                'signal_name': 'portfolio_optimizer',
                'signal_value': target_row['probability'],
                'edge': target_row.get('edge', 0),
                'kickoff_time': target_row.get('maturity_date')
            })
        
        # Find positions to close
        positions_to_remove = current_positions - target_positions
        for pos_key in positions_to_remove:
            current_row = current_portfolio[current_portfolio['position_key'] == pos_key].iloc[0]
            positions_to_close.append({
                'position_id': current_row.get('position_id', current_row.get('bet_id')),
                'match_id': current_row['match_id'],
                'bet_on': current_row['bet_on'],
                'stake': current_row['stake'],
                'reason': 'portfolio_rebalance'
            })
        
        # Find positions to adjust (stake changes)
        common_positions = current_positions & target_positions
        for pos_key in common_positions:
            current_row = current_portfolio[current_portfolio['position_key'] == pos_key].iloc[0]
            target_row = target_portfolio[target_portfolio['position_key'] == pos_key].iloc[0]
            
            current_stake = float(current_row['stake'])
            target_stake = float(target_row['stake'])
            
            # Only adjust if difference is significant (>10% or >$10)
            stake_diff = abs(target_stake - current_stake)
            if stake_diff > max(10, current_stake * 0.1):
                positions_to_adjust.append({
                    'position_id': current_row.get('position_id', current_row.get('bet_id')),
                    'match_id': current_row['match_id'],
                    'bet_on': current_row['bet_on'],
                    'current_stake': current_stake,
                    'target_stake': target_stake,
                    'adjustment': target_stake - current_stake
                })
        
        return {
            'trades_to_execute': trades_to_execute,
            'positions_to_close': positions_to_close,
            'positions_to_adjust': positions_to_adjust
        }
    
    def execute_portfolio_trades(self, session_id: str, markets: List[Dict], 
                               signals: List[Dict], current_bankroll: float) -> Dict[str, Any]:
        """Execute portfolio-based trades"""
        try:
            # Get current portfolio
            current_portfolio = self.get_current_portfolio(session_id)
            
            # Check current exposure
            current_exposure = current_portfolio['stake'].sum() if not current_portfolio.empty else 0
            exposure_pct = (current_exposure / current_bankroll * 100) if current_bankroll > 0 else 0
            
            logger.info(f"Current portfolio: {len(current_portfolio)} positions, ${current_exposure:.2f} exposure ({exposure_pct:.1f}%)")
            
            # Prepare markets for Kelly optimization
            markets_df = self.prepare_markets_for_kelly(markets, signals)
            
            # Calculate target portfolio
            available_capital = current_bankroll - current_exposure
            target_portfolio = self.calculate_target_portfolio(markets_df, current_bankroll)
            
            # Calculate portfolio differences
            portfolio_diff = self.calculate_portfolio_diff(current_portfolio, target_portfolio)
            
            # Log portfolio changes
            logger.info(f"Portfolio changes: {len(portfolio_diff['trades_to_execute'])} new, "
                       f"{len(portfolio_diff['positions_to_close'])} close, "
                       f"{len(portfolio_diff['positions_to_adjust'])} adjust")
            
            # Apply exposure limits before executing
            new_trades = portfolio_diff['trades_to_execute']
            total_new_exposure = sum(t['stake'] for t in new_trades)
            
            MAX_EXPOSURE_PCT = 30  # Maximum 30% exposure
            new_exposure_pct = ((current_exposure + total_new_exposure) / current_bankroll * 100)
            
            if new_exposure_pct > MAX_EXPOSURE_PCT:
                # Scale down new trades proportionally
                scale_factor = (MAX_EXPOSURE_PCT * current_bankroll / 100 - current_exposure) / total_new_exposure
                scale_factor = max(0, scale_factor)
                
                for trade in new_trades:
                    trade['stake'] *= scale_factor
                    
                # Remove trades below minimum
                new_trades = [t for t in new_trades if t['stake'] >= self.strategy_config['min_bet']]
                
                logger.warning(f"Scaled down trades by {scale_factor:.2f} to stay within {MAX_EXPOSURE_PCT}% exposure limit")
            
            # Execute trades if any
            if new_trades:
                self.session_manager.record_trades(session_id, new_trades)
            
            # Return summary
            total_trades = len(new_trades)
            total_stake = sum(t['stake'] for t in new_trades)
            
            return {
                'success': True,
                'trades_made': total_trades,
                'total_stake': total_stake,
                'positions_closed': len(portfolio_diff['positions_to_close']),
                'positions_adjusted': len(portfolio_diff['positions_to_adjust']),
                'current_exposure': current_exposure,
                'new_exposure': current_exposure + total_stake,
                'portfolio_size': len(target_portfolio)
            }
            
        except Exception as e:
            logger.error(f"Error in portfolio trading: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}