#!/usr/bin/env python3
"""
Continuous Portfolio Optimizer
Implements time-chunked portfolio optimization with dynamic rebalancing
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass

# Import dynamic chunking components
from dynamic_chunk_manager import DynamicChunkManager, DynamicChunk
from capital_exposure_tracker import CapitalExposureTracker, get_capital_state_for_chunks
from settlement_analyzer import SettlementAnalyzer
from realtime_valuation_engine import RealTimeValuationEngine

logger = logging.getLogger(__name__)

@dataclass
class TimeChunk:
    """Represents a time chunk for grouping matches"""
    chunk_id: str
    start_time: datetime
    end_time: datetime
    markets: List[Dict]
    
    @property
    def label(self):
        """Human-readable label for the chunk"""
        now = datetime.now(timezone.utc)
        hours_away = (self.start_time - now).total_seconds() / 3600
        
        if hours_away < 0:
            return "Live/Settling"
        elif hours_away < 1:
            return "Next Hour"
        elif hours_away < 3:
            return "Next 3 Hours"
        elif hours_away < 6:
            return "Next 6 Hours"
        elif hours_away < 24:
            return "Today"
        else:
            days = int(hours_away / 24)
            return f"In {days} days"

class ContinuousPortfolioOptimizer:
    """Manages continuous portfolio optimization with dynamic time chunks"""
    
    def __init__(self, portfolio_engine, chunk_hours: float = 2.0, use_dynamic_chunking: bool = True):
        """
        Args:
            portfolio_engine: The base portfolio trading engine
            chunk_hours: Size of time chunks in hours (default 2 hours, used only if dynamic chunking disabled)
            use_dynamic_chunking: Whether to use dynamic chunking based on empirical data
        """
        self.portfolio_engine = portfolio_engine
        self.chunk_hours = chunk_hours
        self.use_dynamic_chunking = use_dynamic_chunking
        self.last_portfolio_state = {}
        
        # Initialize dynamic components
        if self.use_dynamic_chunking:
            self.dynamic_chunk_manager = DynamicChunkManager()
            self.capital_tracker = CapitalExposureTracker()
            self.settlement_analyzer = SettlementAnalyzer()
            self.valuation_engine = None  # Will be initialized when needed
            logger.info("Initialized with dynamic chunking and capital tracking")
        else:
            logger.info("Initialized with fixed time chunking")
        
    def group_markets_by_chunks(self, markets: List[Dict], session_id: str = None) -> List[TimeChunk]:
        """Group markets into time-based chunks using dynamic or fixed chunking"""
        if not markets:
            return []
        
        # Use dynamic chunking if enabled and session provided
        if self.use_dynamic_chunking and session_id:
            return self._create_dynamic_chunks(markets, session_id)
        else:
            return self._create_fixed_chunks(markets)
    
    def _create_dynamic_chunks(self, markets: List[Dict], session_id: str) -> List[TimeChunk]:
        """Create dynamic chunks using empirical data and capital state"""
        try:
            # Get current capital state
            capital_state = get_capital_state_for_chunks(session_id)
            
            # Create dynamic chunks
            dynamic_chunks = self.dynamic_chunk_manager.create_dynamic_chunks(
                markets, capital_state
            )
            
            # Convert to TimeChunk format for compatibility
            time_chunks = []
            for dc in dynamic_chunks:
                time_chunk = TimeChunk(
                    chunk_id=dc.chunk_id,
                    start_time=dc.start_time,
                    end_time=dc.end_time,
                    markets=dc.matches
                )
                time_chunks.append(time_chunk)
            
            logger.info(f"Created {len(time_chunks)} dynamic chunks based on empirical data")
            return time_chunks
            
        except Exception as e:
            logger.error(f"Error creating dynamic chunks, falling back to fixed: {e}")
            return self._create_fixed_chunks(markets)
    
    def _create_fixed_chunks(self, markets: List[Dict]) -> List[TimeChunk]:
        """Create fixed time-based chunks (original logic)"""
        now = datetime.now(timezone.utc)
        chunks = {}
        
        for market in markets:
            # Get maturity date
            maturity = market.get('maturity_date')
            if isinstance(maturity, str):
                maturity = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
            
            # Ensure timezone aware
            if maturity.tzinfo is None:
                maturity = maturity.replace(tzinfo=timezone.utc)
            
            # Skip if already started
            if maturity < now:
                continue
            
            # Calculate chunk start time (floor to chunk_hours)
            hours_away = (maturity - now).total_seconds() / 3600
            chunk_start_hours = int(hours_away / self.chunk_hours) * self.chunk_hours
            chunk_start = now + timedelta(hours=chunk_start_hours)
            chunk_end = chunk_start + timedelta(hours=self.chunk_hours)
            
            chunk_id = f"chunk_{int(chunk_start_hours/self.chunk_hours)}"
            
            if chunk_id not in chunks:
                chunks[chunk_id] = TimeChunk(
                    chunk_id=chunk_id,
                    start_time=chunk_start,
                    end_time=chunk_end,
                    markets=[]
                )
            
            chunks[chunk_id].markets.append(market)
        
        # Sort chunks by time
        sorted_chunks = sorted(chunks.values(), key=lambda x: x.start_time)
        
        logger.info(f"Grouped {len(markets)} markets into {len(sorted_chunks)} time chunks")
        for chunk in sorted_chunks:
            logger.info(f"  {chunk.label}: {len(chunk.markets)} markets")
        
        return sorted_chunks
    
    def calculate_chunk_portfolio(self, chunk: TimeChunk, signals: List[Dict], 
                                 available_capital: float, max_positions: int, 
                                 capital_state: Dict = None) -> pd.DataFrame:
        """Calculate optimal portfolio for a single time chunk with risk management"""
        if not chunk.markets:
            return pd.DataFrame()
        
        # Apply capital exposure limits if available
        if capital_state and self.use_dynamic_chunking:
            risk_check = self.capital_tracker.check_risk_limits(
                self.capital_tracker.capital_history[-1] if self.capital_tracker.capital_history else None,
                available_capital
            )
            
            if not risk_check['approved']:
                logger.warning(f"Risk limits prevent chunk optimization: {risk_check['violations']}")
                return pd.DataFrame()
            
            # Adjust available capital based on risk limits
            available_capital = min(available_capital, risk_check['available_cash'])
        
        # Get signals for this chunk's markets
        market_ids = [m['market_id'] for m in chunk.markets]
        chunk_signals = [s for s in signals if s.get('market_id') in market_ids]
        
        # Use base engine to calculate optimal stakes
        markets_df = self.portfolio_engine.prepare_markets_for_kelly(chunk.markets, chunk_signals)
        
        if markets_df.empty:
            return pd.DataFrame()
        
        # Calculate target portfolio with chunk-specific capital
        target_portfolio = self.portfolio_engine.calculate_target_portfolio(
            markets_df, available_capital
        )
        
        # Apply position limit for this chunk
        if len(target_portfolio) > max_positions:
            # Sort by expected value (edge * odds)
            target_portfolio['expected_value'] = target_portfolio['edge'] * target_portfolio['odds']
            target_portfolio = target_portfolio.nlargest(max_positions, 'expected_value')
        
        return target_portfolio
    
    def calculate_rebalancing_trades(self, current_positions: pd.DataFrame, 
                                   target_portfolio: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Calculate trades needed to rebalance from current to target portfolio"""
        trades = {
            'close': [],     # Positions to close completely
            'reduce': [],    # Positions to reduce
            'increase': [],  # Positions to increase
            'open': []       # New positions to open
        }
        
        if current_positions.empty and target_portfolio.empty:
            return trades
        
        # Create position keys for matching
        if not current_positions.empty:
            current_positions['position_key'] = (
                current_positions['match_id'].astype(str) + '_' + 
                current_positions['bet_on'].astype(str)
            )
            current_dict = current_positions.set_index('position_key').to_dict('index')
        else:
            current_dict = {}
        
        if not target_portfolio.empty:
            # Check if we have 'outcome' or 'normalized_outcome' column
            outcome_col = 'outcome' if 'outcome' in target_portfolio.columns else 'normalized_outcome'
            target_portfolio['position_key'] = (
                target_portfolio['match_id'].astype(str) + '_' + 
                target_portfolio[outcome_col].astype(str)
            )
            target_dict = target_portfolio.set_index('position_key').to_dict('index')
        else:
            target_dict = {}
        
        # Check positions to close or adjust
        for pos_key, current in current_dict.items():
            if pos_key not in target_dict:
                # Position should be closed
                trades['close'].append({
                    'position_key': pos_key,
                    'bet_id': current.get('bet_id'),
                    'match_id': current['match_id'],
                    'bet_on': current['bet_on'],
                    'current_stake': float(current['stake']),
                    'reason': 'not_in_optimal_portfolio'
                })
            else:
                # Check if position size should be adjusted
                target = target_dict[pos_key]
                current_stake = float(current['stake'])
                target_stake = float(target.get('stake', 0))
                
                # Only adjust if difference is significant (>10% or >$10)
                stake_diff = target_stake - current_stake
                if abs(stake_diff) > max(current_stake * 0.1, 10):
                    if stake_diff > 0:
                        trades['increase'].append({
                            'position_key': pos_key,
                            'bet_id': current.get('bet_id'),
                            'match_id': current['match_id'],
                            'bet_on': current['bet_on'],
                            'current_stake': current_stake,
                            'target_stake': target_stake,
                            'increase_amount': stake_diff,
                            'reason': 'rebalance_increase'
                        })
                    else:
                        trades['reduce'].append({
                            'position_key': pos_key,
                            'bet_id': current.get('bet_id'),
                            'match_id': current['match_id'],
                            'bet_on': current['bet_on'],
                            'current_stake': current_stake,
                            'target_stake': target_stake,
                            'reduce_amount': -stake_diff,
                            'reason': 'rebalance_reduce'
                        })
        
        # Check for new positions to open
        for pos_key, target in target_dict.items():
            if pos_key not in current_dict and target.get('stake', 0) > 0:
                # Get outcome column name
                outcome_val = target.get('outcome', target.get('normalized_outcome', ''))
                trades['open'].append({
                    'match_id': target.get('match_id', ''),
                    'sport': target.get('sport', 'Unknown'),
                    'home_team': target.get('home_team', 'Unknown'),
                    'away_team': target.get('away_team', 'Unknown'),
                    'bet_on': outcome_val,
                    'odds': float(target.get('odds', 0)),
                    'stake': float(target.get('stake', 0)),
                    'edge': float(target.get('edge', 0)),
                    'signal_name': 'portfolio_optimizer',
                    'signal_value': float(target.get('probability', 0)),
                    'reason': 'new_optimal_position',
                    'chunk_id': target.get('chunk_id', ''),
                    'chunk_label': target.get('chunk_label', 'Unknown')
                })
        
        # Log rebalancing summary
        total_trades = sum(len(v) for v in trades.values())
        if total_trades > 0:
            logger.info(f"Rebalancing trades: {len(trades['close'])} close, "
                       f"{len(trades['reduce'])} reduce, {len(trades['increase'])} increase, "
                       f"{len(trades['open'])} open")
        
        return trades
    
    def optimize_portfolio_continuously(self, session_id: str, markets: List[Dict], 
                                      signals: List[Dict], current_bankroll: float, 
                                      enable_real_time_valuation: bool = True) -> Dict[str, Any]:
        """Main method for continuous portfolio optimization with dynamic chunking"""
        try:
            # Update capital state if using dynamic chunking
            if self.use_dynamic_chunking:
                capital_state = self.capital_tracker.update_capital_state(session_id)
                available_capital = capital_state.available_cash
                
                # Start real-time valuation if enabled and not already running
                if enable_real_time_valuation and not self.valuation_engine:
                    self.valuation_engine = RealTimeValuationEngine()
                    # Note: In production, would start this as async task
                    logger.info("Real-time valuation engine initialized")
            else:
                # Get current positions (legacy method)
                current_positions = self.portfolio_engine.get_current_portfolio(session_id)
                current_exposure = current_positions['stake'].sum() if not current_positions.empty else 0
                available_capital = current_bankroll - current_exposure
                capital_state = None
            
            # Get current positions for rebalancing
            current_positions = self.portfolio_engine.get_current_portfolio(session_id)
            
            # Group markets by time chunks (dynamic or fixed)
            chunks = self.group_markets_by_chunks(markets, session_id)
            
            if not chunks:
                logger.info("No upcoming markets to trade")
                return {'success': True, 'trades': [], 'message': 'No markets available'}
            
            # Focus on nearest chunks with dynamic time horizon
            if self.use_dynamic_chunking:
                # Use settlement analyzer to determine optimal time horizon
                max_hours_ahead = self._get_optimal_time_horizon(chunks)
            else:
                max_hours_ahead = 6  # Default 6 hours
            
            active_chunks = [c for c in chunks if (c.start_time - datetime.now(timezone.utc)).total_seconds() / 3600 < max_hours_ahead]
            
            if not active_chunks:
                active_chunks = chunks[:1]  # At least consider the nearest chunk
            
            logger.info(f"Optimizing across {len(active_chunks)} active time chunks")
            
            # Calculate capital allocation per chunk
            chunk_capital = available_capital / len(active_chunks) if active_chunks else 0
            max_positions = self.portfolio_engine.strategy_config.get('max_positions', 20)
            positions_per_chunk = max(1, max_positions // len(active_chunks))
            
            all_trades = []
            target_positions = pd.DataFrame()
            
            # Optimize each chunk with capital state awareness
            for chunk in active_chunks:
                chunk_target = self.calculate_chunk_portfolio(
                    chunk, signals, chunk_capital, positions_per_chunk, capital_state
                )
                
                if not chunk_target.empty:
                    chunk_target['chunk_id'] = chunk.chunk_id
                    chunk_target['chunk_label'] = chunk.label
                    target_positions = pd.concat([target_positions, chunk_target], ignore_index=True)
            
            # Calculate rebalancing trades
            rebalancing = self.calculate_rebalancing_trades(current_positions, target_positions)
            
            # Execute trades based on strategy settings
            min_edge = self.portfolio_engine.strategy_config.get('min_edge', 0.02)
            min_bet = self.portfolio_engine.strategy_config.get('min_bet', 10)
            
            # Close positions (always execute to free up capital)
            if rebalancing['close']:
                logger.info(f"Closing {len(rebalancing['close'])} positions")
                # Note: In real implementation, would need to execute close orders
            
            # Open new positions
            new_positions = []
            for trade in rebalancing['open']:
                if trade['edge'] >= min_edge and trade['stake'] >= min_bet:
                    new_positions.append(trade)
            
            # Apply position limits
            current_count = len(current_positions) - len(rebalancing['close'])
            positions_available = max(0, max_positions - current_count)
            
            if len(new_positions) > positions_available:
                new_positions = sorted(new_positions, key=lambda x: x['edge'], reverse=True)[:positions_available]
            
            # Record new trades
            if new_positions:
                self.portfolio_engine.session_manager.record_trades(session_id, new_positions)
                all_trades.extend(new_positions)
            
            # Enhanced summary with dynamic chunking metrics
            summary = {
                'success': True,
                'trades': all_trades,
                'chunks_analyzed': len(active_chunks),
                'total_markets': sum(len(c.markets) for c in active_chunks),
                'positions_closed': len(rebalancing['close']),
                'positions_opened': len(new_positions),
                'positions_adjusted': len(rebalancing['reduce']) + len(rebalancing['increase']),
                'current_exposure': getattr(capital_state, 'in_play_exposure', 0) if capital_state else (current_positions['stake'].sum() if not current_positions.empty else 0),
                'target_exposure': target_positions['stake'].sum() if not target_positions.empty else 0,
                'message': f"Portfolio optimized across {len(active_chunks)} {'dynamic' if self.use_dynamic_chunking else 'fixed'} chunks",
                'chunking_method': 'dynamic' if self.use_dynamic_chunking else 'fixed',
                'time_horizon_hours': max_hours_ahead
            }
            
            # Add capital tracking metrics if available
            if capital_state:
                summary.update({
                    'capital_metrics': {
                        'available_cash': capital_state.available_cash,
                        'utilization_rate': capital_state.utilization_rate,
                        'value_at_risk': capital_state.value_at_risk,
                        'expected_value': capital_state.expected_value
                    }
                })
            
            # Store portfolio state for comparison
            self.last_portfolio_state = {
                'timestamp': datetime.now(timezone.utc),
                'target_positions': target_positions,
                'chunks': active_chunks
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in continuous portfolio optimization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e), 'trades': []}
    
    def _get_optimal_time_horizon(self, chunks: List[TimeChunk]) -> float:
        """Determine optimal time horizon based on settlement patterns"""
        try:
            if not chunks:
                return 6.0  # Default
            
            # Use settlement analyzer to get recommendations
            # For now, use adaptive horizon based on chunk count and timing
            if len(chunks) <= 2:
                return 12.0  # Look further ahead when few chunks
            elif len(chunks) <= 5:
                return 8.0   # Medium horizon
            else:
                return 6.0   # Standard horizon when many chunks
                
        except Exception as e:
            logger.error(f"Error calculating optimal time horizon: {e}")
            return 6.0
    
    def should_rebalance(self, current_positions: pd.DataFrame, 
                        market_prices: Dict[str, float]) -> bool:
        """Determine if portfolio should be rebalanced based on price changes"""
        if current_positions.empty:
            return True
        
        # Check if any position's odds have changed significantly (>5%)
        for _, pos in current_positions.iterrows():
            pos_key = f"{pos['match_id']}_{pos['bet_on']}"
            current_odds = pos['odds']
            new_odds = market_prices.get(pos_key, current_odds)
            
            if abs(new_odds - current_odds) / current_odds > 0.05:
                logger.info(f"Significant price change detected for {pos_key}: {current_odds:.2f} -> {new_odds:.2f}")
                return True
        
        # Check if it's been more than 5 minutes since last optimization
        if hasattr(self, 'last_portfolio_state'):
            last_time = self.last_portfolio_state.get('timestamp')
            if last_time and (datetime.now(timezone.utc) - last_time).total_seconds() > 300:
                return True
        
        return False
    
    async def optimize_and_execute_async(self, session_id: str, markets: List[Dict], signals: List[Dict]) -> Dict:
        """Async version of optimize_and_execute"""
        return self.optimize_and_execute(session_id, markets, signals)
    
    def optimize_and_execute(self, session_id: str, markets: List[Dict], signals: List[Dict]) -> Dict:
        """Main entry point for continuous optimization with execution"""
        session = self.portfolio_engine.session_manager.get_session(session_id)
        if not session:
            return {'success': False, 'error': 'Session not found'}
        
        current_bankroll = float(session['current_bankroll'])
        
        # Run continuous optimization
        result = self.optimize_portfolio_continuously(
            session_id, 
            markets, 
            signals, 
            current_bankroll
        )
        
        if result['success']:
            # Get updated positions summary
            positions = self.portfolio_engine.session_manager.get_positions(session_id)
            open_positions = [p for p in positions if p['status'] == 'pending']
            
            result['total_positions'] = len(open_positions)
            result['positions_value'] = sum(float(p['stake']) for p in open_positions)
            result['cash_balance'] = current_bankroll - result['positions_value']
            result['new_positions'] = len(result.get('trades', []))
            result['rebalanced_positions'] = result.get('positions_adjusted', 0)
            result['closed_positions'] = result.get('positions_closed', 0)
        
        return result
    
    def get_dynamic_chunking_status(self) -> Dict[str, Any]:
        """Get status of dynamic chunking components"""
        if not self.use_dynamic_chunking:
            return {'enabled': False}
        
        status = {
            'enabled': True,
            'chunk_manager': {
                'empirical_data_loaded': len(self.dynamic_chunk_manager.empirical_data) > 0,
                'last_update': getattr(self.dynamic_chunk_manager, 'last_update', None)
            },
            'capital_tracker': {
                'positions_tracked': len(self.capital_tracker.position_valuations),
                'capital_history_points': len(self.capital_tracker.capital_history)
            },
            'settlement_analyzer': {
                'league_stats_available': len(self.settlement_analyzer.league_stats),
                'last_analysis': self.settlement_analyzer.last_update
            },
            'valuation_engine': {
                'active': self.valuation_engine is not None,
                'live_odds_tracked': len(self.valuation_engine.live_odds) if self.valuation_engine else 0,
                'live_scores_tracked': len(self.valuation_engine.live_scores) if self.valuation_engine else 0
            }
        }
        
        return status
    
    async def update_empirical_data(self, lookback_days: int = 30) -> bool:
        """Update empirical settlement data for improved chunking"""
        if not self.use_dynamic_chunking:
            return False
        
        try:
            # Update settlement patterns
            league_stats = self.settlement_analyzer.analyze_historical_patterns(lookback_days)
            
            if league_stats:
                logger.info(f"Updated empirical data with {len(league_stats)} league patterns")
                return True
            else:
                logger.warning("No empirical data available for update")
                return False
                
        except Exception as e:
            logger.error(f"Error updating empirical data: {e}")
            return False