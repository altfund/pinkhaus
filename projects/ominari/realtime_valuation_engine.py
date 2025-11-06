#!/usr/bin/env python3
"""
Real-Time Valuation Engine - Live Position Tracking

Provides real-time expected value tracking with live odds and scores:
- Live odds monitoring and comparison
- Score-based probability updates
- Real-time position valuation
- Expected value calculations
- Performance tracking vs initial expectations

Integrates with blockchain events and API data to maintain accurate position values.
"""

import os
import json
import logging
import asyncio
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

logger = logging.getLogger(__name__)

@dataclass
class LiveOdds:
    """Current market odds"""
    market_id: str
    timestamp: datetime
    
    # Odds by position
    home_odds: Optional[float] = None
    away_odds: Optional[float] = None
    draw_odds: Optional[float] = None
    
    # Metadata
    source: str = 'api'  # api, blockchain, manual
    volume: float = 0.0
    
    @property
    def implied_probabilities(self) -> Dict[str, float]:
        """Calculate implied probabilities from odds"""
        probs = {}
        total_inverse = 0.0
        
        if self.home_odds and self.home_odds > 1:
            probs['home'] = 1.0 / self.home_odds
            total_inverse += probs['home']
        
        if self.away_odds and self.away_odds > 1:
            probs['away'] = 1.0 / self.away_odds
            total_inverse += probs['away']
            
        if self.draw_odds and self.draw_odds > 1:
            probs['draw'] = 1.0 / self.draw_odds
            total_inverse += probs['draw']
        
        # Normalize to remove bookmaker margin
        if total_inverse > 0:
            for key in probs:
                probs[key] = probs[key] / total_inverse
        
        return probs

@dataclass
class LiveScore:
    """Current match score and status"""
    match_id: str
    timestamp: datetime
    
    # Score
    home_score: int = 0
    away_score: int = 0
    
    # Match status
    status: str = 'not_started'  # not_started, live, halftime, finished, cancelled
    minute: int = 0
    period: str = '1H'  # 1H, 2H, ET, PEN
    
    # Additional context
    red_cards_home: int = 0
    red_cards_away: int = 0
    
    @property
    def current_result(self) -> str:
        """Current winning position"""
        if self.home_score > self.away_score:
            return 'home'
        elif self.away_score > self.home_score:
            return 'away'
        else:
            return 'draw'
    
    @property
    def is_active(self) -> bool:
        """Whether match is currently active"""
        return self.status in ['live', 'halftime']

@dataclass
class PositionValuation:
    """Real-time position value"""
    bet_id: str
    market_id: str
    
    # Position details
    stake: float
    original_odds: float
    bet_on: str  # home, away, draw
    bet_type: str
    
    # Current market state
    current_odds: Optional[float] = None
    current_probability: Optional[float] = None
    live_score: Optional[LiveScore] = None
    
    # Valuation
    current_expected_value: float = 0.0
    mark_to_market_value: float = 0.0
    probability_of_win: float = 0.0
    
    # Changes from initial
    odds_movement: float = 0.0  # Current odds / Original odds
    ev_change: float = 0.0      # Change in expected value
    
    def calculate_valuation(self):
        """Calculate current position value"""
        if not self.current_probability:
            return
        
        # Expected value calculation
        if self.current_probability > 0:
            win_pnl = (self.original_odds - 1) * self.stake
            lose_pnl = -self.stake
            self.current_expected_value = (self.current_probability * win_pnl) + ((1 - self.current_probability) * lose_pnl)
        
        # Mark-to-market using current odds
        if self.current_odds and self.current_odds > 1:
            # Value = stake * (current_odds / original_odds)
            self.mark_to_market_value = self.stake * (self.current_odds / self.original_odds)
        else:
            self.mark_to_market_value = self.stake
        
        # Track changes
        if self.current_odds:
            self.odds_movement = self.current_odds / self.original_odds
        
        # Probability of winning (adjust based on live score if available)
        self.probability_of_win = self._calculate_win_probability()
    
    def _calculate_win_probability(self) -> float:
        """Calculate probability of this position winning"""
        if not self.current_probability:
            return 0.0
        
        base_probability = self.current_probability
        
        # Adjust based on live score if available
        if self.live_score and self.live_score.is_active:
            score_adjustment = self._get_score_adjustment()
            # Blend score-based adjustment with market probability
            adjusted_probability = (0.7 * base_probability) + (0.3 * score_adjustment)
            return max(0.0, min(1.0, adjusted_probability))
        
        return base_probability
    
    def _get_score_adjustment(self) -> float:
        """Get probability adjustment based on current score"""
        if not self.live_score:
            return self.current_probability or 0.0
        
        current_result = self.live_score.current_result
        
        # If our bet matches current result, probability increases
        if self.bet_on == current_result:
            # Higher probability if winning, adjust by time remaining
            minutes_remaining = max(0, 90 - self.live_score.minute)
            if minutes_remaining > 60:
                return 0.7  # Early lead, decent chance
            elif minutes_remaining > 30:
                return 0.8  # Mid-game lead
            elif minutes_remaining > 10:
                return 0.9  # Late lead, very likely
            else:
                return 0.95  # Almost certain
        else:
            # Lower probability if not winning
            minutes_remaining = max(0, 90 - self.live_score.minute)
            if minutes_remaining > 60:
                return 0.4  # Still time to come back
            elif minutes_remaining > 30:
                return 0.25  # Getting harder
            elif minutes_remaining > 10:
                return 0.1   # Very difficult
            else:
                return 0.05  # Almost impossible
        
        return self.current_probability or 0.0


class RealTimeValuationEngine:
    """Manages real-time position valuation and tracking"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {
            'host': os.environ.get('PG_HOST', 'localhost'),
            'port': os.environ.get('PG_PORT', '5999'),
            'user': os.environ.get('PG_USER', 'ominari_user'),
            'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.environ.get('PG_DB', 'ominari_production')
        }
        
        # Live data caches
        self.live_odds: Dict[str, LiveOdds] = {}
        self.live_scores: Dict[str, LiveScore] = {}
        self.position_valuations: Dict[str, PositionValuation] = {}
        
        # Data sources
        self.odds_sources = []
        self.score_sources = []
        
        # Configuration
        self.config = {
            'update_interval_seconds': 30,    # How often to update valuations
            'odds_staleness_minutes': 5,      # When odds are considered stale
            'score_staleness_minutes': 2,     # When scores are considered stale
            'min_odds_movement': 0.05,        # Minimum odds change to log
            'valuation_history_hours': 24     # How long to keep valuation history
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_positions_tracked': 0,
            'avg_ev_accuracy': 0.0,
            'odds_prediction_accuracy': 0.0,
            'live_updates_per_minute': 0.0
        }
        
        # Event handlers
        self.valuation_handlers: List[Callable] = []
        
        # Background tasks
        self._update_task = None
        self._running = False
    
    def register_valuation_handler(self, handler: Callable[[Dict], None]):
        """Register handler for valuation updates"""
        self.valuation_handlers.append(handler)
    
    async def start_live_tracking(self, session_id: str):
        """Start real-time position tracking"""
        self._running = True
        logger.info(f"Starting live position tracking for session {session_id}")
        
        # Load active positions
        await self._load_active_positions(session_id)
        
        # Start update loop
        self._update_task = asyncio.create_task(self._update_loop())
        
        logger.info(f"Live tracking started with {len(self.position_valuations)} positions")
    
    async def stop_live_tracking(self):
        """Stop real-time tracking"""
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Live position tracking stopped")
    
    async def _load_active_positions(self, session_id: str):
        """Load active positions from database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            p.bet_id, p.match_id, p.stake, p.odds, p.bet_on, p.bet_type,
                            p.placed_at, p.strategy_config
                        FROM paper_trading_positions p
                        WHERE p.session_id = %s 
                        AND p.status IN ('pending', 'open')
                        ORDER BY p.placed_at
                    """, (session_id,))
                    
                    positions = cur.fetchall()
                    
                    for pos in positions:
                        bet_id, match_id, stake, odds, bet_on, bet_type, placed_at, strategy_config = pos
                        
                        valuation = PositionValuation(
                            bet_id=bet_id,
                            market_id=match_id,
                            stake=float(stake),
                            original_odds=float(odds),
                            bet_on=bet_on,
                            bet_type=bet_type
                        )
                        
                        self.position_valuations[bet_id] = valuation
                        
                        logger.debug(f"Loaded position {bet_id}: {bet_on} @ {odds} for ${stake}")
                    
                    self.performance_metrics['total_positions_tracked'] = len(self.position_valuations)
                    
        except Exception as e:
            logger.error(f"Error loading active positions: {e}")
    
    async def _update_loop(self):
        """Main update loop for live tracking"""
        logger.info("Starting valuation update loop")
        
        while self._running:
            try:
                # Update live odds and scores
                await self._update_market_data()
                
                # Update position valuations
                await self._update_position_valuations()
                
                # Notify handlers
                await self._notify_valuation_handlers()
                
                # Wait for next update
                await asyncio.sleep(self.config['update_interval_seconds'])
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(self.config['update_interval_seconds'])
        
        logger.info("Valuation update loop ended")
    
    async def _update_market_data(self):
        """Update live odds and scores from data sources"""
        try:
            # Get unique market IDs from active positions
            market_ids = set(pos.market_id for pos in self.position_valuations.values())
            
            for market_id in market_ids:
                # Update odds (simulated for now - would integrate with real APIs)
                await self._fetch_live_odds(market_id)
                
                # Update scores (simulated for now - would integrate with real APIs)
                await self._fetch_live_score(market_id)
                
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    async def _fetch_live_odds(self, market_id: str):
        """Fetch live odds for a market (simulated)"""
        # In production, this would call real APIs
        # For now, simulate some odds movement
        
        current_time = datetime.now(timezone.utc)
        
        # Simulate odds with some randomness
        base_odds = {
            'home': 2.5 + (np.random.random() - 0.5) * 0.4,
            'away': 3.0 + (np.random.random() - 0.5) * 0.6,
            'draw': 3.2 + (np.random.random() - 0.5) * 0.4
        }
        
        live_odds = LiveOdds(
            market_id=market_id,
            timestamp=current_time,
            home_odds=base_odds['home'],
            away_odds=base_odds['away'],
            draw_odds=base_odds['draw'],
            source='simulated_api'
        )
        
        self.live_odds[market_id] = live_odds
        logger.debug(f"Updated odds for {market_id}: H={live_odds.home_odds:.2f}, A={live_odds.away_odds:.2f}, D={live_odds.draw_odds:.2f}")
    
    async def _fetch_live_score(self, market_id: str):
        """Fetch live score for a match (simulated)"""
        # In production, this would call real score APIs
        # For now, simulate some basic score progression
        
        current_time = datetime.now(timezone.utc)
        
        # Simulate a match in progress
        minute = (int(current_time.timestamp()) % 5400) // 60  # 90 minute cycle
        
        if minute < 90:
            status = 'live'
            # Simple score simulation
            home_score = max(0, int(np.random.poisson(minute / 45)) - 1)
            away_score = max(0, int(np.random.poisson(minute / 50)) - 1)
        else:
            status = 'finished'
            home_score = max(0, int(np.random.poisson(2)))
            away_score = max(0, int(np.random.poisson(1.8)))
            minute = 90
        
        live_score = LiveScore(
            match_id=market_id,
            timestamp=current_time,
            home_score=home_score,
            away_score=away_score,
            status=status,
            minute=minute
        )
        
        self.live_scores[market_id] = live_score
        logger.debug(f"Updated score for {market_id}: {home_score}-{away_score} ({minute}', {status})")
    
    async def _update_position_valuations(self):
        """Update valuations for all positions"""
        updated_count = 0
        
        for bet_id, position in self.position_valuations.items():
            try:
                # Get current market data
                live_odds = self.live_odds.get(position.market_id)
                live_score = self.live_scores.get(position.market_id)
                
                if live_odds:
                    # Update position with current odds
                    implied_probs = live_odds.implied_probabilities
                    position.current_probability = implied_probs.get(position.bet_on, 0.0)
                    
                    # Get current odds for the position
                    if position.bet_on == 'home':
                        position.current_odds = live_odds.home_odds
                    elif position.bet_on == 'away':
                        position.current_odds = live_odds.away_odds
                    elif position.bet_on == 'draw':
                        position.current_odds = live_odds.draw_odds
                
                if live_score:
                    position.live_score = live_score
                
                # Calculate new valuation
                position.calculate_valuation()
                updated_count += 1
                
                logger.debug(f"Updated valuation for {bet_id}: EV=${position.current_expected_value:.2f}, "
                           f"P(win)={position.probability_of_win:.2%}")
                
            except Exception as e:
                logger.error(f"Error updating position {bet_id}: {e}")
        
        if updated_count > 0:
            logger.debug(f"Updated {updated_count} position valuations")
    
    async def _notify_valuation_handlers(self):
        """Notify registered handlers of valuation updates"""
        if not self.valuation_handlers:
            return
        
        valuation_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'positions': {
                bet_id: {
                    'market_id': pos.market_id,
                    'stake': pos.stake,
                    'original_odds': pos.original_odds,
                    'current_odds': pos.current_odds,
                    'expected_value': pos.current_expected_value,
                    'probability_of_win': pos.probability_of_win,
                    'mark_to_market': pos.mark_to_market_value,
                    'odds_movement': pos.odds_movement
                }
                for bet_id, pos in self.position_valuations.items()
            },
            'summary': self.get_portfolio_summary()
        }
        
        for handler in self.valuation_handlers:
            try:
                await handler(valuation_data)
            except Exception as e:
                logger.error(f"Error in valuation handler: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of entire portfolio valuation"""
        if not self.position_valuations:
            return {
                'total_positions': 0,
                'total_stake': 0.0,
                'total_expected_value': 0.0,
                'total_mark_to_market': 0.0,
                'avg_probability_of_win': 0.0
            }
        
        positions = list(self.position_valuations.values())
        
        total_stake = sum(pos.stake for pos in positions)
        total_ev = sum(pos.current_expected_value for pos in positions)
        total_mtm = sum(pos.mark_to_market_value for pos in positions)
        avg_prob_win = np.mean([pos.probability_of_win for pos in positions])
        
        # Performance metrics
        positions_winning = sum(1 for pos in positions if pos.current_expected_value > 0)
        positions_losing = sum(1 for pos in positions if pos.current_expected_value < 0)
        
        # Odds movement analysis
        odds_improved = sum(1 for pos in positions if pos.odds_movement > 1.05)  # >5% improvement
        odds_worsened = sum(1 for pos in positions if pos.odds_movement < 0.95)  # >5% worse
        
        return {
            'total_positions': len(positions),
            'total_stake': total_stake,
            'total_expected_value': total_ev,
            'total_mark_to_market': total_mtm,
            'avg_probability_of_win': avg_prob_win,
            'expected_return_pct': (total_ev / total_stake * 100) if total_stake > 0 else 0,
            'mark_to_market_pct': ((total_mtm - total_stake) / total_stake * 100) if total_stake > 0 else 0,
            'positions_winning': positions_winning,
            'positions_losing': positions_losing,
            'odds_improved_count': odds_improved,
            'odds_worsened_count': odds_worsened
        }
    
    def get_position_details(self, bet_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information for a specific position"""
        if bet_id not in self.position_valuations:
            return None
        
        pos = self.position_valuations[bet_id]
        live_odds = self.live_odds.get(pos.market_id)
        live_score = self.live_scores.get(pos.market_id)
        
        return {
            'bet_id': bet_id,
            'market_id': pos.market_id,
            'position': {
                'stake': pos.stake,
                'original_odds': pos.original_odds,
                'bet_on': pos.bet_on,
                'bet_type': pos.bet_type
            },
            'current_valuation': {
                'expected_value': pos.current_expected_value,
                'probability_of_win': pos.probability_of_win,
                'mark_to_market_value': pos.mark_to_market_value,
                'odds_movement': pos.odds_movement,
                'ev_change': pos.ev_change
            },
            'market_data': {
                'current_odds': pos.current_odds,
                'current_probability': pos.current_probability,
                'live_odds': {
                    'home': live_odds.home_odds if live_odds else None,
                    'away': live_odds.away_odds if live_odds else None,
                    'draw': live_odds.draw_odds if live_odds else None,
                    'updated_at': live_odds.timestamp.isoformat() if live_odds else None
                } if live_odds else None,
                'live_score': {
                    'home_score': live_score.home_score,
                    'away_score': live_score.away_score,
                    'status': live_score.status,
                    'minute': live_score.minute,
                    'updated_at': live_score.timestamp.isoformat()
                } if live_score else None
            }
        }
    
    async def force_update_position(self, bet_id: str) -> bool:
        """Force update a specific position"""
        if bet_id not in self.position_valuations:
            return False
        
        position = self.position_valuations[bet_id]
        
        try:
            # Force refresh market data
            await self._fetch_live_odds(position.market_id)
            await self._fetch_live_score(position.market_id)
            
            # Update the specific position
            live_odds = self.live_odds.get(position.market_id)
            live_score = self.live_scores.get(position.market_id)
            
            if live_odds:
                implied_probs = live_odds.implied_probabilities
                position.current_probability = implied_probs.get(position.bet_on, 0.0)
                
                if position.bet_on == 'home':
                    position.current_odds = live_odds.home_odds
                elif position.bet_on == 'away':
                    position.current_odds = live_odds.away_odds
                elif position.bet_on == 'draw':
                    position.current_odds = live_odds.draw_odds
            
            if live_score:
                position.live_score = live_score
            
            position.calculate_valuation()
            
            logger.info(f"Force updated position {bet_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error force updating position {bet_id}: {e}")
            return False


# Integration helpers
async def start_realtime_tracking_for_session(session_id: str) -> RealTimeValuationEngine:
    """Start real-time tracking for a trading session"""
    engine = RealTimeValuationEngine()
    await engine.start_live_tracking(session_id)
    return engine


# Example usage and testing
if __name__ == "__main__":
    async def test_valuation_engine():
        print("📊 Real-Time Valuation Engine")
        print("=" * 50)
        
        engine = RealTimeValuationEngine()
        
        # Register a test handler
        async def test_handler(valuation_data):
            summary = valuation_data['summary']
            print(f"Portfolio Update: {summary['total_positions']} positions, "
                  f"EV=${summary['total_expected_value']:.2f}, "
                  f"Return={summary['expected_return_pct']:.1f}%")
        
        engine.register_valuation_handler(test_handler)
        
        try:
            # Start tracking (would need real session data)
            test_session_id = "test_session_001"
            await engine.start_live_tracking(test_session_id)
            
            # Let it run for a bit
            print("Running for 30 seconds...")
            await asyncio.sleep(30)
            
            # Get summary
            summary = engine.get_portfolio_summary()
            print(f"\nFinal Summary:")
            print(f"  Total Positions: {summary['total_positions']}")
            print(f"  Total Stake: ${summary['total_stake']:.2f}")
            print(f"  Expected Value: ${summary['total_expected_value']:.2f}")
            print(f"  Expected Return: {summary['expected_return_pct']:.2f}%")
            
        except Exception as e:
            print(f"Error in testing: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            await engine.stop_live_tracking()
    
    # Run the test
    try:
        asyncio.run(test_valuation_engine())
    except KeyboardInterrupt:
        print("\nTest interrupted")
    except Exception as e:
        print(f"Test error: {e}")
        print("Note: This requires a database with active positions to test properly")