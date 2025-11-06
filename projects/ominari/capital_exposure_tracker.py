#!/usr/bin/env python3
"""
Capital Exposure Tracker - Real-Time Capital & Risk Tracking

Tracks the multi-state capital flow and provides real-time Value at Risk (VaR) metrics:
- Available Capital: Cash ready for new positions
- Pending Stakes: Capital committed but not yet in-play
- In-Play Exposure: Active positions with risk/reward
- Settlement Pending: Won positions awaiting payout
- Lost Stakes: Capital definitively lost

Integrates with dynamic chunking to provide empirical data for capital planning.
"""

import os
import json
import logging
import psycopg2
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class CapitalState:
    """Represents capital state at a point in time"""
    timestamp: datetime
    
    # Capital breakdown
    available_cash: float = 0.0      # Ready for new bets
    pending_stakes: float = 0.0      # Committed but not in-play
    in_play_exposure: float = 0.0    # Active stakes
    settlement_pending: float = 0.0   # Won stakes awaiting payout
    lost_stakes: float = 0.0         # Definitively lost
    
    # Computed metrics
    total_bankroll: float = 0.0      # Sum of all capital states
    utilization_rate: float = 0.0    # (pending + in_play) / total
    
    # Risk metrics
    expected_value: float = 0.0      # Expected value of in-play positions
    value_at_risk: float = 0.0       # 95% VaR of active positions
    maximum_loss: float = 0.0        # Worst case scenario
    
    def __post_init__(self):
        """Calculate derived metrics"""
        self.total_bankroll = (self.available_cash + self.pending_stakes + 
                              self.in_play_exposure + self.settlement_pending)
        
        if self.total_bankroll > 0:
            self.utilization_rate = (self.pending_stakes + self.in_play_exposure) / self.total_bankroll
        else:
            self.utilization_rate = 0.0

@dataclass
class PositionExposure:
    """Individual position exposure data"""
    bet_id: str
    market_id: str
    chunk_id: str
    
    # Position details
    stake: float
    odds: float
    bet_type: str
    bet_on: str
    
    # Timing
    placed_at: datetime
    expected_settlement: Optional[datetime] = None
    
    # Risk metrics
    probability: float = 0.0         # Estimated win probability
    expected_pnl: float = 0.0        # Expected profit/loss
    max_loss: float = 0.0            # Maximum possible loss
    max_win: float = 0.0             # Maximum possible win
    
    # Status tracking
    status: str = 'pending'          # pending, in_play, won, lost, settled
    
    def __post_init__(self):
        """Calculate risk metrics"""
        self.max_loss = self.stake
        self.max_win = (self.odds - 1) * self.stake
        
        if self.probability > 0:
            win_pnl = self.max_win
            lose_pnl = -self.max_loss
            self.expected_pnl = (self.probability * win_pnl) + ((1 - self.probability) * lose_pnl)


class CapitalExposureTracker:
    """Tracks real-time capital exposure and risk metrics"""
    
    def __init__(self, db_config: Optional[Dict] = None):
        self.db_config = db_config or {
            'host': os.environ.get('PG_HOST', 'localhost'),
            'port': os.environ.get('PG_PORT', '5999'),
            'user': os.environ.get('PG_USER', 'ominari_user'),
            'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure'),
            'database': os.environ.get('PG_DB', 'ominari_production')
        }
        
        # Capital tracking
        self.capital_history: List[CapitalState] = []
        self.active_positions: Dict[str, PositionExposure] = {}
        
        # Risk configuration
        self.risk_config = {
            'var_confidence': 0.95,           # 95% VaR confidence level
            'max_utilization': 0.8,           # Maximum 80% capital utilization
            'max_chunk_concentration': 0.3,   # Max 30% per chunk
            'stress_test_scenarios': 10       # Number of stress scenarios to run
        }
        
        # Performance tracking
        self.performance_metrics = {
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'avg_hold_time': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
        
    def update_capital_state(self, session_id: str) -> CapitalState:
        """Calculate current capital state from database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Get session info
                    cur.execute("""
                        SELECT initial_bankroll, current_bankroll
                        FROM paper_trading_sessions
                        WHERE session_id = %s
                    """, (session_id,))
                    
                    session_data = cur.fetchone()
                    if not session_data:
                        raise ValueError(f"Session {session_id} not found")
                    
                    initial_bankroll, current_bankroll = session_data
                    
                    # Get capital breakdown by position status
                    cur.execute("""
                        SELECT 
                            status,
                            SUM(stake) as total_stake,
                            SUM(CASE WHEN pnl > 0 THEN pnl ELSE 0 END) as pending_winnings,
                            SUM(CASE WHEN pnl < 0 THEN ABS(pnl) ELSE 0 END) as realized_losses,
                            COUNT(*) as position_count
                        FROM paper_trading_positions
                        WHERE session_id = %s
                        GROUP BY status
                    """, (session_id,))
                    
                    status_breakdown = {row[0]: row[1:] for row in cur.fetchall()}
                    
                    # Calculate capital states
                    pending_stakes = status_breakdown.get('pending', (0, 0, 0, 0))[0] or 0
                    in_play_exposure = status_breakdown.get('open', (0, 0, 0, 0))[0] or 0
                    settlement_pending = status_breakdown.get('won', (0, 0, 0, 0))[1] or 0
                    lost_stakes = sum(row[2] for row in status_breakdown.values()) or 0
                    
                    # Available cash = current bankroll - committed capital
                    committed_capital = pending_stakes + in_play_exposure
                    available_cash = max(0, current_bankroll - committed_capital)
                    
                    # Create capital state
                    capital_state = CapitalState(
                        timestamp=datetime.now(timezone.utc),
                        available_cash=available_cash,
                        pending_stakes=pending_stakes,
                        in_play_exposure=in_play_exposure,
                        settlement_pending=settlement_pending,
                        lost_stakes=lost_stakes
                    )
                    
                    # Calculate risk metrics
                    capital_state = self._calculate_risk_metrics(capital_state, session_id)
                    
                    # Store in history
                    self.capital_history.append(capital_state)
                    
                    # Keep only recent history (last 24 hours)
                    cutoff_time = datetime.now(timezone.utc) - timedelta(hours=24)
                    self.capital_history = [cs for cs in self.capital_history if cs.timestamp > cutoff_time]
                    
                    logger.debug(f"Updated capital state: Available=${capital_state.available_cash:.2f}, "
                               f"InPlay=${capital_state.in_play_exposure:.2f}, "
                               f"VaR=${capital_state.value_at_risk:.2f}")
                    
                    return capital_state
                    
        except Exception as e:
            logger.error(f"Error updating capital state: {e}")
            raise
    
    def _calculate_risk_metrics(self, capital_state: CapitalState, session_id: str) -> CapitalState:
        """Calculate Value at Risk and expected value metrics"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    # Get active positions with odds and probabilities
                    cur.execute("""
                        SELECT 
                            p.bet_id, p.stake, p.odds, p.bet_type, p.bet_on,
                            p.placed_at, p.match_id,
                            -- Estimate probability from odds (simplified)
                            1.0 / p.odds as implied_probability,
                            p.strategy_config
                        FROM paper_trading_positions p
                        WHERE p.session_id = %s 
                        AND p.status IN ('pending', 'open')
                        ORDER BY p.placed_at
                    """, (session_id,))
                    
                    positions = cur.fetchall()
                    
                    if not positions:
                        capital_state.expected_value = 0.0
                        capital_state.value_at_risk = 0.0
                        capital_state.maximum_loss = 0.0
                        return capital_state
                    
                    # Calculate expected value and VaR
                    total_expected_pnl = 0.0
                    total_stakes = 0.0
                    pnl_scenarios = []
                    
                    for pos in positions:
                        bet_id, stake, odds, bet_type, bet_on, placed_at, match_id, implied_prob, strategy_config = pos
                        
                        # Adjust probability based on our model if available
                        probability = implied_prob
                        if strategy_config:
                            try:
                                config = json.loads(strategy_config) if isinstance(strategy_config, str) else strategy_config
                                if 'predicted_probability' in config:
                                    probability = config['predicted_probability']
                            except:
                                pass
                        
                        # Calculate position metrics
                        max_win = (odds - 1) * stake
                        max_loss = stake
                        expected_pnl = (probability * max_win) + ((1 - probability) * (-max_loss))
                        
                        total_expected_pnl += expected_pnl
                        total_stakes += stake
                        
                        # Generate scenarios for VaR calculation
                        pnl_scenarios.append({
                            'win_pnl': max_win,
                            'lose_pnl': -max_loss,
                            'probability': probability,
                            'stake': stake
                        })
                    
                    # Monte Carlo simulation for VaR
                    var_scenarios = self._run_var_simulation(pnl_scenarios)
                    
                    # Calculate metrics
                    capital_state.expected_value = total_expected_pnl
                    capital_state.value_at_risk = self._calculate_var(var_scenarios, self.risk_config['var_confidence'])
                    capital_state.maximum_loss = total_stakes  # Worst case: lose all stakes
                    
                    return capital_state
                    
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            # Return original state if calculation fails
            return capital_state
    
    def _run_var_simulation(self, scenarios: List[Dict], num_simulations: int = 10000) -> List[float]:
        """Run Monte Carlo simulation for Value at Risk calculation"""
        total_pnl_outcomes = []
        
        for _ in range(num_simulations):
            total_pnl = 0.0
            
            for scenario in scenarios:
                random_outcome = np.random.random()
                if random_outcome < scenario['probability']:
                    total_pnl += scenario['win_pnl']
                else:
                    total_pnl += scenario['lose_pnl']
            
            total_pnl_outcomes.append(total_pnl)
        
        return total_pnl_outcomes
    
    def _calculate_var(self, pnl_outcomes: List[float], confidence_level: float) -> float:
        """Calculate Value at Risk at given confidence level"""
        if not pnl_outcomes:
            return 0.0
        
        # Sort outcomes (worst to best)
        sorted_outcomes = sorted(pnl_outcomes)
        
        # Find percentile corresponding to confidence level
        # For 95% VaR, we want the 5th percentile (worst 5% of outcomes)
        percentile_index = int((1 - confidence_level) * len(sorted_outcomes))
        
        # VaR is the absolute value of the loss at this percentile
        var_loss = sorted_outcomes[percentile_index]
        return abs(min(0, var_loss))  # Only count losses
    
    def get_chunk_exposure(self, session_id: str) -> Dict[str, Dict]:
        """Get capital exposure breakdown by chunk"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            COALESCE(p.strategy_config->>'chunk_id', 'unknown') as chunk_id,
                            p.status,
                            COUNT(*) as position_count,
                            SUM(p.stake) as total_stake,
                            SUM(CASE WHEN p.pnl > 0 THEN p.pnl ELSE 0 END) as pending_winnings,
                            AVG(p.odds) as avg_odds,
                            MIN(p.placed_at) as earliest_position,
                            MAX(p.placed_at) as latest_position
                        FROM paper_trading_positions p
                        WHERE p.session_id = %s
                        GROUP BY chunk_id, p.status
                        ORDER BY chunk_id, p.status
                    """, (session_id,))
                    
                    chunk_data = defaultdict(lambda: {
                        'total_exposure': 0.0,
                        'position_count': 0,
                        'pending_stakes': 0.0,
                        'in_play_stakes': 0.0,
                        'pending_winnings': 0.0,
                        'avg_odds': 0.0,
                        'earliest_position': None,
                        'latest_position': None,
                        'status_breakdown': {}
                    })
                    
                    for row in cur.fetchall():
                        chunk_id, status, count, stake, winnings, odds, earliest, latest = row
                        
                        chunk = chunk_data[chunk_id]
                        chunk['position_count'] += count
                        chunk['total_exposure'] += stake
                        chunk['status_breakdown'][status] = {
                            'count': count,
                            'stake': stake,
                            'winnings': winnings or 0,
                            'avg_odds': odds or 0
                        }
                        
                        if status == 'pending':
                            chunk['pending_stakes'] += stake
                        elif status == 'open':
                            chunk['in_play_stakes'] += stake
                        elif status == 'won':
                            chunk['pending_winnings'] += winnings or 0
                        
                        # Track timing
                        if not chunk['earliest_position'] or earliest < chunk['earliest_position']:
                            chunk['earliest_position'] = earliest
                        if not chunk['latest_position'] or latest > chunk['latest_position']:
                            chunk['latest_position'] = latest
                        
                        # Weighted average odds
                        total_positions = sum(s['count'] for s in chunk['status_breakdown'].values())
                        chunk['avg_odds'] = sum(
                            s['avg_odds'] * s['count'] for s in chunk['status_breakdown'].values() 
                            if s['avg_odds']
                        ) / max(1, total_positions)
                    
                    return dict(chunk_data)
                    
        except Exception as e:
            logger.error(f"Error getting chunk exposure: {e}")
            return {}
    
    def get_capital_flow_forecast(self, chunks: List[Dict], hours_ahead: int = 12) -> List[Dict]:
        """Forecast capital flow based on expected settlement times"""
        current_time = datetime.now(timezone.utc)
        forecast_end = current_time + timedelta(hours=hours_ahead)
        
        # Get current capital state
        capital_events = []
        
        # Add settlement events from chunks
        for chunk in chunks:
            if 'expected_settlement_time' in chunk:
                settlement_time = chunk['expected_settlement_time']
                if isinstance(settlement_time, str):
                    settlement_time = datetime.fromisoformat(settlement_time.replace('Z', '+00:00'))
                
                if current_time <= settlement_time <= forecast_end:
                    capital_events.append({
                        'time': settlement_time,
                        'type': 'settlement',
                        'chunk_id': chunk.get('chunk_id', 'unknown'),
                        'expected_return': chunk.get('expected_return', 0),
                        'stakes_freed': chunk.get('total_stakes', 0),
                        'description': f"Chunk {chunk.get('label', 'Unknown')} settlement"
                    })
        
        # Sort by time
        capital_events.sort(key=lambda x: x['time'])
        
        return capital_events
    
    def check_risk_limits(self, capital_state: CapitalState, proposed_stake: float = 0) -> Dict[str, Any]:
        """Check if proposed position would violate risk limits"""
        warnings = []
        violations = []
        
        # 1. Utilization check
        new_utilization = ((capital_state.pending_stakes + capital_state.in_play_exposure + proposed_stake) / 
                          capital_state.total_bankroll if capital_state.total_bankroll > 0 else 0)
        
        if new_utilization > self.risk_config['max_utilization']:
            violations.append(f"Utilization {new_utilization:.1%} exceeds limit {self.risk_config['max_utilization']:.1%}")
        elif new_utilization > self.risk_config['max_utilization'] * 0.9:
            warnings.append(f"High utilization {new_utilization:.1%} approaching limit")
        
        # 2. Available cash check
        if proposed_stake > capital_state.available_cash:
            violations.append(f"Insufficient cash: ${proposed_stake:.2f} requested, ${capital_state.available_cash:.2f} available")
        
        # 3. VaR check (proposed position would increase VaR)
        if capital_state.value_at_risk + proposed_stake > capital_state.total_bankroll * 0.2:  # Max 20% VaR
            warnings.append(f"High VaR exposure: ${capital_state.value_at_risk + proposed_stake:.2f}")
        
        # 4. Recent performance check
        if len(self.capital_history) > 10:
            recent_trend = self._calculate_capital_trend()
            if recent_trend < -0.05:  # Declining by more than 5%
                warnings.append(f"Recent negative trend: {recent_trend:.1%}")
        
        return {
            'approved': len(violations) == 0,
            'warnings': warnings,
            'violations': violations,
            'new_utilization': new_utilization,
            'available_cash': capital_state.available_cash - proposed_stake
        }
    
    def _calculate_capital_trend(self) -> float:
        """Calculate recent capital trend (% change over recent history)"""
        if len(self.capital_history) < 2:
            return 0.0
        
        recent_values = [cs.total_bankroll for cs in self.capital_history[-10:]]
        if len(recent_values) < 2:
            return 0.0
        
        # Linear regression to get trend
        x = list(range(len(recent_values)))
        y = recent_values
        
        # Simple trend calculation
        if recent_values[0] > 0:
            return (recent_values[-1] - recent_values[0]) / recent_values[0]
        return 0.0
    
    def get_real_time_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive real-time capital and risk metrics"""
        capital_state = self.update_capital_state(session_id)
        chunk_exposure = self.get_chunk_exposure(session_id)
        
        # Calculate additional metrics
        capital_efficiency = (capital_state.expected_value / capital_state.total_bankroll 
                            if capital_state.total_bankroll > 0 else 0)
        
        risk_adjusted_return = (capital_state.expected_value / max(capital_state.value_at_risk, 1))
        
        return {
            'timestamp': capital_state.timestamp.isoformat(),
            'capital_state': {
                'available_cash': capital_state.available_cash,
                'pending_stakes': capital_state.pending_stakes,
                'in_play_exposure': capital_state.in_play_exposure,
                'settlement_pending': capital_state.settlement_pending,
                'total_bankroll': capital_state.total_bankroll,
                'utilization_rate': capital_state.utilization_rate
            },
            'risk_metrics': {
                'expected_value': capital_state.expected_value,
                'value_at_risk': capital_state.value_at_risk,
                'maximum_loss': capital_state.maximum_loss,
                'capital_efficiency': capital_efficiency,
                'risk_adjusted_return': risk_adjusted_return
            },
            'chunk_exposure': chunk_exposure,
            'limits': {
                'max_utilization': self.risk_config['max_utilization'],
                'current_utilization': capital_state.utilization_rate,
                'utilization_headroom': self.risk_config['max_utilization'] - capital_state.utilization_rate
            }
        }
    
    def log_capital_event(self, event_type: str, details: Dict[str, Any]):
        """Log significant capital events for analysis"""
        event_data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'event_type': event_type,
            'details': details
        }
        
        # Write to daily log file
        log_file = f"capital_events_{datetime.now().strftime('%Y%m%d')}.json"
        try:
            with open(log_file, 'a') as f:
                f.write(json.dumps(event_data) + '\n')
        except Exception as e:
            logger.error(f"Failed to log capital event: {e}")


# Integration helper functions
def get_capital_state_for_chunks(session_id: str) -> Dict[str, Any]:
    """Get capital state formatted for dynamic chunk manager"""
    tracker = CapitalExposureTracker()
    capital_state = tracker.update_capital_state(session_id)
    
    return {
        'total_bankroll': capital_state.total_bankroll,
        'available_cash': capital_state.available_cash,
        'utilization_rate': capital_state.utilization_rate,
        'pending_exposure': capital_state.pending_stakes + capital_state.in_play_exposure
    }


# Example usage and testing
if __name__ == "__main__":
    # Test the capital exposure tracker
    tracker = CapitalExposureTracker()
    
    print("🏦 Capital Exposure Tracker")
    print("=" * 50)
    
    # Test with a sample session (would need real session data)
    try:
        test_session_id = "test_session_001"
        
        # Get real-time metrics
        metrics = tracker.get_real_time_metrics(test_session_id)
        
        print(f"\nCapital State:")
        print(f"  Available Cash: ${metrics['capital_state']['available_cash']:.2f}")
        print(f"  In-Play Exposure: ${metrics['capital_state']['in_play_exposure']:.2f}")
        print(f"  Total Bankroll: ${metrics['capital_state']['total_bankroll']:.2f}")
        print(f"  Utilization: {metrics['capital_state']['utilization_rate']:.1%}")
        
        print(f"\nRisk Metrics:")
        print(f"  Expected Value: ${metrics['risk_metrics']['expected_value']:.2f}")
        print(f"  Value at Risk: ${metrics['risk_metrics']['value_at_risk']:.2f}")
        print(f"  Capital Efficiency: {metrics['risk_metrics']['capital_efficiency']:.1%}")
        
        print(f"\nChunk Exposure:")
        for chunk_id, exposure in metrics['chunk_exposure'].items():
            print(f"  {chunk_id}: ${exposure['total_exposure']:.2f} ({exposure['position_count']} positions)")
        
        # Test risk limit checking
        print(f"\nRisk Limit Check (for $100 proposed stake):")
        capital_state = tracker.capital_history[-1] if tracker.capital_history else None
        if capital_state:
            risk_check = tracker.check_risk_limits(capital_state, 100)
            print(f"  Approved: {risk_check['approved']}")
            if risk_check['warnings']:
                print(f"  Warnings: {', '.join(risk_check['warnings'])}")
            if risk_check['violations']:
                print(f"  Violations: {', '.join(risk_check['violations'])}")
        
    except Exception as e:
        print(f"Error in testing: {e}")
        print("Note: This requires a real database session to test properly")