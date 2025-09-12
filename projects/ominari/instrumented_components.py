#!/usr/bin/env python3
"""
Instrumented versions of key components with telemetry.
Wraps existing functionality with metrics and tracing.
"""

import time
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import pandas as pd

from telemetry import (
    get_telemetry, traced, timed, 
    trace_db_operation, trace_signal_generation, trace_api_call
)
from database_v2 import db_manager
from models import Bet, PaperTradingPosition
from paper_trading_engine import PaperTradingEngine
from signal_registry import SignalRegistry, BaseSignalProvider
from risk_manager import RiskManager

logger = logging.getLogger(__name__)


class InstrumentedPaperTradingEngine(PaperTradingEngine):
    """Paper trading engine with telemetry instrumentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = get_telemetry()
        
    @traced("paper_trading.place_bet")
    def place_bet(self, 
                  market_id: str,
                  outcome: str,
                  stake: float,
                  odds: float,
                  strategy_name: str = "default") -> Optional[str]:
        """Place a bet with telemetry."""
        
        start = time.time()
        bet_id = None
        
        try:
            # Record attempt
            self.telemetry.record_bet_placed(
                amount=stake,
                market="Soccer",  # Would get from market data
                strategy=strategy_name
            )
            
            # Place bet
            bet_id = super().place_bet(
                market_id=market_id,
                outcome=outcome,
                stake=stake,
                odds=odds,
                strategy_name=strategy_name
            )
            
            # Record success
            if bet_id:
                duration_ms = (time.time() - start) * 1000
                logger.info(f"Bet placed in {duration_ms:.1f}ms: {bet_id}")
            
            return bet_id
            
        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            raise
            
    @traced("paper_trading.settle_bet")
    def settle_bet(self, position_id: str, final_odds: float) -> Dict:
        """Settle a bet with telemetry."""
        
        result = super().settle_bet(position_id, final_odds)
        
        if result:
            # Record settlement metrics
            self.telemetry.record_bet_settled(
                pnl=result['pnl'],
                won=result['pnl'] > 0,
                market="Soccer",  # Would get from position data
                strategy=result.get('strategy_name', 'unknown')
            )
            
        return result


class InstrumentedSignalProvider(BaseSignalProvider):
    """Signal provider wrapper with telemetry."""
    
    def __init__(self, signal: BaseSignalProvider):
        self.signal = signal
        self.telemetry = get_telemetry()
        super().__init__(signal.name, signal.version)
        
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        """Get probabilities with telemetry."""
        
        with trace_signal_generation(self.signal.name):
            start = time.time()
            
            # Get predictions
            probs = self.signal.get_probs(df)
            
            # Record metrics
            duration_ms = (time.time() - start) * 1000
            if not probs.empty:
                # Simple accuracy proxy - how confident are predictions
                avg_confidence = abs(probs - 0.5).mean()
                self.telemetry.record_signal_accuracy(
                    avg_confidence,
                    self.signal.name
                )
                
            logger.debug(f"Signal {self.signal.name} generated {len(probs)} predictions in {duration_ms:.1f}ms")
            
            return probs
            
    def get_parameters(self) -> Dict[str, Any]:
        return self.signal.get_parameters()
        
    def get_required_columns(self) -> List[str]:
        return self.signal.get_required_columns()


class InstrumentedRiskManager(RiskManager):
    """Risk manager with telemetry instrumentation."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.telemetry = get_telemetry()
        
    @traced("risk.check_pre_bet_limits")
    def check_pre_bet_limits(self, 
                           bet_proposal: Dict,
                           bankroll: float) -> tuple[bool, List[str]]:
        """Check bet limits with telemetry."""
        
        # Trace the check
        with self.telemetry.span("risk.validation", {
            "stake": bet_proposal.get('stake', 0),
            "odds": bet_proposal.get('odds', 0),
            "bankroll": bankroll
        }):
            is_valid, violations = super().check_pre_bet_limits(
                bet_proposal, bankroll
            )
            
            # Record validation result
            if not is_valid:
                logger.warning(f"Bet rejected: {violations}")
                
            return is_valid, violations
            
    @traced("risk.calculate_metrics")
    @timed("risk")
    def calculate_current_metrics(self, bankroll: float):
        """Calculate metrics with telemetry."""
        return super().calculate_current_metrics(bankroll)


class InstrumentedDatabaseOperations:
    """Database operations with telemetry."""
    
    def __init__(self):
        self.telemetry = get_telemetry()
        
    @traced("db.fetch_open_markets")
    def fetch_open_markets(self, as_of: datetime, limit: int = 100):
        """Fetch open markets with tracing."""
        
        with trace_db_operation("select", "markets"):
            with db_manager.get_db_session() as db:
                # Your query here
                pass
                
    @traced("db.save_bet")
    def save_bet(self, bet_data: Dict):
        """Save bet with tracing."""
        
        with trace_db_operation("insert", "bets"):
            with db_manager.get_db_session() as db:
                bet = Bet(**bet_data)
                db.add(bet)
                db.commit()
                return bet.id


def instrument_signal_registry(registry: SignalRegistry):
    """Add telemetry to all signals in registry."""
    instrumented_count = 0
    
    for signal_name in registry.list_signals():
        signal = registry.get_signal(signal_name)
        if signal and not isinstance(signal, InstrumentedSignalProvider):
            # Wrap with instrumentation
            instrumented = InstrumentedSignalProvider(signal)
            registry.signals[signal_name] = instrumented
            instrumented_count += 1
            
    logger.info(f"Instrumented {instrumented_count} signals in registry")
    return registry


# Instrumented evaluation function
@traced("evaluation.generate_betting_session")
def instrumented_evaluate_markets(
    kelly_bankroll: float,
    execution_bankroll: float,
    **kwargs
) -> Dict:
    """Instrumented version of evaluate_open_markets."""
    
    telemetry = get_telemetry()
    
    with telemetry.span("betting_session", {
        "kelly_bankroll": kelly_bankroll,
        "execution_bankroll": execution_bankroll,
        "mode": kwargs.get('mode', 'unknown')
    }) as span:
        
        # Import here to avoid circular dependency
        from evaluate_open_markets import generate_betting_session_report_and_save
        
        # Add span context
        span.set_attribute("strategy", kwargs.get('strat', 'default'))
        
        # Run evaluation
        result = generate_betting_session_report_and_save(
            kelly_bankroll=kelly_bankroll,
            execution_bankroll=execution_bankroll,
            **kwargs
        )
        
        # Record results
        if result and 'trimmed' in result:
            bets = result['trimmed'][result['trimmed']['stake'] > 0]
            span.set_attribute("bets_recommended", len(bets))
            span.set_attribute("total_stake", float(bets['stake'].sum()))
            
        return result


def setup_instrumentation():
    """Setup instrumentation for all components."""
    logger.info("Setting up component instrumentation...")
    
    # Initialize telemetry
    from telemetry import initialize_telemetry
    telemetry = initialize_telemetry()
    
    # Instrument signal registry
    try:
        from integrate_signal_registry import get_or_create_registry
        registry, _ = get_or_create_registry()
        instrument_signal_registry(registry)
    except Exception as e:
        logger.error(f"Failed to instrument signal registry: {e}")
        
    logger.info("Component instrumentation complete")
    
    return telemetry


def demonstrate_instrumented_components():
    """Demonstrate instrumented components."""
    print("=== Instrumented Components Demo ===\n")
    
    # Setup
    telemetry = setup_instrumentation()
    
    # Create instrumented components
    engine = InstrumentedPaperTradingEngine(
        db_manager=db_manager,
        session_id="test_session"
    )
    
    risk_mgr = InstrumentedRiskManager()
    
    # Simulate operations
    print("1. Testing instrumented paper trading...")
    try:
        bet_id = engine.place_bet(
            market_id="test_market",
            outcome="option_1",
            stake=100.0,
            odds=2.5,
            strategy_name="test_strategy"
        )
        print(f"   Bet placed: {bet_id}")
    except Exception as e:
        print(f"   Error: {e}")
        
    print("\n2. Testing instrumented risk checking...")
    bet_proposal = {
        'stake': 50,
        'stake_pct': 0.01,
        'odds': 2.0,
        'edge': 0.02,
        'event_time': datetime.now(timezone.utc),
        'sport': 'Soccer'
    }
    
    is_valid, violations = risk_mgr.check_pre_bet_limits(bet_proposal, 5000)
    print(f"   Valid: {is_valid}")
    if violations:
        print(f"   Violations: {violations}")
        
    print("\n3. Testing instrumented signal generation...")
    from signals import ImpliedRawSignal
    
    signal = ImpliedRawSignal()
    instrumented_signal = InstrumentedSignalProvider(signal)
    
    test_data = pd.DataFrame({
        'implied_raw': [45.0, 55.0],
        'odds': [2.22, 1.82]
    })
    
    probs = instrumented_signal.get_probs(test_data)
    print(f"   Generated {len(probs)} predictions")
    
    print("\nInstrumentation demo complete!")
    print("Check telemetry backend for metrics and traces")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_instrumented_components()