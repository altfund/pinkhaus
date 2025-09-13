#!/usr/bin/env python3
"""Database-backed paper trading system."""

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any

from sqlalchemy import func, and_, or_
from database_v2 import db_manager
from paper_trading_models_v2 import (
    PaperTradingSession, PaperTradingPosition, PaperTradingTrade,
    PaperTradingSnapshot, MarketName, PositionStatus, SessionStatus,
    TradeType, Outcome, Result
)

logger = logging.getLogger(__name__)


class PaperTradingDB:
    """Database-backed paper trading system."""
    
    def __init__(self, session_id: str = None):
        """Initialize paper trading with database backend."""
        self.session_id = session_id
        self.session = None
        self.min_bet = 10
        self.kelly_fraction = 0.25
        
    def create_session(self, initial_bankroll: float = 10000, 
                      session_name: str = None) -> str:
        """Create a new paper trading session."""
        session_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        
        with db_manager.get_db_session() as db:
            session = PaperTradingSession(
                session_id=session_id,
                session_name=session_name or f"Session {session_id}",
                initial_bankroll=Decimal(str(initial_bankroll)),
                status=SessionStatus.ACTIVE,
                strategy_config={
                    'min_bet': self.min_bet,
                    'kelly_fraction': self.kelly_fraction
                }
            )
            db.add(session)
            db.commit()
            
        self.session_id = session_id
        logger.info(f"Created new paper trading session: {session_id}")
        return session_id
        
    def load_session(self, session_id: str):
        """Load an existing session."""
        with db_manager.get_db_session() as db:
            session = db.query(PaperTradingSession).filter_by(
                session_id=session_id
            ).first()
            
            if not session:
                raise ValueError(f"Session {session_id} not found")
                
            self.session = session
            self.session_id = session_id
            self.min_bet = session.strategy_config.get('min_bet', 10)
            self.kelly_fraction = session.strategy_config.get('kelly_fraction', 0.25)
            
    def get_current_bankroll(self) -> float:
        """Get current bankroll (cash balance)."""
        with db_manager.get_db_session() as db:
            # Get latest snapshot
            snapshot = db.query(PaperTradingSnapshot).filter_by(
                session_id=self.session_id
            ).order_by(PaperTradingSnapshot.snapshot_time.desc()).first()
            
            if snapshot:
                return float(snapshot.cash_balance)
            
            # Otherwise get initial bankroll
            session = db.query(PaperTradingSession).filter_by(
                session_id=self.session_id
            ).first()
            return float(session.initial_bankroll) if session else 0
            
    def get_portfolio_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        with db_manager.get_db_session() as db:
            cash = self.get_current_bankroll()
            
            # Get open positions value
            positions = db.query(PaperTradingPosition).filter_by(
                session_id=self.session_id,
                status=PositionStatus.OPEN
            ).all()
            
            positions_value = sum(float(pos.stake) for pos in positions)
            
            return cash + positions_value
            
    def place_bet(self, market_id: str, market_name: str, outcome: str,
                  stake: float, odds: float, signal_name: str,
                  maturity_date: datetime, fee_info: Dict[str, float]) -> bool:
        """Place a new bet."""
        try:
            with db_manager.get_db_session() as db:
                # Store market name
                self._store_market_name(db, market_id, market_name)
                
                # Calculate fees
                safebox_fee = fee_info.get('safebox_fee_pct', 0.02)
                skew_fee = fee_info.get('skew_fee_pct', 0.01)
                fee_amount = stake * (safebox_fee + skew_fee)
                execution_stake = stake + fee_amount
                
                # Check if we have enough balance
                cash = self.get_current_bankroll()
                if execution_stake > cash:
                    logger.warning(f"Insufficient balance: {cash} < {execution_stake}")
                    return False
                
                # Create position
                position = PaperTradingPosition(
                    session_id=self.session_id,
                    market_id=market_id,
                    outcome=Outcome[outcome.upper()],
                    stake=Decimal(str(stake)),
                    execution_stake=Decimal(str(execution_stake)),
                    avg_odds=Decimal(str(odds)),
                    safebox_fee_bps=int(safebox_fee * 10000),
                    skew_fee_bps=int(skew_fee * 10000),
                    status=PositionStatus.OPEN,
                    maturity_date=maturity_date
                )
                db.add(position)
                db.flush()
                
                # Create trade record
                trade = PaperTradingTrade(
                    position_id=position.position_id,
                    session_id=self.session_id,
                    trade_type=TradeType.OPEN,
                    stake_delta=Decimal(str(stake)),
                    odds=Decimal(str(odds)),
                    fee_amount=Decimal(str(fee_amount))
                )
                db.add(trade)
                
                # Update cash balance in snapshot
                self._update_snapshot(db)
                
                db.commit()
                logger.info(f"Placed bet: {market_name} {outcome} ${stake} @ {odds}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to place bet: {e}")
            return False
            
    def settle_position(self, market_id: str, resolved_outcome: str) -> Dict[str, Any]:
        """Settle a position based on market resolution."""
        results = []
        
        with db_manager.get_db_session() as db:
            # Find all positions for this market
            positions = db.query(PaperTradingPosition).filter_by(
                session_id=self.session_id,
                market_id=market_id,
                status=PositionStatus.OPEN
            ).all()
            
            for position in positions:
                # Determine result
                if resolved_outcome.upper() == position.outcome.value.upper():
                    result = Result.WON
                    final_value = position.stake * position.avg_odds
                    pnl = final_value - position.execution_stake
                else:
                    result = Result.LOST
                    final_value = Decimal('0')
                    pnl = -position.execution_stake
                    
                # Update position
                position.status = PositionStatus.CLOSED
                position.closed_at = datetime.now(timezone.utc)
                position.result = result
                position.resolved_outcome = Outcome[resolved_outcome.upper()]
                position.final_value = final_value
                position.pnl = pnl
                
                # Create close trade
                trade = PaperTradingTrade(
                    position_id=position.position_id,
                    session_id=self.session_id,
                    trade_type=TradeType.CLOSE,
                    stake_delta=-position.stake if result == Result.LOST else final_value - position.stake,
                    odds=position.avg_odds,
                    fee_amount=Decimal('0')
                )
                db.add(trade)
                
                results.append({
                    'outcome': position.outcome.value,
                    'result': result.value,
                    'pnl': float(pnl)
                })
                
            # Update snapshot
            self._update_snapshot(db)
            db.commit()
            
        return {'settled': len(results), 'results': results}
        
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        with db_manager.get_db_session() as db:
            # Get all positions
            positions = db.query(PaperTradingPosition).filter_by(
                session_id=self.session_id
            ).all()
            
            closed = [p for p in positions if p.status == PositionStatus.CLOSED]
            
            # Calculate stats
            stats = {
                'total_trades': len(positions),
                'open_trades': len(positions) - len(closed),
                'closed_trades': len(closed),
                'winning_trades': sum(1 for p in closed if p.result == Result.WON),
                'losing_trades': sum(1 for p in closed if p.result == Result.LOST),
                'total_pnl': sum(float(p.pnl) for p in closed if p.pnl),
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
            
            if closed:
                wins = [float(p.pnl) for p in closed if p.result == Result.WON]
                losses = [-float(p.pnl) for p in closed if p.result == Result.LOST]
                
                stats['win_rate'] = len(wins) / len(closed) if closed else 0
                stats['avg_win'] = sum(wins) / len(wins) if wins else 0
                stats['avg_loss'] = sum(losses) / len(losses) if losses else 0
                
                total_wins = sum(wins)
                total_losses = sum(losses)
                stats['profit_factor'] = total_wins / total_losses if total_losses > 0 else float('inf')
                
            return stats
            
    def _store_market_name(self, db, market_id: str, market_name: str):
        """Store market name in lookup table."""
        existing = db.query(MarketName).filter_by(market_id=market_id).first()
        if existing:
            return
            
        # Parse teams
        home_team = 'Unknown'
        away_team = 'Unknown'
        
        if '_vs_' in market_name:
            parts = market_name.split('_vs_')
            home_team = parts[0]
            away_team = parts[1] if len(parts) > 1 else 'Unknown'
        elif ' vs ' in market_name:
            parts = market_name.split(' vs ')
            home_team = parts[0]
            away_team = parts[1] if len(parts) > 1 else 'Unknown'
            
        market = MarketName(
            market_id=market_id,
            market_name=market_name,
            home_team=home_team,
            away_team=away_team
        )
        db.add(market)
        
    def _update_snapshot(self, db):
        """Update performance snapshot."""
        # Get current stats
        cash = self.get_current_bankroll()
        
        positions = db.query(PaperTradingPosition).filter_by(
            session_id=self.session_id,
            status=PositionStatus.OPEN
        ).all()
        
        positions_value = sum(float(pos.stake) for pos in positions)
        
        closed = db.query(PaperTradingPosition).filter_by(
            session_id=self.session_id,
            status=PositionStatus.CLOSED
        ).all()
        
        # Create snapshot
        snapshot = PaperTradingSnapshot(
            session_id=self.session_id,
            cash_balance=Decimal(str(cash)),
            positions_value=Decimal(str(positions_value)),
            portfolio_value=Decimal(str(cash + positions_value)),
            total_pnl=Decimal(str(sum(float(p.pnl) for p in closed if p.pnl))),
            win_count=sum(1 for p in closed if p.result == Result.WON),
            loss_count=sum(1 for p in closed if p.result == Result.LOST),
            pending_count=len(positions)
        )
        db.add(snapshot)