#!/usr/bin/env python3
"""Migrate paper trading from JSON to optimized database schema."""

import json
import logging
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, Any

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_v2 import db_manager
from paper_trading_models_v2 import (
    Base, PaperTradingSession, PaperTradingPosition, PaperTradingTrade,
    PaperTradingSnapshot, MarketName, PositionStatus, SessionStatus,
    TradeType, Outcome, Result
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PaperTradingMigrator:
    """Migrate paper trading data from JSON to database."""
    
    def __init__(self):
        """Initialize migrator."""
        self.session = None
        self.market_names = {}  # Cache for market names
        
    def migrate(self):
        """Main migration process."""
        try:
            # Load JSON data
            logger.info("Loading paper trading sessions from JSON...")
            with open('paper_trading_sessions.json', 'r') as f:
                data = json.load(f)
            
            sessions = data.get('sessions', {})
            logger.info(f"Found {len(sessions)} sessions to migrate")
            
            # Create tables
            logger.info("Creating optimized database tables...")
            with db_manager.get_db_session() as db:
                Base.metadata.create_all(bind=db.bind)
            
            # Migrate each session
            for session_id, session_data in sessions.items():
                self._migrate_session(session_id, session_data)
            
            logger.info("Migration completed successfully!")
            
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            raise
    
    def _migrate_session(self, session_id: str, data: Dict[str, Any]):
        """Migrate a single session."""
        logger.info(f"Migrating session {session_id}...")
        
        with db_manager.get_db_session() as db:
            try:
                # Check if session already exists
                existing = db.query(PaperTradingSession).filter_by(
                    session_id=session_id
                ).first()
                
                if existing:
                    logger.warning(f"Session {session_id} already exists, skipping")
                    return
                
                # Create session
                session = PaperTradingSession(
                    session_id=session_id,
                    session_name=data.get('session_name', f"Session {session_id}"),
                    created_at=self._parse_datetime(data.get('created_at')),
                    initial_bankroll=Decimal(str(data.get('initial_bankroll', 10000))),
                    status=SessionStatus.ACTIVE if data.get('status') == 'active' else SessionStatus.COMPLETED,
                    strategy_config={
                        'min_bet': data.get('min_bet', 10),
                        'kelly_fraction': data.get('kelly_fraction', 0.25)
                    }
                )
                db.add(session)
                db.flush()  # Get session_id for foreign keys
                
                # Migrate positions
                positions = data.get('positions', {})
                closed_positions = data.get('closed_positions', [])
                
                # Open positions
                for pos_key, pos_data in positions.items():
                    self._migrate_position(db, session_id, pos_data, is_open=True)
                
                # Closed positions
                for pos_data in closed_positions:
                    self._migrate_position(db, session_id, pos_data, is_open=False)
                
                # Create current snapshot
                self._create_snapshot(db, session_id, data)
                
                db.commit()
                logger.info(f"Session {session_id} migrated successfully")
                
            except Exception as e:
                db.rollback()
                logger.error(f"Failed to migrate session {session_id}: {e}")
                raise
    
    def _migrate_position(self, db, session_id: str, pos_data: Dict[str, Any], is_open: bool):
        """Migrate a single position."""
        market_id = pos_data.get('market_id')
        if not market_id:
            logger.warning("Position missing market_id, skipping")
            return
        
        # Store market name
        market_name = pos_data.get('market_name', 'Unknown')
        self._store_market_name(db, market_id, market_name)
        
        # Get fee info
        fee_info = pos_data.get('fee_info', {})
        safebox_fee = fee_info.get('safebox_fee_pct', 0.02)
        skew_fee = fee_info.get('skew_fee_pct', 0.01)
        
        # Create position
        position = PaperTradingPosition(
            session_id=session_id,
            market_id=market_id,
            outcome=self._parse_outcome(pos_data.get('outcome')),
            stake=Decimal(str(pos_data.get('total_stake', 0))),
            execution_stake=Decimal(str(pos_data.get('execution_stake', 0))),
            avg_odds=Decimal(str(pos_data.get('avg_odds', 0))),
            safebox_fee_bps=int(safebox_fee * 10000),  # Convert to basis points
            skew_fee_bps=int(skew_fee * 10000),
            status=PositionStatus.OPEN if is_open else PositionStatus.CLOSED,
            opened_at=self._parse_datetime(pos_data.get('opened_at')),
            closed_at=self._parse_datetime(pos_data.get('closed_at')) if not is_open else None,
            maturity_date=self._parse_datetime(pos_data.get('maturity_date'))
        )
        
        # Add results for closed positions
        if not is_open:
            position.final_value = Decimal(str(pos_data.get('final_value', 0)))
            position.pnl = Decimal(str(pos_data.get('pnl', 0)))
            position.result = self._parse_result(pos_data.get('result'))
            position.resolved_outcome = self._parse_outcome(pos_data.get('resolved_outcome'))
        
        db.add(position)
        db.flush()  # Get position_id
        
        # Migrate trades
        trades = pos_data.get('trades', [])
        for trade_data in trades:
            self._migrate_trade(db, session_id, position.position_id, trade_data)
    
    def _migrate_trade(self, db, session_id: str, position_id: int, trade_data: Dict[str, Any]):
        """Migrate a single trade."""
        trade_type = trade_data.get('type', 'open')
        
        trade = PaperTradingTrade(
            position_id=position_id,
            session_id=session_id,
            trade_type=self._parse_trade_type(trade_type),
            stake_delta=Decimal(str(trade_data.get('stake', 0))),
            odds=Decimal(str(trade_data.get('odds', 0))),
            fee_amount=Decimal(str(trade_data.get('fee_info', {}).get('fee_amount', 0))),
            traded_at=self._parse_datetime(trade_data.get('timestamp'))
        )
        db.add(trade)
    
    def _create_snapshot(self, db, session_id: str, data: Dict[str, Any]):
        """Create a performance snapshot."""
        perf = data.get('performance', {})
        
        snapshot = PaperTradingSnapshot(
            session_id=session_id,
            cash_balance=Decimal(str(data.get('current_bankroll', 0))),
            positions_value=Decimal(str(
                sum(p.get('current_value', 0) for p in data.get('positions', {}).values())
            )),
            portfolio_value=Decimal(str(data.get('portfolio_value', 0))),
            total_pnl=Decimal(str(perf.get('total_pnl', 0))),
            daily_pnl=None,  # Will calculate later
            win_count=perf.get('winning_trades', 0),
            loss_count=perf.get('losing_trades', 0),
            pending_count=perf.get('pending_trades', 0),
            max_drawdown=Decimal(str(perf.get('max_drawdown', 0) * 100)) if perf.get('max_drawdown') else None,
            sharpe_ratio=Decimal(str(perf.get('sharpe_ratio', 0))) if perf.get('sharpe_ratio') else None
        )
        db.add(snapshot)
    
    def _store_market_name(self, db, market_id: str, market_name: str):
        """Store market name in lookup table."""
        if market_id in self.market_names:
            return
        
        # Check if already exists
        existing = db.query(MarketName).filter_by(market_id=market_id).first()
        if existing:
            self.market_names[market_id] = existing
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
        self.market_names[market_id] = market
    
    def _parse_datetime(self, dt_str: str) -> datetime:
        """Parse datetime string."""
        if not dt_str:
            return datetime.now(timezone.utc)
        
        try:
            # Try ISO format with timezone
            if 'T' in dt_str:
                return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
            else:
                return datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
        except:
            return datetime.now(timezone.utc)
    
    def _parse_outcome(self, outcome: str) -> Outcome:
        """Parse outcome string to enum."""
        if not outcome:
            return None
        
        outcome_map = {
            'home': Outcome.HOME,
            'draw': Outcome.DRAW,
            'away': Outcome.AWAY
        }
        return outcome_map.get(outcome.lower(), Outcome[outcome.upper()])
    
    def _parse_result(self, result: str) -> Result:
        """Parse result string to enum."""
        if not result:
            return None
        
        result_map = {
            'won': Result.WON,
            'lost': Result.LOST,
            'void': Result.VOID
        }
        return result_map.get(result.lower())
    
    def _parse_trade_type(self, trade_type: str) -> TradeType:
        """Parse trade type string to enum."""
        type_map = {
            'open': TradeType.OPEN,
            'rebalance': TradeType.REBALANCE,
            'close': TradeType.CLOSE
        }
        return type_map.get(trade_type.lower(), TradeType.OPEN)


if __name__ == "__main__":
    migrator = PaperTradingMigrator()
    migrator.migrate()