#!/usr/bin/env python3
"""
PostgreSQL-based Paper Trading Session Manager
Drop-in replacement for JSON-based session manager
Uses the same database as web_monitor.py
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
import json
import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Use the same database configuration as web_monitor.py
DB_CONFIG = {
    'host': os.environ.get('PG_HOST', 'localhost'),
    'port': os.environ.get('PG_PORT', '5999'),
    'database': os.environ.get('PG_DB', 'ominari_production'),
    'user': os.environ.get('PG_USER', 'ominari_user'),
    'password': os.environ.get('PG_PASSWORD', 'ominari_2025_secure')
}

class PaperTradingSessionManager:
    """PostgreSQL-based paper trading session manager with JSON-compatible interface."""
    
    def __init__(self, sessions_file: str = None):
        """Initialize manager. sessions_file parameter ignored for compatibility."""
        self.db_config = DB_CONFIG
        self.current_session_id = None
        self._ensure_tables_exist()
        
        # Load current session if exists
        session = self._get_current_session_from_db()
        if session:
            self.current_session_id = session['session_id']
        
    @contextmanager
    def get_connection(self):
        """Get database connection with context management."""
        conn = psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
        try:
            yield conn
        finally:
            conn.close()
    
    def _ensure_tables_exist(self):
        """Ensure paper trading tables exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Use paper_trading_sessions (with 's') to match create_paper_trading_tables.py
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_sessions (
                        session_id VARCHAR(255) PRIMARY KEY,
                        session_name VARCHAR(255),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        initial_bankroll DECIMAL(15,2) DEFAULT 10000,
                        status VARCHAR(50) DEFAULT 'active',
                        strategy_config JSONB,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
                
                # Use paper_trading_positions to match existing schema
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_positions (
                        position_id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES paper_trading_sessions(session_id),
                        match_id VARCHAR(500) NOT NULL,
                        bet_id VARCHAR(255) UNIQUE NOT NULL,
                        sport VARCHAR(100),
                        home_team VARCHAR(255),
                        away_team VARCHAR(255),
                        bet_type VARCHAR(50),
                        bet_on VARCHAR(50),
                        odds DECIMAL(10, 4),
                        stake DECIMAL(15, 2),
                        potential_return DECIMAL(15, 2),
                        actual_return DECIMAL(15, 2),
                        status VARCHAR(50) DEFAULT 'pending',
                        placed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        settled_at TIMESTAMP WITH TIME ZONE,
                        result VARCHAR(50),
                        is_winner BOOLEAN,
                        pnl DECIMAL(15, 2),
                        running_balance DECIMAL(15, 2),
                        signal_name VARCHAR(100),
                        signal_value DECIMAL(10, 4),
                        edge DECIMAL(10, 4),
                        kickoff_time TIMESTAMP WITH TIME ZONE,
                        metadata JSONB DEFAULT '{}'::jsonb
                    )
                """)
                
                # Create snapshot table for session state
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_snapshots (
                        snapshot_id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) NOT NULL,
                        snapshot_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        cash_balance DECIMAL(15, 2) DEFAULT 10000,
                        positions_value DECIMAL(15, 2) DEFAULT 0,
                        portfolio_value DECIMAL(15, 2) DEFAULT 10000,
                        total_pnl DECIMAL(15, 2) DEFAULT 0,
                        daily_pnl DECIMAL(15, 2) DEFAULT 0,
                        win_count INTEGER DEFAULT 0,
                        loss_count INTEGER DEFAULT 0,
                        pending_count INTEGER DEFAULT 0,
                        max_drawdown DECIMAL(10, 4) DEFAULT 0,
                        sharpe_ratio DECIMAL(10, 4),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes
                cur.execute("CREATE INDEX IF NOT EXISTS idx_ptp_session_status ON paper_trading_positions(session_id, status)")
                cur.execute("CREATE INDEX IF NOT EXISTS idx_pts_session_time ON paper_trading_snapshots(session_id, snapshot_time DESC)")
                
                conn.commit()
                logger.info("Paper trading PostgreSQL tables verified/created")
    
    def create_session(self, initial_bankroll: float = 10000.0, 
                      session_name: str = None) -> str:
        """Create new trading session (JSON-compatible interface)."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if not session_name:
            session_name = f"Paper Trading {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Insert session
                cur.execute("""
                    INSERT INTO paper_trading_sessions 
                    (session_id, session_name, initial_bankroll, status, metadata)
                    VALUES (%s, %s, %s, %s, %s)
                """, (session_id, session_name, initial_bankroll, 'active', 
                      json.dumps({'created_via': 'postgres_integrated'})))
                
                # Insert initial snapshot
                cur.execute("""
                    INSERT INTO paper_trading_snapshots 
                    (session_id, cash_balance, portfolio_value)
                    VALUES (%s, %s, %s)
                """, (session_id, initial_bankroll, initial_bankroll))
                
                conn.commit()
                
        self.current_session_id = session_id
        logger.info(f"Created PostgreSQL session: {session_id}")
        return session_id
    
    def _get_current_session_from_db(self) -> Optional[Dict]:
        """Get the most recent active session."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id, session_name, initial_bankroll, created_at, status
                    FROM paper_trading_sessions 
                    WHERE status = 'active' OR status = 'ACTIVE'
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                return cur.fetchone()
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data (JSON-compatible interface)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get session details
                cur.execute("""
                    SELECT s.*, 
                           snap.cash_balance,
                           snap.portfolio_value,
                           snap.total_pnl,
                           snap.win_count,
                           snap.loss_count,
                           snap.pending_count
                    FROM paper_trading_sessions s
                    LEFT JOIN LATERAL (
                        SELECT * FROM paper_trading_snapshots 
                        WHERE session_id = s.session_id 
                        ORDER BY snapshot_time DESC 
                        LIMIT 1
                    ) snap ON true
                    WHERE s.session_id = %s
                """, (session_id,))
                
                session = cur.fetchone()
                if not session:
                    return None
                
                # Get positions summary
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN status = 'pending' OR status = 'open' THEN 1 END) as open_trades,
                        COALESCE(SUM(CASE WHEN status = 'pending' OR status = 'open' THEN stake ELSE 0 END), 0) as exposure
                    FROM paper_trading_positions
                    WHERE session_id = %s
                """, (session_id,))
                
                pos_summary = cur.fetchone()
                
                # Build JSON-compatible response
                return {
                    'session_id': session_id,
                    'session_name': session.get('session_name', ''),
                    'created_at': session.get('created_at').isoformat() if session.get('created_at') else None,
                    'initial_bankroll': float(session.get('initial_bankroll', 10000)),
                    'current_bankroll': float(session.get('cash_balance', 10000)),
                    'portfolio_value': float(session.get('portfolio_value', 10000)),
                    'total_pnl': float(session.get('total_pnl', 0)),
                    'status': session.get('status', 'active'),
                    'total_bets': pos_summary['total_trades'],
                    'open_positions': pos_summary['open_trades'],
                    'exposure': float(pos_summary['exposure']),
                    'wins': session.get('win_count', 0),
                    'losses': session.get('loss_count', 0),
                    'pending': session.get('pending_count', 0)
                }
    
    def get_current_session(self) -> Optional[str]:
        """Get current session ID (JSON-compatible interface)."""
        if self.current_session_id:
            return self.current_session_id
        
        session = self._get_current_session_from_db()
        if session:
            self.current_session_id = session['session_id']
            return self.current_session_id
        return None
    
    def update_session(self, session_id: str, update_data: Dict[str, Any]):
        """Update session data (JSON-compatible interface)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Extract values
                cash = update_data.get('current_bankroll')
                total_pnl = update_data.get('total_pnl', 0)
                
                # Calculate portfolio value
                exposure = update_data.get('exposure', 0)
                portfolio_value = cash + exposure if cash else None
                
                # Insert new snapshot
                if cash is not None:
                    cur.execute("""
                        INSERT INTO paper_trading_snapshots
                        (session_id, cash_balance, positions_value, portfolio_value, 
                         total_pnl, win_count, loss_count, pending_count)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        session_id,
                        cash,
                        exposure,
                        portfolio_value,
                        total_pnl,
                        update_data.get('wins', 0),
                        update_data.get('losses', 0),
                        update_data.get('pending', 0)
                    ))
                
                conn.commit()
    
    def record_bet(self, session_id: str, bet_data: Dict[str, Any]):
        """Record a new bet (JSON-compatible interface)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create unique bet ID
                bet_id = f"{session_id}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                
                # Convert numpy types to Python types
                odds = float(bet_data.get('odds', 0))
                stake = float(bet_data.get('stake', 0))
                signal_value = float(bet_data.get('signal_value', 0))
                edge = float(bet_data.get('edge', 0))
                
                cur.execute("""
                    INSERT INTO paper_trading_positions (
                        session_id, bet_id, match_id, sport, home_team, away_team,
                        bet_type, bet_on, odds, stake, potential_return,
                        signal_name, signal_value, edge, kickoff_time
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session_id,
                    bet_id,
                    str(bet_data.get('match_id', '')),
                    str(bet_data.get('sport', '')),
                    str(bet_data.get('home_team', '')),
                    str(bet_data.get('away_team', '')),
                    str(bet_data.get('bet_type', 'moneyline')),
                    str(bet_data.get('bet_on', '')),
                    odds,
                    stake,
                    stake * odds,
                    str(bet_data.get('signal_name', '')),
                    signal_value,
                    edge,
                    bet_data.get('kickoff_time')
                ))
                
                conn.commit()
                return bet_id
    
    def get_positions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all positions for a session (JSON-compatible interface)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT * FROM paper_trading_positions
                    WHERE session_id = %s
                    ORDER BY placed_at DESC
                """, (session_id,))
                
                positions = []
                for row in cur.fetchall():
                    positions.append({
                        'bet_id': row['bet_id'],
                        'match_id': row['match_id'],
                        'sport': row['sport'],
                        'home_team': row['home_team'],
                        'away_team': row['away_team'],
                        'bet_type': row['bet_type'],
                        'bet_on': row['bet_on'],
                        'odds': float(row['odds']),
                        'stake': float(row['stake']),
                        'potential_return': float(row['potential_return']),
                        'status': row['status'],
                        'placed_at': row['placed_at'].isoformat() if row['placed_at'] else None,
                        'signal_name': row['signal_name'],
                        'signal_value': float(row['signal_value']) if row['signal_value'] else 0,
                        'edge': float(row['edge']) if row['edge'] else 0
                    })
                
                return positions
    
    def close_session(self, session_id: str):
        """Close a session (JSON-compatible interface)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE paper_trading_sessions 
                    SET status = 'closed'
                    WHERE session_id = %s
                """, (session_id,))
                conn.commit()
        
        if self.current_session_id == session_id:
            self.current_session_id = None
    
    # Compatibility methods for JSON interface
    def _load_sessions(self) -> Dict[str, Any]:
        """Compatibility method - returns empty dict."""
        return {"sessions": {}, "metadata": {}}
    
    def _save_sessions(self):
        """Compatibility method - no-op for PostgreSQL."""
        pass
    
    @property
    def sessions(self) -> Dict[str, Any]:
        """Compatibility property - returns session data."""
        sessions_data = {"sessions": {}, "metadata": {"last_updated": datetime.now().isoformat()}}
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id FROM paper_trading_sessions 
                    WHERE status = 'active' OR status = 'ACTIVE'
                    ORDER BY created_at DESC LIMIT 5
                """)
                
                for row in cur.fetchall():
                    session_id = row['session_id']
                    session_data = self.get_session(session_id)
                    if session_data:
                        sessions_data["sessions"][session_id] = session_data
        
        return sessions_data
    
    def record_trades(self, session_id: str, trades: List[Dict[str, Any]]):
        """Record multiple trades at once."""
        for trade in trades:
            self.record_bet(session_id, trade)
    
    def get_session_performance(self, session_id: str) -> Dict[str, Any]:
        """Get session performance metrics."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get basic session info
                cur.execute("""
                    SELECT 
                        s.*,
                        snap.cash_balance,
                        snap.portfolio_value,
                        snap.total_pnl,
                        snap.win_count,
                        snap.loss_count
                    FROM paper_trading_sessions s
                    LEFT JOIN LATERAL (
                        SELECT * FROM paper_trading_snapshots
                        WHERE session_id = s.session_id
                        ORDER BY snapshot_time DESC LIMIT 1
                    ) snap ON true
                    WHERE s.session_id = %s
                """, (session_id,))
                
                session = cur.fetchone()
                if not session:
                    return {}
                
                # Get trade statistics
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN status = 'won' THEN 1 END) as wins,
                        COUNT(CASE WHEN status = 'lost' THEN 1 END) as losses,
                        AVG(CASE WHEN status IN ('won', 'lost') AND stake > 0 
                            THEN pnl/stake ELSE NULL END) as avg_return,
                        MAX(pnl) as best_trade,
                        MIN(pnl) as worst_trade
                    FROM paper_trading_positions
                    WHERE session_id = %s
                """, (session_id,))
                
                stats = cur.fetchone()
                
                # Calculate additional metrics
                win_rate = (stats['wins'] / stats['total_trades'] * 100) if stats['total_trades'] > 0 else 0
                pnl_pct = (float(session['total_pnl']) / float(session['initial_bankroll']) * 100) if session['initial_bankroll'] else 0
                
                return {
                    'total_trades': stats['total_trades'] or 0,
                    'wins': stats['wins'] or 0,
                    'losses': stats['losses'] or 0,
                    'win_rate': win_rate,
                    'total_pnl': float(session['total_pnl']) if session['total_pnl'] else 0,
                    'pnl_percentage': pnl_pct,
                    'portfolio_value': float(session['portfolio_value']) if session['portfolio_value'] else float(session['initial_bankroll']),
                    'cash_balance': float(session['cash_balance']) if session['cash_balance'] else float(session['initial_bankroll']),
                    'avg_return': float(stats['avg_return'] * 100) if stats['avg_return'] else 0,
                    'best_trade': float(stats['best_trade']) if stats['best_trade'] else 0,
                    'worst_trade': float(stats['worst_trade']) if stats['worst_trade'] else 0,
                    'status': 'active'
                }