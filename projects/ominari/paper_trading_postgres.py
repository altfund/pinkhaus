#!/usr/bin/env python3
"""
PostgreSQL-based Paper Trading Session Manager
Replaces JSON file storage with proper database transactions
"""
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timezone
import json
import logging
from typing import Dict, List, Any, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class PostgresPaperTradingManager:
    """Manages paper trading sessions using PostgreSQL for data persistence."""
    
    def __init__(self):
        self.db_config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'ominari_trading',
            'user': 'ominari_user',
            'password': 'ominari_password'
        }
        self._ensure_tables_exist()
        
    def get_connection(self):
        """Get database connection."""
        return psycopg2.connect(**self.db_config, cursor_factory=RealDictCursor)
    
    def _ensure_tables_exist(self):
        """Ensure paper trading tables exist."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Create sessions table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trading_session (
                        session_id VARCHAR(255) PRIMARY KEY,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        initial_bankroll DECIMAL(15,2) DEFAULT 10000,
                        current_bankroll DECIMAL(15,2) DEFAULT 10000,
                        status VARCHAR(50) DEFAULT 'active',
                        strategy_config JSONB
                    )
                """)
                
                # Create trades table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_trade (
                        trade_id SERIAL PRIMARY KEY,
                        session_id VARCHAR(255) REFERENCES paper_trading_session(session_id),
                        market_id VARCHAR(255),
                        market_name VARCHAR(500),
                        outcome VARCHAR(50),
                        odds DECIMAL(10,4),
                        stake DECIMAL(15,2),
                        potential_return DECIMAL(15,2),
                        actual_return DECIMAL(15,2),
                        status VARCHAR(50) DEFAULT 'open',
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                        settled_at TIMESTAMP WITH TIME ZONE,
                        pnl DECIMAL(15,2),
                        edge DECIMAL(10,4),
                        maturity_date TIMESTAMP WITH TIME ZONE
                    )
                """)
                
                # Create index for performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_trade_session 
                    ON paper_trade(session_id, status)
                """)
                
                conn.commit()
                logger.info("Paper trading tables verified/created")
    
    def create_session(self, initial_bankroll: float = 10000, strategy_config: dict = None) -> str:
        """Create new trading session."""
        session_id = datetime.now().strftime("S%Y%m%d%H%M%S%f")
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO paper_trading_session 
                    (session_id, initial_bankroll, current_bankroll, strategy_config)
                    VALUES (%s, %s, %s, %s)
                """, (session_id, initial_bankroll, initial_bankroll, 
                      json.dumps(strategy_config or {})))
                conn.commit()
                
        logger.info(f"Created new session: {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session details."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        session_id,
                        created_at,
                        initial_bankroll,
                        current_bankroll,
                        status,
                        strategy_config,
                        (SELECT COUNT(*) FROM paper_trade WHERE session_id = %s) as total_trades,
                        (SELECT COUNT(*) FROM paper_trade WHERE session_id = %s AND status = 'open') as open_trades,
                        (SELECT COALESCE(SUM(stake), 0) FROM paper_trade WHERE session_id = %s AND status = 'open') as exposure,
                        (SELECT COALESCE(SUM(pnl), 0) FROM paper_trade WHERE session_id = %s AND status = 'settled') as total_pnl
                    FROM paper_trading_session
                    WHERE session_id = %s
                """, (session_id, session_id, session_id, session_id, session_id))
                
                session = cur.fetchone()
                if session:
                    # Calculate exposure percentage
                    exposure_pct = 0
                    if session['initial_bankroll'] > 0:
                        exposure_pct = float(session['exposure']) / float(session['initial_bankroll']) * 100
                    
                    return {
                        'session_id': session['session_id'],
                        'created_at': session['created_at'],
                        'initial_bankroll': float(session['initial_bankroll']),
                        'current_bankroll': float(session['current_bankroll']),
                        'status': session['status'],
                        'strategy_config': session['strategy_config'],
                        'total_trades': session['total_trades'],
                        'open_trades': session['open_trades'],
                        'exposure': float(session['exposure']),
                        'exposure_pct': exposure_pct,
                        'total_pnl': float(session['total_pnl']),
                        'portfolio_value': float(session['current_bankroll']) + float(session['exposure'])
                    }
                return None
    
    def get_current_session(self) -> Optional[Dict]:
        """Get the most recent active session."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT session_id 
                    FROM paper_trading_session 
                    WHERE status = 'active' 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """)
                result = cur.fetchone()
                if result:
                    return self.get_session(result['session_id'])
                return None
    
    def record_trades(self, session_id: str, trades: List[Dict]):
        """Record multiple trades in a transaction."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                for trade in trades:
                    # Insert trade
                    cur.execute("""
                        INSERT INTO paper_trade (
                            session_id, market_id, market_name, outcome, 
                            odds, stake, potential_return, edge, maturity_date
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        session_id,
                        trade['market_id'],
                        trade['market_name'],
                        trade['outcome'],
                        trade['odds'],
                        trade['stake'],
                        trade['stake'] * trade['odds'],
                        trade.get('edge', 0),
                        trade.get('maturity_date')
                    ))
                
                # Update session bankroll
                total_stake = sum(t['stake'] for t in trades)
                cur.execute("""
                    UPDATE paper_trading_session 
                    SET current_bankroll = current_bankroll - %s 
                    WHERE session_id = %s
                """, (total_stake, session_id))
                
                conn.commit()
                logger.info(f"Recorded {len(trades)} trades for session {session_id}")
    
    def get_positions(self, session_id: str) -> Dict:
        """Get all positions for a session."""
        positions = {}
        closed_positions = []
        
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Get open positions
                cur.execute("""
                    SELECT * FROM paper_trade 
                    WHERE session_id = %s AND status = 'open'
                    ORDER BY created_at DESC
                """, (session_id,))
                
                for row in cur.fetchall():
                    market_key = f"{row['market_name']}_{row['outcome']}"
                    if market_key not in positions:
                        positions[market_key] = {
                            'market_name': row['market_name'],
                            'outcome': row['outcome'],
                            'trades': [],
                            'total_stake': 0,
                            'avg_odds': 0,
                            'current_value': 0,
                            'pnl': 0,
                            'status': 'open'
                        }
                    
                    positions[market_key]['trades'].append({
                        'trade_id': row['trade_id'],
                        'stake': float(row['stake']),
                        'odds': float(row['odds'])
                    })
                    positions[market_key]['total_stake'] += float(row['stake'])
                
                # Calculate weighted average odds
                for pos in positions.values():
                    total_weight = sum(t['stake'] * t['odds'] for t in pos['trades'])
                    pos['avg_odds'] = total_weight / pos['total_stake'] if pos['total_stake'] > 0 else 0
                    pos['current_value'] = pos['total_stake']  # For now, assuming no market movement
                
                # Get closed positions (last 20)
                cur.execute("""
                    SELECT 
                        market_name, outcome, 
                        SUM(stake) as total_stake,
                        AVG(odds) as avg_odds,
                        SUM(pnl) as pnl,
                        MAX(settled_at) as settled_at
                    FROM paper_trade 
                    WHERE session_id = %s AND status = 'settled'
                    GROUP BY market_name, outcome
                    ORDER BY MAX(settled_at) DESC
                    LIMIT 20
                """, (session_id,))
                
                for row in cur.fetchall():
                    closed_positions.append({
                        'market_name': row['market_name'],
                        'outcome': row['outcome'],
                        'total_stake': float(row['total_stake']),
                        'avg_odds': float(row['avg_odds']),
                        'pnl': float(row['pnl']),
                        'result': 'won' if row['pnl'] > 0 else 'lost',
                        'settled_at': row['settled_at']
                    })
        
        return {'open': positions, 'closed': closed_positions}
    
    def get_performance_stats(self, session_id: str) -> Dict:
        """Get performance statistics for a session."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(CASE WHEN status = 'settled' THEN 1 END) as settled_trades,
                        COUNT(CASE WHEN status = 'settled' AND pnl > 0 THEN 1 END) as wins,
                        COUNT(CASE WHEN status = 'settled' AND pnl < 0 THEN 1 END) as losses,
                        COALESCE(SUM(CASE WHEN status = 'settled' AND pnl > 0 THEN pnl ELSE 0 END), 0) as gross_wins,
                        COALESCE(SUM(CASE WHEN status = 'settled' AND pnl < 0 THEN ABS(pnl) ELSE 0 END), 0) as gross_losses,
                        COALESCE(SUM(CASE WHEN status = 'settled' THEN pnl ELSE 0 END), 0) as net_pnl,
                        COALESCE(SUM(CASE WHEN status = 'settled' THEN stake ELSE 0 END), 0) as total_wagered
                    FROM paper_trade
                    WHERE session_id = %s
                """, (session_id,))
                
                stats = cur.fetchone()
                
                # Calculate derived metrics
                win_rate = 0
                if stats['settled_trades'] > 0:
                    win_rate = (stats['wins'] / stats['settled_trades']) * 100
                
                roi = 0
                if stats['total_wagered'] > 0:
                    roi = (stats['net_pnl'] / stats['total_wagered']) * 100
                
                profit_factor = 0
                if stats['gross_losses'] > 0:
                    profit_factor = stats['gross_wins'] / stats['gross_losses']
                
                return {
                    'total_trades': stats['total_trades'],
                    'settled_trades': stats['settled_trades'],
                    'wins': stats['wins'],
                    'losses': stats['losses'],
                    'win_rate': win_rate,
                    'roi': roi,
                    'profit_factor': profit_factor,
                    'net_pnl': float(stats['net_pnl']),
                    'total_wagered': float(stats['total_wagered'])
                }
    
    def reset_session(self, session_id: str):
        """Reset a session (for emergency use)."""
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                # Delete all trades
                cur.execute("DELETE FROM paper_trade WHERE session_id = %s", (session_id,))
                
                # Reset session
                cur.execute("""
                    UPDATE paper_trading_session 
                    SET current_bankroll = initial_bankroll
                    WHERE session_id = %s
                """, (session_id,))
                
                conn.commit()
                logger.info(f"Reset session {session_id}")

# Export the manager
paper_trading_manager = PostgresPaperTradingManager()