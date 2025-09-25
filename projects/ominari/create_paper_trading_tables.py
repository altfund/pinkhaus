#!/usr/bin/env python3
"""Create paper trading tables in PostgreSQL."""

import os
from sqlalchemy import create_engine, text

# Set environment for PostgreSQL on port 5999
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

def create_tables():
    """Create missing paper trading tables."""
    engine = create_engine(
        f"postgresql://{os.environ['PG_USER']}:{os.environ['PG_PASSWORD']}@{os.environ['PG_HOST']}:{os.environ['PG_PORT']}/{os.environ['PG_DB']}"
    )
    
    with engine.connect() as conn:
        # Create paper trading sessions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trading_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                session_name VARCHAR(255),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                initial_bankroll DECIMAL(15, 2) DEFAULT 10000,
                status VARCHAR(50) DEFAULT 'active',
                strategy_config JSONB
            );
        """))
        
        # Create paper trading snapshots table
        conn.execute(text("""
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
            );
        """))
        
        # Create paper trading positions table
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trading_positions (
                position_id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
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
                kickoff_time TIMESTAMP WITH TIME ZONE
            );
        """))
        
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_pts_session_time ON paper_trading_snapshots(session_id, snapshot_time DESC);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ptp_session_status ON paper_trading_positions(session_id, status);"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_ptp_placed_at ON paper_trading_positions(placed_at DESC);"))
        
        # Insert initial session if needed
        session_result = conn.execute(text(
            "SELECT COUNT(*) FROM paper_trading_sessions WHERE session_id = '20250912_181426'"
        )).scalar()
        
        if session_result == 0:
            conn.execute(text("""
                INSERT INTO paper_trading_sessions (
                    session_id, session_name, initial_bankroll, status
                ) VALUES (
                    '20250912_181426', 'Default Session', 10000, 'ACTIVE'
                )
            """))
        
        # Insert initial snapshot if needed
        result = conn.execute(text(
            "SELECT COUNT(*) FROM paper_trading_snapshots WHERE session_id = '20250912_181426'"
        )).scalar()
        
        if result == 0:
            conn.execute(text("""
                INSERT INTO paper_trading_snapshots (
                    session_id, cash_balance, positions_value, 
                    portfolio_value, total_pnl, daily_pnl
                ) VALUES (
                    '20250912_181426', 10000, 0, 10000, 0, 0
                )
            """))
        
        conn.commit()
        
    print("✅ Paper trading tables created successfully!")
    
    # Verify tables
    with engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name LIKE 'paper_trading%'
            ORDER BY table_name
        """)).fetchall()
        
        print("\nCreated tables:")
        for table in tables:
            print(f"  - {table[0]}")

if __name__ == "__main__":
    create_tables()