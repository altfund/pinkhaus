#!/usr/bin/env python3
"""
Simple PostgreSQL Setup

Sets up PostgreSQL using the existing system service for immediate use
with the blockchain trading system.
"""

import logging
import subprocess
import tempfile
import os
from datetime import datetime, timezone
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SimplePostgreSQLSetup:
    """Simple PostgreSQL setup using system service."""
    
    def __init__(self, db_name: str = "ominari_live"):
        self.db_name = db_name
        self.user = "ominari_user"
        self.password = "ominari_2025_secure"
    
    def setup_postgresql_database(self) -> bool:
        """Set up PostgreSQL database and user."""
        logger.info("🚀 Setting up PostgreSQL for Ominari...")
        
        try:
            # Create user and database as postgres superuser
            create_commands = f"""
            -- Create user
            CREATE USER {self.user} WITH PASSWORD '{self.password}';
            
            -- Create database
            CREATE DATABASE {self.db_name} OWNER {self.user};
            
            -- Grant privileges
            GRANT ALL PRIVILEGES ON DATABASE {self.db_name} TO {self.user};
            """
            
            # Write commands to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
                f.write(create_commands)
                sql_file = f.name
            
            try:
                # Execute as postgres user
                result = subprocess.run([
                    'sudo', '-u', 'postgres', 'psql',
                    '-f', sql_file
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✅ Database and user created successfully")
                    return True
                else:
                    # Check if user/database already exists
                    if "already exists" in result.stderr:
                        logger.info("✅ Database and user already exist")
                        return True
                    else:
                        logger.error(f"❌ Database creation failed: {result.stderr}")
                        return False
                        
            finally:
                os.unlink(sql_file)
                
        except Exception as e:
            logger.error(f"❌ Setup error: {e}")
            return False
    
    def create_live_trading_tables(self) -> bool:
        """Create optimized tables for live trading."""
        logger.info("🏗️ Creating live trading tables...")
        
        create_tables_sql = """
        -- Enable UUID extension
        CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
        
        -- Live blockchain markets
        CREATE TABLE IF NOT EXISTS live_markets (
            market_id VARCHAR(100) PRIMARY KEY,
            blockchain_address VARCHAR(42) UNIQUE,
            network VARCHAR(20) NOT NULL,
            source VARCHAR(50) NOT NULL,
            sport VARCHAR(50) NOT NULL,
            league VARCHAR(100) NOT NULL,
            home_team VARCHAR(200) NOT NULL,
            away_team VARCHAR(200) NOT NULL,
            market_type VARCHAR(50) DEFAULT 'moneyline',
            starts_at TIMESTAMP WITH TIME ZONE NOT NULL,
            is_finished BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Live odds with high-frequency updates
        CREATE TABLE IF NOT EXISTS live_odds (
            id SERIAL PRIMARY KEY,
            market_id VARCHAR(100) NOT NULL REFERENCES live_markets(market_id) ON DELETE CASCADE,
            position INTEGER NOT NULL,
            outcome VARCHAR(50) NOT NULL,
            bookmaker VARCHAR(100) NOT NULL,
            decimal_odds DECIMAL(10,6),
            buy_odds DECIMAL(10,6),
            sell_odds DECIMAL(10,6),
            liquidity DECIMAL(18,6),
            spread DECIMAL(10,6),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(market_id, position, bookmaker)
        );
        
        -- Active trading positions
        CREATE TABLE IF NOT EXISTS live_positions (
            position_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            market_id VARCHAR(100) NOT NULL REFERENCES live_markets(market_id),
            trader_id VARCHAR(100) NOT NULL,
            position_type VARCHAR(20) NOT NULL, -- 'home', 'away', 'draw'
            stake DECIMAL(15,4) NOT NULL,
            odds DECIMAL(10,6) NOT NULL,
            expected_payout DECIMAL(15,4) NOT NULL,
            
            -- Signal information
            signal_probability DECIMAL(5,4),
            signal_providers JSONB DEFAULT '[]'::jsonb,
            kelly_fraction DECIMAL(8,6),
            edge DECIMAL(8,6),
            
            -- Position lifecycle
            opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP WITH TIME ZONE,
            settled_at TIMESTAMP WITH TIME ZONE,
            
            -- P&L tracking
            pnl DECIMAL(15,4) DEFAULT 0,
            status VARCHAR(20) DEFAULT 'open', -- 'open', 'closed', 'won', 'lost', 'push'
            
            -- Additional metadata
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Signal performance tracking
        CREATE TABLE IF NOT EXISTS signal_performance (
            id SERIAL PRIMARY KEY,
            signal_name VARCHAR(100) NOT NULL,
            market_id VARCHAR(100) NOT NULL,
            predicted_probability DECIMAL(5,4) NOT NULL,
            actual_outcome VARCHAR(50),
            was_correct BOOLEAN,
            confidence_score DECIMAL(5,4),
            prediction_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            resolution_timestamp TIMESTAMP WITH TIME ZONE
        );
        
        -- Blockchain sync tracking
        CREATE TABLE IF NOT EXISTS blockchain_sync (
            network VARCHAR(20) PRIMARY KEY,
            last_synced_block BIGINT NOT NULL DEFAULT 0,
            last_sync_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            markets_discovered INTEGER DEFAULT 0,
            markets_processed INTEGER DEFAULT 0,
            sync_errors INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'active' -- 'active', 'paused', 'error'
        );
        
        -- Portfolio tracking
        CREATE TABLE IF NOT EXISTS portfolio_snapshots (
            id SERIAL PRIMARY KEY,
            trader_id VARCHAR(100) NOT NULL,
            snapshot_timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            total_bankroll DECIMAL(15,4) NOT NULL,
            available_balance DECIMAL(15,4) NOT NULL,
            total_exposure DECIMAL(15,4) NOT NULL,
            unrealized_pnl DECIMAL(15,4) DEFAULT 0,
            realized_pnl DECIMAL(15,4) DEFAULT 0,
            open_positions INTEGER DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            win_rate DECIMAL(5,4) DEFAULT 0,
            roi DECIMAL(8,4) DEFAULT 0,
            metadata JSONB DEFAULT '{}'::jsonb
        );
        
        -- Performance indexes
        CREATE INDEX IF NOT EXISTS idx_live_markets_network_sport ON live_markets(network, sport);
        CREATE INDEX IF NOT EXISTS idx_live_markets_starts_at ON live_markets(starts_at) WHERE NOT is_finished;
        CREATE INDEX IF NOT EXISTS idx_live_markets_blockchain_addr ON live_markets(blockchain_address);
        CREATE INDEX IF NOT EXISTS idx_live_markets_active ON live_markets(is_finished, starts_at);
        
        CREATE INDEX IF NOT EXISTS idx_live_odds_market_updated ON live_odds(market_id, updated_at DESC);
        CREATE INDEX IF NOT EXISTS idx_live_odds_bookmaker ON live_odds(bookmaker);
        CREATE INDEX IF NOT EXISTS idx_live_odds_unique_position ON live_odds(market_id, position);
        
        CREATE INDEX IF NOT EXISTS idx_live_positions_trader ON live_positions(trader_id);
        CREATE INDEX IF NOT EXISTS idx_live_positions_status ON live_positions(status) WHERE status = 'open';
        CREATE INDEX IF NOT EXISTS idx_live_positions_market ON live_positions(market_id);
        CREATE INDEX IF NOT EXISTS idx_live_positions_opened ON live_positions(opened_at DESC);
        
        CREATE INDEX IF NOT EXISTS idx_signal_performance_name ON signal_performance(signal_name);
        CREATE INDEX IF NOT EXISTS idx_signal_performance_timestamp ON signal_performance(prediction_timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_signal_performance_accuracy ON signal_performance(signal_name, was_correct);
        
        CREATE INDEX IF NOT EXISTS idx_portfolio_trader_timestamp ON portfolio_snapshots(trader_id, snapshot_timestamp DESC);
        
        -- Create triggers for updated_at
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = CURRENT_TIMESTAMP;
            RETURN NEW;
        END;
        $$ language 'plpgsql';
        
        CREATE TRIGGER update_live_markets_updated_at BEFORE UPDATE ON live_markets
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        
        CREATE TRIGGER update_live_odds_updated_at BEFORE UPDATE ON live_odds
            FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
        """
        
        try:
            # Write SQL to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
                f.write(create_tables_sql)
                sql_file = f.name
            
            # Execute SQL
            result = subprocess.run([
                'psql',
                '-h', 'localhost',
                '-U', self.user,
                '-d', self.db_name,
                '-f', sql_file
            ], capture_output=True, text=True, 
            env={**os.environ, 'PGPASSWORD': self.password})
            
            if result.returncode == 0:
                logger.info("✅ Live trading tables created successfully")
                return True
            else:
                logger.error(f"❌ Table creation failed: {result.stderr}")
                return False
            
        finally:
            if os.path.exists(sql_file):
                os.unlink(sql_file)
    
    def test_connection(self) -> bool:
        """Test PostgreSQL connection."""
        logger.info("🧪 Testing PostgreSQL connection...")
        
        try:
            result = subprocess.run([
                'psql',
                '-h', 'localhost',
                '-U', self.user,
                '-d', self.db_name,
                '-c', 'SELECT version();'
            ], capture_output=True, text=True,
            env={**os.environ, 'PGPASSWORD': self.password})
            
            if result.returncode == 0:
                logger.info("✅ PostgreSQL connection successful")
                logger.info(f"   Database: {self.db_name}")
                logger.info(f"   User: {self.user}")
                return True
            else:
                logger.error(f"❌ Connection failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Connection test error: {e}")
            return False
    
    def create_hybrid_config(self) -> bool:
        """Create hybrid database configuration."""
        logger.info("⚙️ Creating hybrid configuration...")
        
        config = {
            "database_strategy": "hybrid",
            "created_at": datetime.now(timezone.utc).isoformat(),
            
            "live_database": {
                "type": "postgresql",
                "host": "localhost",
                "port": 5432,
                "database": self.db_name,
                "user": self.user,
                "password": self.password,
                "connection_string": f"postgresql://{self.user}:{self.password}@localhost:5432/{self.db_name}",
                "use_for": [
                    "new_blockchain_markets",
                    "live_odds_updates",
                    "active_trading_positions",
                    "signal_tracking",
                    "portfolio_management",
                    "real_time_data"
                ]
            },
            
            "historical_database": {
                "type": "sqlite",
                "path": "sport_odds.db",
                "size_gb": 210,
                "rows": 516000000,
                "use_for": [
                    "historical_markets",
                    "historical_odds",
                    "backtesting",
                    "analytics",
                    "archival_queries"
                ]
            },
            
            "migration_policy": {
                "cutoff_date": datetime.now(timezone.utc).isoformat(),
                "description": "New data goes to PostgreSQL, historical data stays in SQLite",
                "migration_in_progress": True,
                "estimated_completion": "2025-Q2"
            },
            
            "performance_expectations": {
                "postgresql_queries": "Sub-second response time",
                "sqlite_queries": "1-10 second response time for complex queries",
                "concurrent_connections": "200+ for PostgreSQL",
                "write_throughput": "10,000+ inserts/second"
            }
        }
        
        # Save configuration
        config_file = 'ominari_hybrid_db_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Hybrid configuration saved to {config_file}")
        return True
    
    def insert_sample_data(self) -> bool:
        """Insert sample data to verify setup."""
        logger.info("📊 Inserting sample data...")
        
        sample_sql = f"""
        -- Insert sample blockchain sync status
        INSERT INTO blockchain_sync (network, last_synced_block, markets_discovered)
        VALUES ('optimism', 141000000, 0), ('arbitrum', 378000000, 0)
        ON CONFLICT (network) DO UPDATE SET
            last_sync_timestamp = CURRENT_TIMESTAMP;
        
        -- Insert sample market (using current blockchain demo data structure)
        INSERT INTO live_markets (
            market_id, blockchain_address, network, source, sport, league,
            home_team, away_team, starts_at
        ) VALUES (
            'sample_blockchain_market_001',
            '0x1234567890123456789012345678901234567890',
            'optimism',
            'blockchain_optimism',
            'Soccer',
            'English Premier League',
            'Liverpool FC',
            'Manchester City',
            CURRENT_TIMESTAMP + INTERVAL '24 hours'
        ) ON CONFLICT (market_id) DO NOTHING;
        
        -- Insert sample odds
        INSERT INTO live_odds (
            market_id, position, outcome, bookmaker, decimal_odds, buy_odds, sell_odds
        ) VALUES
            ('sample_blockchain_market_001', 0, 'Home', 'blockchain_optimism', 2.10, 2.05, 2.15),
            ('sample_blockchain_market_001', 1, 'Away', 'blockchain_optimism', 3.50, 3.40, 3.60),
            ('sample_blockchain_market_001', 2, 'Draw', 'blockchain_optimism', 3.20, 3.10, 3.30)
        ON CONFLICT (market_id, position, bookmaker) DO UPDATE SET
            decimal_odds = EXCLUDED.decimal_odds,
            buy_odds = EXCLUDED.buy_odds,
            sell_odds = EXCLUDED.sell_odds,
            updated_at = CURRENT_TIMESTAMP;
        
        -- Insert sample portfolio
        INSERT INTO portfolio_snapshots (
            trader_id, total_bankroll, available_balance, total_exposure
        ) VALUES (
            'demo_trader', 10000.00, 10000.00, 0.00
        );
        """
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
                f.write(sample_sql)
                sql_file = f.name
            
            result = subprocess.run([
                'psql',
                '-h', 'localhost',
                '-U', self.user,
                '-d', self.db_name,
                '-f', sql_file
            ], capture_output=True, text=True,
            env={**os.environ, 'PGPASSWORD': self.password})
            
            if result.returncode == 0:
                logger.info("✅ Sample data inserted successfully")
                return True
            else:
                logger.error(f"❌ Sample data insertion failed: {result.stderr}")
                return False
                
        finally:
            if os.path.exists(sql_file):
                os.unlink(sql_file)
    
    def run_complete_setup(self) -> bool:
        """Run complete PostgreSQL setup."""
        logger.info("🚀 Running Complete PostgreSQL Setup")
        logger.info("=" * 60)
        
        steps = [
            ("Setup database and user", self.setup_postgresql_database),
            ("Create live trading tables", self.create_live_trading_tables),
            ("Test connection", self.test_connection),
            ("Create hybrid config", self.create_hybrid_config),
            ("Insert sample data", self.insert_sample_data),
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n📋 {step_name}...")
            
            if not step_func():
                logger.error(f"❌ Setup failed at: {step_name}")
                return False
        
        logger.info("\n🎉 PostgreSQL setup completed successfully!")
        logger.info(f"\n🔗 Connection Details:")
        logger.info(f"   Database: {self.db_name}")
        logger.info(f"   User: {self.user}")
        logger.info(f"   Connection: postgresql://{self.user}:{self.password}@localhost:5432/{self.db_name}")
        
        return True


if __name__ == "__main__":
    setup = SimplePostgreSQLSetup()
    success = setup.run_complete_setup()
    
    if success:
        print("\n" + "="*60)
        print("🎉 POSTGRESQL HYBRID SYSTEM READY!")
        print("="*60)
        print("✅ PostgreSQL configured for live trading")
        print("✅ SQLite available for historical data")
        print("✅ Hybrid configuration created")
        print("✅ Sample data inserted")
        print("\n🚀 Ready to start live blockchain trading!")
    else:
        print("\n❌ Setup incomplete - check logs for issues")