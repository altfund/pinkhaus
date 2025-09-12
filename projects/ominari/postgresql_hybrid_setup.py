#!/usr/bin/env python3
"""
PostgreSQL Hybrid Setup

Sets up PostgreSQL as the primary database for new data while keeping
SQLite for historical data. This hybrid approach provides immediate
performance benefits without waiting for full migration.
"""

import logging
import os
import subprocess
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional
import json
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PostgreSQLHybridSetup:
    """Sets up PostgreSQL for hybrid architecture."""
    
    def __init__(self, 
                 pg_data_dir: str = "/tmp/ominari_postgres_data",
                 pg_port: int = 5433,
                 pg_user: str = "ominari",
                 pg_database: str = "ominari_live"):
        
        self.pg_data_dir = pg_data_dir
        self.pg_port = pg_port
        self.pg_user = pg_user
        self.pg_database = pg_database
        self.pg_password = "ominari_secure_2025"
        
        self.setup_status = {
            'postgresql_installed': False,
            'database_initialized': False,
            'server_running': False,
            'database_created': False,
            'tables_created': False,
            'hybrid_config_ready': False
        }
    
    def check_postgresql_installation(self) -> bool:
        """Check if PostgreSQL is installed."""
        try:
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ PostgreSQL found: {result.stdout.strip()}")
                self.setup_status['postgresql_installed'] = True
                return True
            else:
                logger.warning("⚠️ PostgreSQL not found")
                return False
        except FileNotFoundError:
            logger.warning("⚠️ PostgreSQL not installed")
            return False
    
    def install_postgresql_ubuntu(self) -> bool:
        """Install PostgreSQL on Ubuntu (if needed)."""
        logger.info("📦 Installing PostgreSQL...")
        
        try:
            # Update package list
            subprocess.run(['sudo', 'apt', 'update'], check=True)
            
            # Install PostgreSQL
            subprocess.run([
                'sudo', 'apt', 'install', '-y', 
                'postgresql', 'postgresql-contrib', 'libpq-dev'
            ], check=True)
            
            logger.info("✅ PostgreSQL installation completed")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ PostgreSQL installation failed: {e}")
            return False
    
    def initialize_database_cluster(self) -> bool:
        """Initialize PostgreSQL database cluster."""
        logger.info(f"🔧 Initializing PostgreSQL cluster at {self.pg_data_dir}")
        
        try:
            # Create data directory
            os.makedirs(self.pg_data_dir, exist_ok=True)
            
            # Initialize database cluster
            subprocess.run([
                'initdb', 
                '-D', self.pg_data_dir,
                '-U', self.pg_user,
                '--auth-local=trust',
                '--auth-host=md5'
            ], check=True)
            
            # Create postgresql.conf settings
            self._configure_postgresql()
            
            logger.info("✅ Database cluster initialized")
            self.setup_status['database_initialized'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database initialization failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Database initialization error: {e}")
            return False
    
    def _configure_postgresql(self):
        """Configure PostgreSQL for performance."""
        config_file = os.path.join(self.pg_data_dir, 'postgresql.conf')
        
        # Performance settings optimized for trading data
        config_additions = f"""
# Ominari Trading System Configuration
port = {self.pg_port}
max_connections = 200
shared_buffers = 512MB
effective_cache_size = 2GB
work_mem = 16MB
maintenance_work_mem = 256MB

# WAL settings for high throughput
wal_level = replica
max_wal_size = 2GB
min_wal_size = 512MB
checkpoint_completion_target = 0.7

# Logging for monitoring
log_statement = 'mod'
log_duration = on
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '

# Enable extensions
shared_preload_libraries = 'pg_stat_statements'
"""
        
        # Append configuration
        with open(config_file, 'a') as f:
            f.write(config_additions)
        
        logger.info("✅ PostgreSQL configuration updated")
    
    def start_postgresql_server(self) -> bool:
        """Start PostgreSQL server."""
        logger.info("🚀 Starting PostgreSQL server...")
        
        try:
            # Start PostgreSQL
            result = subprocess.run([
                'pg_ctl', 
                '-D', self.pg_data_dir,
                '-l', os.path.join(self.pg_data_dir, 'postgresql.log'),
                'start'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                # Wait for server to be ready
                time.sleep(3)
                
                # Check if server is running
                if self._check_server_status():
                    logger.info("✅ PostgreSQL server started successfully")
                    self.setup_status['server_running'] = True
                    return True
                else:
                    logger.error("❌ PostgreSQL server failed to start properly")
                    return False
            else:
                logger.error(f"❌ Failed to start PostgreSQL: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting PostgreSQL: {e}")
            return False
    
    def _check_server_status(self) -> bool:
        """Check if PostgreSQL server is running."""
        try:
            result = subprocess.run([
                'pg_isready', 
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user
            ], capture_output=True)
            
            return result.returncode == 0
        except:
            return False
    
    def create_database_and_user(self) -> bool:
        """Create database and user for Ominari."""
        logger.info(f"👤 Creating database '{self.pg_database}' and user '{self.pg_user}'")
        
        try:
            # Connect as superuser and create database
            subprocess.run([
                'createdb',
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user,
                self.pg_database
            ], check=True)
            
            # Set password for user (using SQL)
            sql_commands = f"""
            ALTER USER {self.pg_user} PASSWORD '{self.pg_password}';
            GRANT ALL PRIVILEGES ON DATABASE {self.pg_database} TO {self.pg_user};
            """
            
            subprocess.run([
                'psql',
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user,
                '-d', 'postgres',
                '-c', sql_commands
            ], check=True)
            
            logger.info("✅ Database and user created successfully")
            self.setup_status['database_created'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Database creation failed: {e}")
            return False
    
    def create_trading_tables(self) -> bool:
        """Create optimized trading tables in PostgreSQL."""
        logger.info("🏗️ Creating trading tables in PostgreSQL...")
        
        # SQL to create trading-optimized tables
        create_tables_sql = """
        -- Markets table (optimized for trading)
        CREATE TABLE IF NOT EXISTS live_markets (
            market_id VARCHAR(100) PRIMARY KEY,
            blockchain_address VARCHAR(42),
            network VARCHAR(20),
            source VARCHAR(50),
            sport VARCHAR(50),
            league VARCHAR(100),
            home_team VARCHAR(200),
            away_team VARCHAR(200),
            market_type VARCHAR(50),
            starts_at TIMESTAMP WITH TIME ZONE,
            is_finished BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            metadata JSONB
        );

        -- Live odds table (optimized for high-frequency updates)
        CREATE TABLE IF NOT EXISTS live_odds (
            id SERIAL PRIMARY KEY,
            market_id VARCHAR(100) REFERENCES live_markets(market_id),
            position INTEGER,
            outcome VARCHAR(50),
            bookmaker VARCHAR(100),
            decimal_odds DECIMAL(10,4),
            buy_odds DECIMAL(10,4),
            sell_odds DECIMAL(10,4),
            liquidity DECIMAL(15,2),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Trading positions table
        CREATE TABLE IF NOT EXISTS trading_positions (
            position_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            market_id VARCHAR(100) REFERENCES live_markets(market_id),
            trader_id VARCHAR(100),
            position_type VARCHAR(20), -- 'long', 'short'
            outcome VARCHAR(50),
            stake DECIMAL(15,2),
            odds DECIMAL(10,4),
            expected_payout DECIMAL(15,2),
            signal_probability DECIMAL(5,4),
            signal_providers JSONB,
            kelly_fraction DECIMAL(5,4),
            opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP WITH TIME ZONE,
            pnl DECIMAL(15,2) DEFAULT 0,
            status VARCHAR(20) DEFAULT 'open' -- 'open', 'closed', 'settled'
        );

        -- Signal performance tracking
        CREATE TABLE IF NOT EXISTS signal_tracking (
            id SERIAL PRIMARY KEY,
            signal_name VARCHAR(100),
            market_id VARCHAR(100),
            predicted_probability DECIMAL(5,4),
            actual_outcome VARCHAR(50),
            accuracy DECIMAL(5,4),
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );

        -- Blockchain sync status
        CREATE TABLE IF NOT EXISTS blockchain_sync_status (
            network VARCHAR(20) PRIMARY KEY,
            last_synced_block BIGINT,
            last_sync_timestamp TIMESTAMP WITH TIME ZONE,
            markets_synced INTEGER DEFAULT 0,
            status VARCHAR(20) DEFAULT 'active'
        );

        -- Create indexes for performance
        CREATE INDEX IF NOT EXISTS idx_live_markets_sport_league ON live_markets(sport, league);
        CREATE INDEX IF NOT EXISTS idx_live_markets_starts_at ON live_markets(starts_at);
        CREATE INDEX IF NOT EXISTS idx_live_markets_blockchain ON live_markets(blockchain_address);
        CREATE INDEX IF NOT EXISTS idx_live_markets_network ON live_markets(network);

        CREATE INDEX IF NOT EXISTS idx_live_odds_market_updated ON live_odds(market_id, updated_at);
        CREATE INDEX IF NOT EXISTS idx_live_odds_bookmaker ON live_odds(bookmaker);
        CREATE INDEX IF NOT EXISTS idx_live_odds_position ON live_odds(position, outcome);

        CREATE INDEX IF NOT EXISTS idx_trading_positions_trader ON trading_positions(trader_id);
        CREATE INDEX IF NOT EXISTS idx_trading_positions_market ON trading_positions(market_id);
        CREATE INDEX IF NOT EXISTS idx_trading_positions_status ON trading_positions(status);
        CREATE INDEX IF NOT EXISTS idx_trading_positions_opened ON trading_positions(opened_at);

        CREATE INDEX IF NOT EXISTS idx_signal_tracking_signal ON signal_tracking(signal_name);
        CREATE INDEX IF NOT EXISTS idx_signal_tracking_timestamp ON signal_tracking(timestamp);
        """
        
        try:
            # Write SQL to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sql', delete=False) as f:
                f.write(create_tables_sql)
                sql_file = f.name
            
            # Execute SQL
            subprocess.run([
                'psql',
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user,
                '-d', self.pg_database,
                '-f', sql_file
            ], check=True)
            
            # Clean up temp file
            os.unlink(sql_file)
            
            logger.info("✅ Trading tables created successfully")
            self.setup_status['tables_created'] = True
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Table creation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Table creation error: {e}")
            return False
    
    def create_hybrid_configuration(self) -> bool:
        """Create configuration for hybrid SQLite/PostgreSQL setup."""
        logger.info("⚙️ Creating hybrid database configuration...")
        
        config = {
            'database_strategy': 'hybrid',
            'primary_db': {
                'type': 'postgresql',
                'host': 'localhost',
                'port': self.pg_port,
                'database': self.pg_database,
                'user': self.pg_user,
                'password': self.pg_password,
                'use_for': [
                    'new_markets',
                    'live_odds', 
                    'trading_positions',
                    'signal_tracking',
                    'blockchain_sync'
                ]
            },
            'historical_db': {
                'type': 'sqlite',
                'path': 'sport_odds.db',
                'use_for': [
                    'historical_markets',
                    'historical_odds',
                    'backtesting_data',
                    'archived_data'
                ]
            },
            'migration_strategy': {
                'cutoff_date': datetime.now(timezone.utc).isoformat(),
                'description': 'Data before this date stays in SQLite, new data goes to PostgreSQL'
            }
        }
        
        # Save configuration
        config_file = 'hybrid_db_config.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✅ Hybrid configuration saved to {config_file}")
        self.setup_status['hybrid_config_ready'] = True
        return True
    
    def test_hybrid_setup(self) -> bool:
        """Test the hybrid database setup."""
        logger.info("🧪 Testing hybrid database setup...")
        
        try:
            # Test PostgreSQL connection
            result = subprocess.run([
                'psql',
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user,
                '-d', self.pg_database,
                '-c', 'SELECT version();'
            ], capture_output=True, text=True, check=True)
            
            logger.info("✅ PostgreSQL connection test passed")
            
            # Test table creation
            subprocess.run([
                'psql',
                '-h', 'localhost',
                '-p', str(self.pg_port),
                '-U', self.pg_user,
                '-d', self.pg_database,
                '-c', "INSERT INTO live_markets (market_id, source, sport) VALUES ('test_market', 'test', 'Test') ON CONFLICT DO NOTHING;"
            ], check=True)
            
            logger.info("✅ PostgreSQL write test passed")
            
            # Test SQLite still works
            import sqlite3
            conn = sqlite3.connect('sport_odds.db')
            result = conn.execute("SELECT COUNT(*) FROM market LIMIT 1").fetchone()
            conn.close()
            
            logger.info("✅ SQLite connection test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Hybrid setup test failed: {e}")
            return False
    
    def run_complete_setup(self) -> Dict:
        """Run complete PostgreSQL hybrid setup."""
        logger.info("🚀 Starting Complete PostgreSQL Hybrid Setup")
        logger.info("=" * 70)
        
        setup_steps = [
            ("Check PostgreSQL installation", self.check_postgresql_installation),
            ("Initialize database cluster", self.initialize_database_cluster),
            ("Start PostgreSQL server", self.start_postgresql_server),
            ("Create database and user", self.create_database_and_user),
            ("Create trading tables", self.create_trading_tables),
            ("Create hybrid configuration", self.create_hybrid_configuration),
            ("Test hybrid setup", self.test_hybrid_setup)
        ]
        
        results = {}
        
        for step_name, step_function in setup_steps:
            logger.info(f"\n📋 {step_name}...")
            
            try:
                success = step_function()
                results[step_name] = {
                    'success': success,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                if success:
                    logger.info(f"✅ {step_name} completed")
                else:
                    logger.error(f"❌ {step_name} failed")
                    break  # Stop on first failure
                    
            except Exception as e:
                logger.error(f"❌ {step_name} error: {e}")
                results[step_name] = {
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                break
        
        # Summary
        successful_steps = sum(1 for r in results.values() if r.get('success', False))
        total_steps = len(setup_steps)
        
        logger.info(f"\n🎯 Setup Summary:")
        logger.info(f"   Completed steps: {successful_steps}/{total_steps}")
        logger.info(f"   Setup status: {self.setup_status}")
        
        if successful_steps == total_steps:
            logger.info("🎉 Hybrid PostgreSQL setup completed successfully!")
            
            # Provide connection details
            logger.info(f"\n🔗 Connection Details:")
            logger.info(f"   PostgreSQL: postgresql://{self.pg_user}:{self.pg_password}@localhost:{self.pg_port}/{self.pg_database}")
            logger.info(f"   SQLite: sport_odds.db (210 GB historical data)")
            logger.info(f"   Configuration: hybrid_db_config.json")
            
        else:
            logger.error("❌ Hybrid setup incomplete - check logs for issues")
        
        return {
            'success': successful_steps == total_steps,
            'completed_steps': successful_steps,
            'total_steps': total_steps,
            'results': results,
            'setup_status': self.setup_status
        }


def main():
    """Run PostgreSQL hybrid setup."""
    setup = PostgreSQLHybridSetup()
    results = setup.run_complete_setup()
    
    print(f"\n{'='*70}")
    print("POSTGRESQL HYBRID SETUP RESULTS")
    print(f"{'='*70}")
    
    if results['success']:
        print("🎉 SUCCESS: Hybrid database setup completed!")
        print("\n🚀 Next Steps:")
        print("   1. Update application to use hybrid configuration")
        print("   2. Start directing new data to PostgreSQL")
        print("   3. Keep SQLite for historical queries")
        print("   4. Monitor performance improvements")
    else:
        print("❌ SETUP INCOMPLETE")
        print("Check logs for specific issues")


if __name__ == "__main__":
    main()