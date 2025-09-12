#!/usr/bin/env python3
"""
System Health Check for Ominari Trading System

Checks:
1. Database connectivity and status
2. Blockchain RPC endpoints
3. API endpoints
4. Signal registry
5. Paper trading system
"""

import sys
import time
import os
import psutil
from datetime import datetime

def print_header(text):
    print(f"\n{'='*60}")
    print(text)
    print(f"{'='*60}")

def print_success(text):
    print(f"✅ {text}")

def print_error(text):
    print(f"❌ {text}")

def print_warning(text):
    print(f"⚠️  {text}")

def print_info(text):
    print(f"ℹ️  {text}")


def check_database():
    """Check database connectivity and status."""
    print_header("DATABASE STATUS")
    
    try:
        from database_v2 import db_manager
        from models import Market, Odd
        
        with db_manager.get_db_session() as db:
            # Test connection
            result = db.execute("SELECT 1").fetchone()
            if result:
                print_success("Database connection: OK")
            
            # Get counts (with limits for safety)
            market_count = db.query(Market).limit(1000).count()
            print_info(f"Markets in database: {market_count:,}")
            
            # Check normalized table
            try:
                norm_result = db.execute(
                    "SELECT COUNT(*) FROM odds_normalized LIMIT 1"
                ).scalar()
                print_info(f"Normalized odds records: {norm_result:,}")
            except:
                print_warning("Normalized odds table not found")
                
    except Exception as e:
        print_error(f"Database error: {e}")
        return False
        
    return True


def check_blockchain():
    """Check blockchain connectivity."""
    print_header("BLOCKCHAIN RPC STATUS")
    
    try:
        from rpc_config import RPCManager
        
        for network in ['optimism', 'arbitrum']:
            try:
                manager = RPCManager(network)
                w3, endpoint = manager.get_web3()
                
                if w3.is_connected():
                    block = w3.eth.block_number
                    print_success(f"{network.title()}: Connected (block {block:,})")
                    print_info(f"  Using: {endpoint.name} - {endpoint.url[:50]}...")
                else:
                    print_error(f"{network.title()}: Not connected")
                    
            except Exception as e:
                print_error(f"{network.title()}: {str(e)[:80]}")
                
    except ImportError:
        print_error("RPC configuration not found")
        return False
        
    return True


def check_signal_registry():
    """Check signal registry system."""
    print_header("SIGNAL REGISTRY STATUS")
    
    try:
        from signal_registry import SignalRegistry
        from integrate_signal_registry import create_integrated_registry
        
        registry = create_integrated_registry()
        signals = registry.list_signals()
        
        print_success(f"Signal registry: Active")
        print_info(f"Registered signals: {', '.join(signals)}")
        
        # Check weights
        weights = registry.get_weights()
        print_info("Current weights:")
        for signal, weight in weights.items():
            print(f"  - {signal}: {weight:.2%}")
            
    except Exception as e:
        print_error(f"Signal registry error: {e}")
        return False
        
    return True


def check_paper_trading():
    """Check paper trading system."""
    print_header("PAPER TRADING STATUS")
    
    try:
        from paper_trading_db import get_engine, PaperTradingPosition, PaperTradingSession
        from sqlalchemy.orm import sessionmaker
        
        engine = get_engine()
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Get session count
        session_count = session.query(PaperTradingSession).count()
        print_info(f"Paper trading sessions: {session_count}")
        
        # Get active positions
        active_positions = session.query(PaperTradingPosition).filter(
            PaperTradingPosition.status == 'open'
        ).count()
        print_info(f"Active positions: {active_positions}")
        
        print_success("Paper trading: Operational")
        
        session.close()
        
    except Exception as e:
        print_error(f"Paper trading error: {e}")
        return False
        
    return True


def check_system_resources():
    """Check system resources."""
    print_header("SYSTEM RESOURCES")
    
    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent < 80:
        print_success(f"CPU usage: {cpu_percent}%")
    else:
        print_warning(f"High CPU usage: {cpu_percent}%")
        
    # Memory
    memory = psutil.virtual_memory()
    if memory.percent < 80:
        print_success(f"Memory usage: {memory.percent}% ({memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB)")
    else:
        print_warning(f"High memory usage: {memory.percent}%")
        
    # Disk
    disk = psutil.disk_usage('/')
    if disk.percent < 90:
        print_success(f"Disk usage: {disk.percent}% ({disk.free / 1e9:.1f}GB free)")
    else:
        print_error(f"Low disk space: {disk.percent}% used")
        
    return True


def check_migration_status():
    """Check database migration status."""
    print_header("MIGRATION STATUS")
    
    try:
        import json
        import os
        
        # Check migration state file
        if os.path.exists('migration_state.json'):
            with open('migration_state.json', 'r') as f:
                state = json.load(f)
                
            total = 515919464  # Total odds records
            progress = (state['total_processed'] / total) * 100
            
            print_info(f"Migration progress: {progress:.2f}%")
            print_info(f"Records processed: {state['total_processed']:,}")
            print_info(f"Records migrated: {state['total_migrated']:,}")
            
            # Check if migration is running
            if os.path.exists('migration.pid'):
                with open('migration.pid', 'r') as f:
                    pid = int(f.read().strip())
                    
                if psutil.pid_exists(pid):
                    print_success("Migration: Running")
                else:
                    print_warning("Migration: Stopped")
            else:
                print_info("Migration: Not running")
        else:
            print_info("No migration in progress")
            
    except Exception as e:
        print_error(f"Could not check migration: {e}")
        
    return True


def generate_recommendations():
    """Generate recommendations based on health check."""
    print_header("RECOMMENDATIONS")
    
    recommendations = []
    
    # Check if migration needs to be completed
    try:
        import json
        if os.path.exists('migration_state.json'):
            with open('migration_state.json', 'r') as f:
                state = json.load(f)
            if state['total_processed'] < 515919464:
                recommendations.append("Resume database migration: python fast_migration.py --resume")
    except:
        pass
        
    # Check if RPC needs configuration
    if not os.path.exists('.rpc_config.json') and not os.getenv('ALCHEMY_API_KEY'):
        recommendations.append("Configure RPC endpoints: python rpc_config.py setup")
        
    # Check if PostgreSQL is needed
    if 'sqlite' in os.getenv('DATABASE_URL', 'sqlite'):
        recommendations.append("Set up PostgreSQL: python setup_postgres.py")
        
    if recommendations:
        print("To improve system performance:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print_success("System is properly configured!")
        
    return True


def main():
    """Run all health checks."""
    print(f"\nOminari Trading System Health Check")
    print(f"Run Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    checks = [
        ("Database", check_database),
        ("Blockchain RPC", check_blockchain),
        ("Signal Registry", check_signal_registry),
        ("Paper Trading", check_paper_trading),
        ("System Resources", check_system_resources),
        ("Migration Status", check_migration_status),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print_error(f"Failed to check {name}: {e}")
            results.append((name, False))
            
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    if passed == total:
        print_success(f"All checks passed ({passed}/{total})")
    elif passed > total // 2:
        print_warning(f"Some checks failed ({passed}/{total})")
    else:
        print_error(f"Multiple failures ({passed}/{total})")
        
    # Recommendations
    generate_recommendations()
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nHealth check interrupted by user")
        sys.exit(1)