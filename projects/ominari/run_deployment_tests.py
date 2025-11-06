#!/usr/bin/env python3
"""
Quick deployment test runner
Runs essential tests for deployment validation
"""

import os
import sys
import logging
from datetime import datetime

# Set environment
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production',
    'USE_POSTGRESQL': '1'
})

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_quick_tests():
    """Run essential tests quickly"""
    logger.info("🚀 Running Quick Deployment Tests")
    logger.info("=" * 50)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Database Connection
    tests_total += 1
    logger.info("1. Testing database connection...")
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
            database=os.environ['PG_DB']
        )
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM market WHERE is_finished = false;")
        market_count = cur.fetchone()[0]
        conn.close()
        
        if market_count > 0:
            logger.info(f"   ✅ Database connected ({market_count} active markets)")
            tests_passed += 1
        else:
            logger.error("   ❌ No active markets found")
            
    except Exception as e:
        logger.error(f"   ❌ Database connection failed: {e}")
    
    # Test 2: Dashboard Response
    tests_total += 1
    logger.info("2. Testing dashboard response...")
    try:
        import requests
        response = requests.get("http://localhost:8888", timeout=5)
        if response.status_code == 200 and '🛑 STOP' in response.text:
            logger.info("   ✅ Dashboard responds with STOP button")
            tests_passed += 1
        else:
            logger.error(f"   ❌ Dashboard response issue (status: {response.status_code})")
    except Exception as e:
        logger.error(f"   ❌ Dashboard not accessible: {e}")
    
    # Test 3: Paper Trading System
    tests_total += 1
    logger.info("3. Testing paper trading system...")
    try:
        from paper_trading_postgres_integrated import PaperTradingSessionManager
        session_manager = PaperTradingSessionManager()
        session_id = session_manager.get_current_session()
        if session_id:
            session = session_manager.get_session(session_id)
            if session and session.get('current_bankroll'):
                logger.info(f"   ✅ Paper trading active (session: {session_id}, bankroll: ${session.get('current_bankroll'):.2f})")
                tests_passed += 1
            else:
                logger.error("   ❌ Session data incomplete")
        else:
            logger.error("   ❌ No active trading session")
    except Exception as e:
        logger.error(f"   ❌ Paper trading test failed: {e}")
    
    # Test 4: Stop Loss System
    tests_total += 1
    logger.info("4. Testing stop loss system...")
    try:
        from stop_loss_manager import StopLossManager
        from paper_trading_postgres_integrated import PaperTradingSessionManager
        
        session_manager = PaperTradingSessionManager()
        stop_loss = StopLossManager(session_manager)
        status = stop_loss.get_stop_status()
        
        if 'is_stopped' in status:
            logger.info(f"   ✅ Stop loss system functional (stopped: {status['is_stopped']})")
            tests_passed += 1
        else:
            logger.error("   ❌ Stop loss system not responding")
    except Exception as e:
        logger.error(f"   ❌ Stop loss test failed: {e}")
    
    # Results
    logger.info("\n" + "=" * 50)
    logger.info(f"📊 Results: {tests_passed}/{tests_total} tests passed")
    
    if tests_passed == tests_total:
        logger.info("✅ ALL TESTS PASSED - System ready for deployment")
        return True
    else:
        logger.error(f"❌ {tests_total - tests_passed} TESTS FAILED - Fix issues before deployment")
        return False

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)