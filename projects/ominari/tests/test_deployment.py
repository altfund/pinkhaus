#!/usr/bin/env python3
"""
Deployment validation tests
Tests critical functionality before deployment
"""

import pytest
import os
import requests
import psycopg2
from datetime import datetime, timezone
import pandas as pd

# Set test environment
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production',
    'USE_POSTGRESQL': '1'
})

@pytest.mark.deployment
@pytest.mark.database
def test_database_connectivity():
    """Test PostgreSQL database connectivity"""
    pg_config = {
        'host': os.environ['PG_HOST'],
        'port': os.environ['PG_PORT'],
        'user': os.environ['PG_USER'],
        'password': os.environ['PG_PASSWORD'],
        'database': os.environ['PG_DB']
    }
    
    conn = psycopg2.connect(**pg_config)
    cur = conn.cursor()
    
    # Test basic query
    cur.execute("SELECT version();")
    version = cur.fetchone()[0]
    assert "PostgreSQL" in version
    
    # Test critical tables
    critical_tables = ['market', 'odd', 'paper_trading_sessions']
    for table in critical_tables:
        cur.execute(f"SELECT COUNT(*) FROM {table};")
        count = cur.fetchone()[0]
        assert count >= 0, f"Table {table} not accessible"
    
    conn.close()

@pytest.mark.deployment  
@pytest.mark.dashboard
def test_dashboard_accessibility():
    """Test dashboard web interface"""
    dashboard_url = "http://localhost:8888"
    
    response = requests.get(dashboard_url, timeout=10)
    assert response.status_code == 200
    
    html = response.text
    assert "Ominari Trading System" in html
    assert "🛑 STOP" in html, "Stop button not found"
    assert "socket.io" in html, "WebSocket script missing"

@pytest.mark.deployment
@pytest.mark.trading
def test_paper_trading_system():
    """Test paper trading functionality"""
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    
    session_manager = PaperTradingSessionManager()
    
    # Test session retrieval
    session_id = session_manager.get_current_session()
    assert session_id is not None, "No active session found"
    
    session_data = session_manager.get_session(session_id)
    assert session_data is not None
    assert 'current_bankroll' in session_data
    assert session_data['current_bankroll'] > 0

@pytest.mark.deployment
def test_stop_loss_system():
    """Test stop loss functionality"""
    from stop_loss_manager import StopLossManager
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    
    session_manager = PaperTradingSessionManager()
    stop_loss = StopLossManager(session_manager)
    
    # Test configuration
    test_config = {'drawdown_pct': 10}
    stop_loss.set_stop_loss_config(test_config)
    
    # Test status methods
    status = stop_loss.get_stop_status()
    assert 'is_stopped' in status
    assert isinstance(status['is_stopped'], bool)
    
    can_resume, _ = stop_loss.can_resume_trading()
    assert isinstance(can_resume, bool)

@pytest.mark.deployment
def test_portfolio_engine():
    """Test portfolio trading engine"""
    from portfolio_trading_engine import PortfolioTradingEngine
    from paper_trading_postgres_integrated import PaperTradingSessionManager
    from edge_calculator import EdgeCalculator
    
    session_manager = PaperTradingSessionManager()
    edge_calculator = EdgeCalculator()
    
    strategy_config = {
        'bankroll': 1000,
        'kelly_fraction': 0.25,
        'cap_per_game': 0.02,
        'cap_per_bet': 0.01,
        'min_bet': 10
    }
    
    portfolio_engine = PortfolioTradingEngine(
        session_manager, edge_calculator, strategy_config
    )
    
    # Test with minimal data
    markets = [{'market_id': 'TEST', 'home_team': 'A', 'away_team': 'B', 'sport': 'Soccer'}]
    signals = [{'home_odds': 2.0, 'draw_odds': 3.0, 'away_odds': 2.5, 
               'home_edge': 0, 'draw_edge': 0, 'away_edge': 0}]
    
    markets_df = portfolio_engine.prepare_markets_for_kelly(markets, signals)
    assert not markets_df.empty
    assert 'normalized_outcome' in markets_df.columns

@pytest.mark.deployment
def test_data_quality():
    """Test basic data quality"""
    from database_v2 import db_manager
    from models import Market, Odd
    
    with db_manager.get_db_session() as db:
        # Check recent markets exist
        active_markets = db.query(Market).filter(
            Market.is_finished == False
        ).limit(10).all()
        
        assert len(active_markets) > 0, "No active markets found"
        
        # Check odds exist for markets  
        for market in active_markets[:3]:
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).limit(5).all()
            
            assert len(odds) >= 0, f"Could not query odds for market {market.source_id}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])