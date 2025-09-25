#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pytest configuration and fixtures
"""

import pytest
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# Set test environment
os.environ['ENVIRONMENT'] = 'testing'
os.environ['DATABASE_URL'] = 'sqlite:///test_sport_odds.db'
os.environ['TESTING'] = 'true'


@pytest.fixture(scope='session')
def test_data_dir(tmp_path_factory):
    """Create temporary data directory for tests."""
    return tmp_path_factory.mktemp('test_data')


@pytest.fixture
def paper_trading_engine(tmp_path):
    """Create paper trading engine with unique database."""
    from paper_trading_engine import PaperTradingEngine
    db_path = str(tmp_path / f"paper_trades_{datetime.now().timestamp()}.db")
    engine = PaperTradingEngine(db_path=db_path)
    return engine


@pytest.fixture(scope='session')
def test_config():
    """Get test configuration."""
    from config import settings, reload_settings
    
    # Force reload with test environment
    settings = reload_settings()
    settings.testing = True
    
    return settings


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range('2024-01-01', '2024-01-07', freq='H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'market_id': 'TEST_001',
        'source_id': '0x12345',
        'sport': 'NFL',
        'home_team': 'TeamA',
        'away_team': 'TeamB',
        'bet_name': ['TeamA'] * len(dates),
        'odds': 2.0 + 0.1 * np.random.randn(len(dates)),
        'volume': 1000 + 500 * np.random.random(len(dates)),
        'implied_raw': 50 + 5 * np.random.randn(len(dates))
    })
    
    return data


@pytest.fixture
def sample_predictions():
    """Generate sample predictions for testing."""
    n = 10
    predictions = pd.Series(
        0.5 + 0.1 * np.random.randn(n),
        index=pd.MultiIndex.from_arrays([
            [f'market_{i}' for i in range(n)],
            2.0 + 0.5 * np.random.random(n)
        ], names=['market_id', 'odds'])
    )
    
    return predictions.clip(0, 1)


@pytest.fixture
def mock_database(monkeypatch):
    """Mock database for testing."""
    
    class MockSession:
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
            
        def query(self, *args):
            return self
            
        def filter(self, *args):
            return self
            
        def limit(self, *args):
            return self
            
        def all(self):
            return []
            
        def execute(self, query):
            return self
            
        def scalar(self):
            return 1
            
        def commit(self):
            pass
            
        def rollback(self):
            pass
            
    monkeypatch.setattr('database.SessionLocal', MockSession)
    
    return MockSession()


@pytest.fixture
async def mock_quote():
    """Mock quote for paper trading tests."""
    from paper_trading_engine import Quote
    
    return Quote(
        source_id='TEST_001',
        timestamp=datetime.now(timezone.utc),
        bid_price=1.95,
        bid_size=1000,
        ask_price=2.05,
        ask_size=1000,
        mid_price=2.00,
        spread=0.10,
        liquidity_score=0.8
    )


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )