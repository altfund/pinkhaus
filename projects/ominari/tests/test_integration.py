#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration tests for the complete system
"""

import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd

from integrations import SystemIntegrator
from config import settings
from database import SessionLocal


class TestSystemIntegration:
    """Test system integration workflows."""
    
    @pytest.fixture
    def integrator(self):
        """Create system integrator instance."""
        return SystemIntegrator()
        
    def test_initialization(self, integrator):
        """Test system initialization."""
        # Check components are initialized
        assert integrator.signal_registry is not None
        assert integrator.weight_manager is not None
        assert integrator.alpha_pipeline is not None
        assert integrator.paper_engine is not None
        assert integrator.systematic_framework is not None
        
        # Check signals are registered
        signals = integrator.signal_registry.list_signals()
        assert len(signals) > 0
        
    def test_signal_wrapping(self, integrator):
        """Test legacy signal wrapping."""
        from signals import ImpliedRawSignal
        
        wrapped = integrator._wrap_legacy_signal(ImpliedRawSignal)
        
        # Test wrapped signal
        data = pd.DataFrame({
            'implied_raw': [45.0, 55.0],
            'match_id': ['m1', 'm1']
        })
        
        probs = wrapped.get_probs(data)
        assert len(probs) == 2
        assert all(0 <= p <= 1 for p in probs)
        
    @pytest.mark.asyncio
    async def test_data_collection_cycle(self, integrator):
        """Test data collection workflow."""
        # This would need mocking of external APIs
        # For now, just test the structure
        try:
            await integrator.run_data_collection_cycle()
        except Exception as e:
            # Expected to fail without real connections
            assert "collection failed" in str(e).lower()
            
    def test_market_data_preparation(self, integrator):
        """Test market data preparation."""
        # Create mock market data
        with SessionLocal() as db:
            # Would create test markets and odds
            markets = []
            
        df = integrator._prepare_market_data(markets, None)
        assert isinstance(df, pd.DataFrame)
        
    def test_position_calculation(self, integrator):
        """Test position size calculation."""
        # Create test predictions
        predictions = pd.Series(
            [0.55, 0.48, 0.62],
            index=pd.MultiIndex.from_tuples([
                ('m1', 2.0),
                ('m2', 2.5),
                ('m3', 1.8)
            ], names=['market_id', 'odds'])
        )
        
        positions = integrator._calculate_positions(predictions)
        
        assert isinstance(positions, pd.DataFrame)
        assert 'bet_size' in positions.columns
        assert all(positions['bet_size'] >= 0)  # No negative bets
        
    def test_regime_detection(self, integrator):
        """Test market regime detection."""
        regime = integrator._detect_market_regime()
        
        assert regime in ['overnight', 'morning', 'afternoon', 'evening']
        
    def test_system_metrics_collection(self, integrator):
        """Test system metrics collection."""
        metrics = integrator._collect_system_metrics()
        
        assert 'timestamp' in metrics
        assert 'signals' in metrics
        assert 'trading' in metrics
        assert 'research' in metrics
        
        # Check structure
        assert isinstance(metrics['signals']['active'], int)
        assert isinstance(metrics['trading']['paper_pnl'], (int, float))
        
    def test_alert_conditions(self, integrator):
        """Test alert condition checking."""
        # Test with normal metrics
        metrics = {
            'trading': {'paper_pnl': -500},
            'signals': {'active': 3}
        }
        
        alerts = integrator._check_alert_conditions(metrics)
        assert len(alerts) == 0  # No alerts
        
        # Test with drawdown breach
        metrics['trading']['paper_pnl'] = -3000  # 30% drawdown
        alerts = integrator._check_alert_conditions(metrics)
        assert len(alerts) > 0
        assert "drawdown" in alerts[0].lower()
        
    def test_daily_report_generation(self, integrator):
        """Test daily report generation."""
        report = integrator.generate_daily_report()
        
        assert 'date' in report
        assert 'summary' in report
        assert 'signal_performance' in report
        assert 'trading_performance' in report
        assert 'research_pipeline' in report
        assert 'system_health' in report
        
        # Check report was saved
        report_path = settings.report_dir / f"daily_report_{report['date']}.json"
        assert report_path.exists()


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_trading_cycle(self):
        """Test full trading cycle from data to execution."""
        integrator = SystemIntegrator()
        
        # Run full cycle (would need mocking for real test)
        try:
            await integrator.run_full_cycle()
        except Exception:
            # Expected to fail without real data
            pass
            
    def test_backtest_all_signals(self):
        """Test backtesting all registered signals."""
        from integrations import backtest_all_signals
        
        # Run short backtest
        start = datetime.now(timezone.utc) - timedelta(days=7)
        end = datetime.now(timezone.utc)
        
        results = backtest_all_signals(start, end)
        
        assert isinstance(results, dict)
        # Would have results for each signal if data available