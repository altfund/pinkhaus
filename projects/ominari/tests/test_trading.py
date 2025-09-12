#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for trading components
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone

from paper_trading_engine import (
    PaperTradingEngine, PaperOrder, Quote, 
    MarketImpactModel
)
from carver_framework import (
    SystematicFramework, TradingRule, ForecastCombiner,
    VolatilityCalculator, PositionSizer
)


class TestPaperTrading:
    """Test paper trading engine."""
    
    @pytest.mark.asyncio
    async def test_paper_order_submission(self):
        """Test submitting a paper order."""
        engine = PaperTradingEngine(initial_capital=10000)
        
        order = PaperOrder(
            order_id="TEST_001",
            timestamp=datetime.now(timezone.utc),
            source_id="0x12345",
            market_type="winner",
            bet_name="TeamA",
            side="buy",
            size=100,
            limit_price=None,
            signal_name="test_signal",
            expected_edge=0.05
        )
        
        # Submit order
        fill = await engine.submit_order(order)
        
        assert fill is not None
        assert fill.order_id == order.order_id
        assert fill.fill_size == order.size
        assert fill.commission > 0
        
    def test_market_impact_model(self):
        """Test market impact calculation."""
        model = MarketImpactModel()
        
        # Test impact calculation
        impact = model.estimate_impact(
            size=1000,
            liquidity=10000,
            volatility=0.20
        )
        
        assert 0 < impact < 0.05  # Capped at 5%
        
        # Test slippage
        quote = Quote(
            source_id="test",
            timestamp=datetime.now(timezone.utc),
            bid_price=1.95,
            bid_size=500,
            ask_price=2.05,
            ask_size=500,
            mid_price=2.00,
            spread=0.10,
            liquidity_score=0.8
        )
        
        exec_price, slippage = model.calculate_slippage(
            quote, size=100, side="buy"
        )
        
        assert exec_price >= quote.ask_price
        assert slippage >= 0
        
    def test_performance_calculation(self):
        """Test performance metric calculation."""
        engine = PaperTradingEngine()
        
        # No trades yet
        metrics = engine.calculate_performance()
        assert metrics['total_pnl'] == 0
        assert metrics['win_rate'] == 0
        
        # Generate report
        report = engine.generate_report()
        assert not report.empty
        assert report['initial_capital'].iloc[0] == 10000


class TestCarverFramework:
    """Test Carver-style systematic framework."""
    
    def test_forecast_combiner(self):
        """Test forecast combination."""
        combiner = ForecastCombiner()
        
        # Create test forecasts
        forecasts = pd.DataFrame({
            'momentum': [5, -3, 10, -8, 2],
            'carry': [2, 4, -1, 3, 6],
            'value': [-2, 1, -5, 4, -3]
        })
        
        # Calculate equal weights
        weights, div_mult = combiner.calculate_forecast_weights(
            forecasts, method='equal'
        )
        
        assert len(weights) == 3
        assert abs(weights.sum() - 1.0) < 0.01
        assert 0 < div_mult <= 1.0
        
        # Combine forecasts
        combined = combiner.combine_forecasts(forecasts, weights, div_mult)
        
        assert len(combined) == 5
        assert all(-20 <= f <= 20 for f in combined)
        
    def test_volatility_calculator(self):
        """Test volatility calculation."""
        calc = VolatilityCalculator()
        
        # Create test returns
        returns = pd.Series(np.random.randn(100) * 0.02)
        
        # Calculate volatility
        vol = calc.calculate_volatility(returns, method='simple')
        
        assert len(vol) == 100
        assert vol.iloc[-1] > 0  # Positive volatility
        assert vol.iloc[-1] < 1.0  # Reasonable range
        
    def test_position_sizer(self):
        """Test position sizing."""
        sizer = PositionSizer(
            capital=100000,
            volatility_target=0.16,
            instrument_weight=0.25,
            leverage=1.0
        )
        
        # Calculate position
        position = sizer.calculate_position(
            forecast=10,  # Half of max forecast
            volatility=0.20,
            price=2.0
        )
        
        assert position > 0
        assert position < 100000  # Less than total capital
        
        # Test with negative forecast (no shorting in betting)
        position_neg = sizer.calculate_position(
            forecast=-10,
            volatility=0.20,
            price=2.0
        )
        
        assert position_neg == 0
        
    def test_trading_rules(self):
        """Test trading rule configuration."""
        rule = TradingRule(
            name="test_momentum",
            forecast_scalar=8.0,
            turnover=12.0
        )
        
        assert rule.name == "test_momentum"
        assert rule.forecast_cap == 20.0
        assert rule.forecast_floor == -20.0
        
    def test_systematic_framework(self):
        """Test complete systematic framework."""
        framework = SystematicFramework(capital=100000)
        
        # Add trading rules
        framework.add_trading_rule(TradingRule(
            name='momentum_20',
            forecast_scalar=8.0
        ))
        
        framework.add_trading_rule(TradingRule(
            name='carry',
            forecast_scalar=10.0
        ))
        
        assert len(framework.trading_rules) == 2
        
        # Add instrument
        from carver_framework import Instrument
        
        instrument = Instrument(
            market_id='TEST_001',
            sport='NFL',
            league='NFL'
        )
        
        framework.add_instrument(instrument)
        assert len(framework.instruments) == 1
        
        # Test portfolio weight calculation
        returns = {'TEST_001': pd.Series(np.random.randn(100) * 0.01)}
        
        weights = framework.calculate_portfolio_weights(
            returns, method='equal'
        )
        
        assert weights['TEST_001'] == 1.0  # Only one instrument