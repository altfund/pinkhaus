#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Carver-Style Systematic Trading Framework
Adapts Robert Carver's systematic futures trading approach to sports betting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Any
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.covariance import LedoitWolf

logger = logging.getLogger(__name__)


@dataclass
class TradingRule:
    """A systematic trading rule following Carver's approach."""
    name: str
    forecast_scalar: float  # Scales raw forecast to [-20, +20] range
    forecast_cap: float = 20.0
    forecast_floor: float = -20.0
    turnover: float = 0.0  # Annual turnover for cost calculation
    
    
@dataclass
class Instrument:
    """Represents a betting market (adapted from futures contract)."""
    market_id: str
    sport: str
    league: str
    volatility_target: float = 0.16  # 16% annualized vol target
    leverage: float = 1.0
    
    
@dataclass
class PortfolioWeights:
    """Portfolio allocation weights."""
    instrument_weights: Dict[str, float]
    forecast_weights: Dict[str, Dict[str, float]]  # Per instrument
    forecast_diversification_multiplier: float
    instrument_diversification_multiplier: float


class ForecastCombiner:
    """
    Combines multiple forecasts using Carver's approach.
    - Equal weights as default
    - Correlation adjustments
    - Diversification multiplier
    """
    
    def __init__(self, min_correlation_periods: int = 250):
        self.min_correlation_periods = min_correlation_periods
        
    def calculate_forecast_weights(self, 
                                 forecasts: pd.DataFrame,
                                 method: str = 'equal') -> Tuple[pd.Series, float]:
        """
        Calculate forecast weights and diversification multiplier.
        
        Returns:
            weights: Forecast weights
            div_mult: Diversification multiplier
        """
        n_forecasts = len(forecasts.columns)
        
        if method == 'equal':
            weights = pd.Series(1.0 / n_forecasts, index=forecasts.columns)
        elif method == 'inverse_variance':
            weights = self._inverse_variance_weights(forecasts)
        elif method == 'mean_variance':
            weights = self._mean_variance_weights(forecasts)
        else:
            raise ValueError(f"Unknown method: {method}")
            
        # Calculate diversification multiplier
        if len(forecasts) >= self.min_correlation_periods:
            corr_matrix = forecasts.corr()
            avg_correlation = self._average_correlation(corr_matrix)
            div_mult = 1.0 / np.sqrt(1 + (n_forecasts - 1) * avg_correlation)
        else:
            # Conservative assumption if not enough data
            div_mult = 1.0 / np.sqrt(n_forecasts)
            
        return weights, div_mult
        
    def _inverse_variance_weights(self, forecasts: pd.DataFrame) -> pd.Series:
        """Calculate inverse variance weights."""
        variances = forecasts.var()
        inv_var = 1.0 / variances
        return inv_var / inv_var.sum()
        
    def _mean_variance_weights(self, forecasts: pd.DataFrame) -> pd.Series:
        """Calculate mean-variance optimal weights."""
        # Use Ledoit-Wolf shrinkage for robust covariance
        lw = LedoitWolf()
        cov_matrix = lw.fit(forecasts.values).covariance_
        
        # Equal expected returns assumption
        n = len(forecasts.columns)
        expected_returns = np.ones(n) / n
        
        # Solve for weights (simplified - no constraints)
        inv_cov = np.linalg.inv(cov_matrix)
        weights = inv_cov @ expected_returns
        weights = weights / weights.sum()
        
        return pd.Series(weights, index=forecasts.columns)
        
    def _average_correlation(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate average pairwise correlation."""
        n = len(corr_matrix)
        if n <= 1:
            return 0
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = np.triu_indices(n, k=1)
        correlations = corr_matrix.values[upper_triangle]
        
        return correlations.mean()
        
    def combine_forecasts(self, 
                         forecasts: pd.DataFrame,
                         weights: pd.Series,
                         div_mult: float) -> pd.Series:
        """Combine multiple forecasts into single forecast."""
        # Weighted average
        combined = (forecasts * weights).sum(axis=1)
        
        # Apply diversification multiplier
        combined = combined * div_mult
        
        # Cap at [-20, +20] range
        return combined.clip(-20, 20)


class VolatilityCalculator:
    """
    Calculates volatility for position sizing.
    Uses multiple methods with blending.
    """
    
    def __init__(self, 
                 lookback_days: int = 25,
                 min_periods: int = 10,
                 ewm_span: int = 32):
        self.lookback_days = lookback_days
        self.min_periods = min_periods
        self.ewm_span = ewm_span
        
    def calculate_volatility(self, returns: pd.Series, 
                           method: str = 'combined') -> pd.Series:
        """Calculate volatility using specified method."""
        if method == 'simple':
            return self._simple_volatility(returns)
        elif method == 'ewma':
            return self._ewma_volatility(returns)
        elif method == 'yang_zhang':
            return self._yang_zhang_volatility(returns)
        elif method == 'combined':
            return self._combined_volatility(returns)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _simple_volatility(self, returns: pd.Series) -> pd.Series:
        """Simple rolling standard deviation."""
        return returns.rolling(
            window=self.lookback_days,
            min_periods=self.min_periods
        ).std() * np.sqrt(252)  # Annualize
        
    def _ewma_volatility(self, returns: pd.Series) -> pd.Series:
        """Exponentially weighted volatility."""
        return returns.ewm(
            span=self.ewm_span,
            min_periods=self.min_periods
        ).std() * np.sqrt(252)
        
    def _yang_zhang_volatility(self, returns: pd.Series) -> pd.Series:
        """Yang-Zhang volatility estimator (simplified)."""
        # For betting, we don't have OHLC, so fall back to close-to-close
        return self._simple_volatility(returns)
        
    def _combined_volatility(self, returns: pd.Series) -> pd.Series:
        """Combine multiple volatility estimates."""
        simple = self._simple_volatility(returns)
        ewma = self._ewma_volatility(returns)
        
        # Average with slight bias toward EWMA (more responsive)
        combined = 0.4 * simple + 0.6 * ewma
        
        # Floor at 1% annualized to avoid division issues
        return combined.clip(lower=0.01)


class PositionSizer:
    """
    Calculate position sizes using Carver's methodology.
    Adapts futures concepts to betting.
    """
    
    def __init__(self, 
                 capital: float,
                 volatility_target: float = 0.16,
                 instrument_weight: float = 1.0,
                 leverage: float = 1.0):
        self.capital = capital
        self.volatility_target = volatility_target
        self.instrument_weight = instrument_weight
        self.leverage = leverage
        
    def calculate_position(self, 
                         forecast: float,
                         volatility: float,
                         price: float) -> float:
        """
        Calculate position size for a bet.
        
        Carver formula adapted:
        N = (capital * IDM * instrument_weight * volatility_target) / 
            (instrument_price * volatility * 10)
        
        Then multiply by forecast/10 for subsystem position.
        """
        if volatility <= 0 or price <= 0:
            return 0
            
        # Volatility scalar (convert forecast to position)
        volatility_scalar = self.volatility_target / (volatility * 10)
        
        # Base position (in currency units)
        base_position = self.capital * self.instrument_weight * volatility_scalar
        
        # Apply forecast
        position = base_position * (forecast / 10.0)
        
        # Apply leverage cap
        position = position * min(self.leverage, 1.0)
        
        # Convert to bet size (number of units at given price)
        bet_size = abs(position) / price
        
        return bet_size if forecast > 0 else 0  # Only positive bets in sports
        
    def calculate_subsystem_position(self,
                                   forecast: float,
                                   forecast_scalar: float,
                                   volatility: float) -> float:
        """Calculate subsystem position before instrument weights."""
        # Scale forecast
        scaled_forecast = forecast * forecast_scalar
        scaled_forecast = np.clip(scaled_forecast, -20, 20)
        
        # Volatility scalar
        if volatility > 0:
            vol_scalar = self.volatility_target / volatility
        else:
            vol_scalar = 0
            
        # Subsystem position
        return scaled_forecast * vol_scalar / 10.0


class TradingCosts:
    """
    Model trading costs for sports betting.
    Includes spread, commission, and market impact.
    """
    
    def __init__(self,
                 commission_rate: float = 0.02,
                 spread_cost: float = 0.01,
                 price_slippage: float = 0.005):
        self.commission_rate = commission_rate
        self.spread_cost = spread_cost
        self.price_slippage = price_slippage
        
    def calculate_costs(self, 
                       trade_value: float,
                       turnover: float) -> float:
        """Calculate expected trading costs."""
        # Fixed costs per trade
        fixed_costs = self.commission_rate + self.spread_cost + self.price_slippage
        
        # Adjust for turnover
        annual_costs = fixed_costs * turnover
        
        return trade_value * annual_costs
        
    def calculate_sr_cost(self, turnover: float, 
                         expected_sr: float = 0.5) -> float:
        """
        Calculate Sharpe ratio degradation from costs.
        Uses Carver's approximation.
        """
        total_cost_rate = (self.commission_rate + self.spread_cost + 
                          self.price_slippage) * turnover
        
        # SR degradation approximation
        return expected_sr * total_cost_rate * 2


class SystematicFramework:
    """
    Complete systematic trading framework for sports betting.
    Implements Carver's modular approach.
    """
    
    def __init__(self,
                 capital: float = 100000,
                 volatility_target: float = 0.16):
        self.capital = capital
        self.volatility_target = volatility_target
        
        # Components
        self.forecast_combiner = ForecastCombiner()
        self.volatility_calculator = VolatilityCalculator()
        self.trading_costs = TradingCosts()
        
        # Rules and instruments
        self.trading_rules: Dict[str, TradingRule] = {}
        self.instruments: Dict[str, Instrument] = {}
        
        # Weights
        self.portfolio_weights = PortfolioWeights(
            instrument_weights={},
            forecast_weights={},
            forecast_diversification_multiplier=1.0,
            instrument_diversification_multiplier=1.0
        )
        
    def add_trading_rule(self, rule: TradingRule):
        """Add a trading rule to the system."""
        self.trading_rules[rule.name] = rule
        logger.info(f"Added trading rule: {rule.name}")
        
    def add_instrument(self, instrument: Instrument):
        """Add an instrument (betting market) to the system."""
        self.instruments[instrument.market_id] = instrument
        logger.info(f"Added instrument: {instrument.market_id}")
        
    def calculate_raw_forecasts(self, 
                              market_data: pd.DataFrame,
                              market_id: str) -> pd.DataFrame:
        """Calculate raw forecasts for all trading rules."""
        forecasts = pd.DataFrame(index=market_data.index)
        
        for rule_name, rule in self.trading_rules.items():
            # Each rule would implement its own forecast calculation
            # For now, simulate with different momentum periods
            if 'momentum' in rule_name:
                period = int(rule_name.split('_')[1])
                returns = market_data['close'].pct_change()
                raw_forecast = returns.rolling(period).mean()
                
                # Scale to roughly -1 to +1 range
                raw_forecast = raw_forecast / returns.std()
                
            elif 'carry' in rule_name:
                # Simulate carry (edge in betting odds)
                if 'implied_prob' in market_data and 'true_prob' in market_data:
                    raw_forecast = (market_data['true_prob'] - 
                                  market_data['implied_prob'])
                else:
                    raw_forecast = pd.Series(0, index=market_data.index)
                    
            elif 'value' in rule_name:
                # Simulate value signal
                raw_forecast = -market_data['close'].rolling(50).apply(
                    lambda x: stats.zscore(x)[-1]
                )
            else:
                raw_forecast = pd.Series(0, index=market_data.index)
                
            forecasts[rule_name] = raw_forecast
            
        return forecasts
        
    def scale_forecasts(self, 
                       raw_forecasts: pd.DataFrame) -> pd.DataFrame:
        """Scale raw forecasts to [-20, +20] range."""
        scaled = pd.DataFrame(index=raw_forecasts.index)
        
        for rule_name, raw in raw_forecasts.items():
            rule = self.trading_rules[rule_name]
            
            # Apply forecast scalar
            scaled_forecast = raw * rule.forecast_scalar
            
            # Cap at limits
            scaled[rule_name] = scaled_forecast.clip(
                rule.forecast_floor, 
                rule.forecast_cap
            )
            
        return scaled
        
    def calculate_combined_forecast(self,
                                  scaled_forecasts: pd.DataFrame,
                                  market_id: str) -> pd.Series:
        """Combine multiple forecasts for an instrument."""
        # Get weights for this instrument
        if market_id in self.portfolio_weights.forecast_weights:
            weights = pd.Series(self.portfolio_weights.forecast_weights[market_id])
        else:
            # Calculate and cache weights
            weights, div_mult = self.forecast_combiner.calculate_forecast_weights(
                scaled_forecasts
            )
            self.portfolio_weights.forecast_weights[market_id] = weights.to_dict()
            
        # Use appropriate diversification multiplier
        div_mult = self.portfolio_weights.forecast_diversification_multiplier
        
        # Combine
        return self.forecast_combiner.combine_forecasts(
            scaled_forecasts, weights, div_mult
        )
        
    def calculate_positions(self,
                          market_data: Dict[str, pd.DataFrame],
                          forecasts: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calculate positions for all instruments."""
        positions = pd.DataFrame()
        
        for market_id, data in market_data.items():
            if market_id not in forecasts:
                continue
                
            instrument = self.instruments.get(market_id)
            if not instrument:
                continue
                
            # Calculate returns
            returns = data['close'].pct_change()
            
            # Calculate volatility
            volatility = self.volatility_calculator.calculate_volatility(returns)
            
            # Get instrument weight
            inst_weight = self.portfolio_weights.instrument_weights.get(
                market_id, 1.0 / len(self.instruments)
            )
            
            # Position sizer
            sizer = PositionSizer(
                capital=self.capital,
                volatility_target=self.volatility_target,
                instrument_weight=inst_weight,
                leverage=instrument.leverage
            )
            
            # Calculate positions
            market_positions = []
            for idx in data.index:
                if idx in forecasts[market_id].index and idx in volatility.index:
                    forecast = forecasts[market_id].loc[idx]
                    vol = volatility.loc[idx]
                    price = data.loc[idx, 'close']
                    
                    position = sizer.calculate_position(forecast, vol, price)
                    market_positions.append(position)
                else:
                    market_positions.append(0)
                    
            positions[market_id] = market_positions
            
        positions.index = data.index
        return positions
        
    def calculate_portfolio_weights(self,
                                  returns: Dict[str, pd.Series],
                                  method: str = 'equal') -> Dict[str, float]:
        """Calculate instrument weights."""
        n_instruments = len(returns)
        
        if method == 'equal':
            weight = 1.0 / n_instruments
            return {inst: weight for inst in returns.keys()}
            
        elif method == 'inverse_variance':
            variances = {inst: ret.var() for inst, ret in returns.items()}
            total_inv_var = sum(1/v for v in variances.values() if v > 0)
            
            weights = {}
            for inst, var in variances.items():
                if var > 0:
                    weights[inst] = (1/var) / total_inv_var
                else:
                    weights[inst] = 0
                    
            return weights
            
        elif method == 'risk_parity':
            # Simplified risk parity
            vols = {inst: ret.std() for inst, ret in returns.items()}
            total_inv_vol = sum(1/v for v in vols.values() if v > 0)
            
            weights = {}
            for inst, vol in vols.items():
                if vol > 0:
                    weights[inst] = (1/vol) / total_inv_vol
                else:
                    weights[inst] = 0
                    
            return weights
            
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def run_system(self,
                  market_data: Dict[str, pd.DataFrame],
                  start_date: datetime,
                  end_date: datetime) -> Dict[str, Any]:
        """Run the complete systematic trading system."""
        logger.info("Running systematic trading system")
        
        # Storage for results
        all_forecasts = {}
        all_positions = {}
        
        # Process each instrument
        for market_id, data in market_data.items():
            if market_id not in self.instruments:
                continue
                
            # Filter data to date range
            mask = (data.index >= start_date) & (data.index <= end_date)
            data = data[mask]
            
            # Calculate raw forecasts
            raw_forecasts = self.calculate_raw_forecasts(data, market_id)
            
            # Scale forecasts
            scaled_forecasts = self.scale_forecasts(raw_forecasts)
            
            # Combine forecasts
            combined_forecast = self.calculate_combined_forecast(
                scaled_forecasts, market_id
            )
            
            all_forecasts[market_id] = combined_forecast
            
        # Calculate positions
        positions = self.calculate_positions(market_data, all_forecasts)
        
        # Calculate performance
        performance = self._calculate_performance(
            positions, market_data, start_date, end_date
        )
        
        return {
            'forecasts': all_forecasts,
            'positions': positions,
            'performance': performance,
            'weights': self.portfolio_weights
        }
        
    def _calculate_performance(self,
                             positions: pd.DataFrame,
                             market_data: Dict[str, pd.DataFrame],
                             start_date: datetime,
                             end_date: datetime) -> Dict[str, float]:
        """Calculate system performance metrics."""
        # Simulate returns (would use actual outcomes in practice)
        returns = []
        
        for idx in positions.index:
            daily_return = 0
            
            for market_id in positions.columns:
                if market_id in market_data:
                    position = positions.loc[idx, market_id]
                    
                    # Simulate outcome (50% + edge based on forecast)
                    if position > 0:
                        # Simple simulation - would use real outcomes
                        market_return = market_data[market_id].loc[idx, 'close'] * 0.01
                        daily_return += position * market_return
                        
            returns.append(daily_return / self.capital)
            
        returns = pd.Series(returns, index=positions.index)
        
        # Calculate metrics
        total_return = returns.sum()
        
        if len(returns) > 1 and returns.std() > 0:
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
        else:
            sharpe = 0
            
        cumulative = (1 + returns).cumprod()
        drawdowns = cumulative / cumulative.cummax() - 1
        max_dd = drawdowns.min()
        
        # Calculate turnover
        position_changes = positions.diff().abs().sum(axis=1)
        avg_turnover = position_changes.mean() * 252  # Annualized
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'turnover': avg_turnover,
            'n_trades': (positions.diff() != 0).sum().sum()
        }


def create_carver_system(capital: float = 100000) -> SystematicFramework:
    """Create a pre-configured Carver-style system."""
    system = SystematicFramework(capital=capital)
    
    # Add standard trading rules
    system.add_trading_rule(TradingRule(
        name='momentum_20',
        forecast_scalar=8.0,
        turnover=12.0
    ))
    
    system.add_trading_rule(TradingRule(
        name='momentum_50',
        forecast_scalar=4.0,
        turnover=6.0
    ))
    
    system.add_trading_rule(TradingRule(
        name='carry',
        forecast_scalar=10.0,
        turnover=4.0
    ))
    
    system.add_trading_rule(TradingRule(
        name='value',
        forecast_scalar=2.0,
        turnover=2.0
    ))
    
    return system


def main():
    """Example usage of Carver framework."""
    # Create system
    system = create_carver_system(capital=100000)
    
    # Add some instruments
    system.add_instrument(Instrument(
        market_id='NFL_2024_W1_TB_DAL',
        sport='NFL',
        league='NFL'
    ))
    
    system.add_instrument(Instrument(
        market_id='NBA_2024_LAL_BOS',
        sport='NBA',
        league='NBA'
    ))
    
    # Generate dummy data
    dates = pd.date_range('2024-01-01', '2024-03-31', freq='H')
    
    market_data = {}
    for inst in system.instruments.values():
        data = pd.DataFrame({
            'close': 2.0 + 0.5 * np.random.randn(len(dates)).cumsum() * 0.01,
            'volume': 1000 + 500 * np.random.random(len(dates)),
            'implied_prob': 0.5 + 0.1 * np.random.randn(len(dates)),
            'true_prob': 0.5 + 0.1 * np.random.randn(len(dates))
        }, index=dates)
        market_data[inst.market_id] = data
    
    # Run system
    results = system.run_system(
        market_data,
        datetime(2024, 1, 1),
        datetime(2024, 3, 31)
    )
    
    print("Performance Summary:")
    print(f"Sharpe Ratio: {results['performance']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['performance']['max_drawdown']:.2%}")
    print(f"Turnover: {results['performance']['turnover']:.1f}x per year")
    print(f"Total Trades: {results['performance']['n_trades']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()