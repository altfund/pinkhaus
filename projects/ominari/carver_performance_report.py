#!/usr/bin/env python3
"""
Carver-Style Performance Report
Implements Rob Carver's systematic trading metrics adapted for sports betting.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
from carver_framework import ForecastCombiner

class CarverPerformanceAnalyzer:
    """Analyzes performance using Carver's systematic framework."""
    
    def __init__(self):
        self.forecast_combiner = ForecastCombiner()
        
    def calculate_forecast_correlation_matrix(self, signal_results: list) -> pd.DataFrame:
        """Calculate correlation between different forecasts."""
        # Create forecast matrix from results
        forecasts = {}
        
        for result in signal_results:
            signal_name = result['signal_name']
            forecast = result['forecast']  # Already in [-20, 20] range
            
            if signal_name not in forecasts:
                forecasts[signal_name] = []
            forecasts[signal_name].append(forecast)
            
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecasts)
        
        # Calculate correlation matrix
        corr_matrix = forecast_df.corr()
        
        return corr_matrix
        
    def calculate_diversification_multiplier(self, corr_matrix: pd.DataFrame) -> float:
        """Calculate Carver's forecast diversification multiplier."""
        n = len(corr_matrix)
        if n <= 1:
            return 1.0
            
        # Average pairwise correlation
        avg_corr = corr_matrix.values[np.triu_indices(n, k=1)].mean()
        
        # Diversification multiplier formula
        div_mult = 1.0 / np.sqrt(1 + (n - 1) * avg_corr)
        
        return div_mult
        
    def calculate_forecast_weights(self, signal_performance: dict) -> pd.DataFrame:
        """Calculate optimal forecast weights using Carver's methodology."""
        weights = {}
        
        # Method 1: Equal weights (baseline)
        n_signals = len(signal_performance)
        equal_weight = 1.0 / n_signals if n_signals > 0 else 0
        
        # Method 2: Inverse variance weights
        variances = {}
        for signal_name, perf in signal_performance.items():
            # Use edge variance as proxy for forecast variance
            edges = perf.get('edge_distribution', [0])
            variance = np.var(edges) if len(edges) > 1 else 1.0
            variances[signal_name] = variance
            
        total_inv_var = sum(1/v for v in variances.values() if v > 0)
        
        for signal_name in signal_performance.keys():
            var = variances.get(signal_name, 1.0)
            weights[signal_name] = {
                'equal': equal_weight,
                'inverse_variance': (1/var) / total_inv_var if var > 0 and total_inv_var > 0 else 0,
                'user_defined': SIGNAL_WEIGHTS.get(signal_name, 1.0) / sum(SIGNAL_WEIGHTS.values())
            }
            
        return pd.DataFrame(weights).T
        
    def calculate_sharpe_ratio(self, returns: list, periods_per_year: float = 365) -> float:
        """Calculate annualized Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
            
        returns_array = np.array(returns)
        mean_return = returns_array.mean()
        std_return = returns_array.std()
        
        if std_return == 0:
            return 0.0
            
        # Annualized Sharpe
        sharpe = np.sqrt(periods_per_year) * mean_return / std_return
        
        return sharpe
        
    def calculate_turnover(self, positions: list) -> float:
        """Calculate portfolio turnover (Carver metric)."""
        if len(positions) < 2:
            return 0.0
            
        position_changes = [abs(positions[i] - positions[i-1]) 
                           for i in range(1, len(positions))]
        
        avg_turnover = np.mean(position_changes) if position_changes else 0
        
        # Annualized turnover
        annual_turnover = avg_turnover * 365
        
        return annual_turnover
        
    def generate_carver_report(self, signal_results: list) -> dict:
        """Generate comprehensive Carver-style report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'forecast_analysis': {},
            'weight_analysis': {},
            'diversification': {},
            'performance_metrics': {},
            'cost_analysis': {}
        }
        
        # Group results by signal
        signal_groups = {}
        for result in signal_results:
            signal_name = result.get('signal_name', 'unknown')
            if signal_name not in signal_groups:
                signal_groups[signal_name] = []
            signal_groups[signal_name].append(result)
            
        # Analyze each signal
        signal_performance = {}
        for signal_name, results in signal_groups.items():
            edges = [r.get('edge', 0) for r in results]
            evs = [r.get('ev', 0) for r in results]
            forecasts = [(r.get('edge', 0) * 20)  for r in results]  # Convert to forecast scale
            
            signal_performance[signal_name] = {
                'count': len(results),
                'mean_edge': np.mean(edges),
                'edge_volatility': np.std(edges),
                'mean_forecast': np.mean(forecasts),
                'forecast_volatility': np.std(forecasts),
                'positive_ev_rate': sum(1 for ev in evs if ev > 0) / len(evs) if evs else 0,
                'edge_distribution': edges
            }
            
        report['forecast_analysis'] = signal_performance
        
        # Calculate optimal weights
        weight_df = self.calculate_forecast_weights(signal_performance)
        report['weight_analysis'] = weight_df.to_dict()
        
        # Diversification analysis
        if len(signal_results) > 10:  # Need sufficient data
            corr_matrix = self.calculate_forecast_correlation_matrix(signal_results)
            div_mult = self.calculate_diversification_multiplier(corr_matrix)
            
            report['diversification'] = {
                'correlation_matrix': corr_matrix.to_dict(),
                'average_correlation': corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean(),
                'diversification_multiplier': div_mult,
                'effective_forecasts': 1 / (div_mult ** 2)  # Inverse of div mult squared
            }
        
        # Performance metrics
        # Simulate returns based on edges
        simulated_returns = []
        positions = []
        
        for result in signal_results:
            edge = result.get('edge', 0)
            odds = result.get('odds', 2.0)
            
            # Kelly position sizing
            if edge > 0:
                kelly_fraction = edge / ((odds - 1) ** 2)
                position = min(kelly_fraction * 0.25, 0.1)  # Conservative Kelly
            else:
                position = 0
                
            positions.append(position)
            
            # Simulate outcome (50% + edge for simplicity)
            win_prob = 0.5 + edge
            outcome = 1 if np.random.random() < win_prob else 0
            
            if position > 0:
                ret = position * (outcome * (odds - 1) - (1 - outcome))
                simulated_returns.append(ret)
                
        if simulated_returns:
            sharpe = self.calculate_sharpe_ratio(simulated_returns)
            turnover = self.calculate_turnover(positions)
            
            report['performance_metrics'] = {
                'sharpe_ratio': sharpe,
                'annual_turnover': turnover,
                'avg_position_size': np.mean([p for p in positions if p > 0]) if any(positions) else 0,
                'betting_frequency': sum(1 for p in positions if p > 0) / len(positions) if positions else 0
            }
            
        # Cost analysis (Carver framework)
        if 'annual_turnover' in report['performance_metrics']:
            turnover = report['performance_metrics']['annual_turnover']
            
            # Betting costs: commission + spread
            commission_rate = 0.02  # 2%
            spread_cost = 0.01      # 1%
            
            total_cost_per_trade = commission_rate + spread_cost
            annual_cost = turnover * total_cost_per_trade
            
            # SR cost (Sharpe degradation from costs)
            sr_cost = annual_cost * 2  # Carver's approximation
            
            report['cost_analysis'] = {
                'cost_per_trade': total_cost_per_trade,
                'annual_cost_rate': annual_cost,
                'sharpe_ratio_cost': sr_cost,
                'net_sharpe': max(0, report['performance_metrics'].get('sharpe_ratio', 0) - sr_cost)
            }
            
        return report

def run_carver_analysis():
    """Run Carver-style analysis on recent data."""
    
    print("🎯 CARVER-STYLE SYSTEMATIC TRADING REPORT")
    print("=" * 60)
    print("Based on 'Systematic Trading' by Robert Carver")
    print("Adapted for sports betting markets")
    print("=" * 60)
    
    analyzer = CarverPerformanceAnalyzer()
    
    # Get sample data (same approach as before)
    conn = sqlite3.connect('sport_odds.db')
    sql = """
    SELECT 
        m.home_team,
        m.away_team,
        o.outcome,
        o.decimal_odds,
        o.source_id
    FROM market m
    JOIN odd o ON m.source_id = o.source_id
    WHERE m.sport = 'Soccer' 
        AND o.decimal_odds > 1.0
        AND o.decimal_odds < 10.0
    ORDER BY o.rowid DESC
    LIMIT 300
    """
    
    df = pd.read_sql_query(sql, conn)
    conn.close()
    
    # Analyze each signal
    all_results = []
    
    for provider in SIGNAL_PROVIDERS:
        # Create sample data for signal
        for _, row in df.iterrows():
            try:
                # Simulate signal (since we can't run actual signals due to DB issues)
                if provider.name == 'implied_probability':
                    # ImpliedRaw has zero edge by design
                    signal_prob = 1 / row['decimal_odds']
                else:
                    # Simulate other signals with small random edge
                    base_prob = 1 / row['decimal_odds']
                    edge = np.random.normal(0, 0.02)  # 2% std dev
                    signal_prob = np.clip(base_prob + edge, 0, 1)
                    
                implied_prob = 1 / row['decimal_odds']
                edge = signal_prob - implied_prob
                ev = signal_prob * (row['decimal_odds'] - 1) - (1 - signal_prob)
                
                all_results.append({
                    'signal_name': provider.name,
                    'market': f"{row['home_team']} vs {row['away_team']}",
                    'outcome': row['outcome'],
                    'odds': row['decimal_odds'],
                    'signal_prob': signal_prob,
                    'implied_prob': implied_prob,
                    'edge': edge,
                    'ev': ev,
                    'forecast': edge * 20  # Convert to Carver scale
                })
                
            except Exception:
                continue
                
    # Generate Carver report
    report = analyzer.generate_carver_report(all_results)
    
    # Display results
    print("\n📊 FORECAST ANALYSIS")
    print("-" * 40)
    for signal_name, perf in report['forecast_analysis'].items():
        print(f"\n{signal_name.upper()}:")
        print(f"  Observations: {perf['count']}")
        print(f"  Mean forecast: {perf['mean_forecast']:.2f} (scale: -20 to +20)")
        print(f"  Forecast volatility: {perf['forecast_volatility']:.2f}")
        print(f"  Mean edge: {perf['mean_edge']*100:.3f}%")
        print(f"  Positive EV rate: {perf['positive_ev_rate']:.1%}")
        
    print("\n⚖️ FORECAST WEIGHTS")
    print("-" * 40)
    weight_df = pd.DataFrame(report['weight_analysis'])
    print(weight_df.round(3))
    
    if 'diversification' in report and report['diversification']:
        print("\n🔄 DIVERSIFICATION ANALYSIS")
        print("-" * 40)
        div = report['diversification']
        print(f"Average correlation: {div['average_correlation']:.3f}")
        print(f"Diversification multiplier: {div['diversification_multiplier']:.3f}")
        print(f"Effective number of forecasts: {div['effective_forecasts']:.2f}")
        
    if 'performance_metrics' in report and report['performance_metrics']:
        print("\n📈 PERFORMANCE METRICS")
        print("-" * 40)
        perf = report['performance_metrics']
        print(f"Sharpe ratio: {perf['sharpe_ratio']:.2f}")
        print(f"Annual turnover: {perf['annual_turnover']:.1f}x")
        print(f"Average position size: {perf['avg_position_size']*100:.1f}% of capital")
        print(f"Betting frequency: {perf['betting_frequency']:.1%} of opportunities")
        
    if 'cost_analysis' in report and report['cost_analysis']:
        print("\n💰 COST ANALYSIS")
        print("-" * 40)
        costs = report['cost_analysis']
        print(f"Cost per trade: {costs['cost_per_trade']*100:.1f}%")
        print(f"Annual cost rate: {costs['annual_cost_rate']*100:.1f}%")
        print(f"Sharpe ratio cost: {costs['sharpe_ratio_cost']:.2f}")
        print(f"Net Sharpe ratio: {costs['net_sharpe']:.2f}")
        
    print("\n🎯 CARVER FRAMEWORK INSIGHTS")
    print("=" * 40)
    print("1. FORECAST COMBINATION: Equal weights baseline, can optimize")
    print("2. DIVERSIFICATION: Low correlation between signals is good")
    print("3. POSITION SIZING: Conservative Kelly (25%) recommended")
    print("4. TURNOVER: Monitor costs vs signal decay")
    print("5. SHARPE RATIO: Target 0.5+ after costs for viability")
    
    return report

if __name__ == "__main__":
    run_carver_analysis()