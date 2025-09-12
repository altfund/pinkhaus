#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Analysis Tool for Ominari Trading System
Analyzes paper trading performance and generates reports.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from pathlib import Path
import json
import matplotlib.pyplot as plt
from typing import Dict


class PortfolioAnalyzer:
    """Analyze paper trading portfolio performance."""
    
    def __init__(self):
        self.portfolio_file = Path("paper_portfolio.json")
        self.trades_file = Path("paper_trades.csv")
        self.load_data()
        
    def load_data(self):
        """Load portfolio and trade data."""
        # Load portfolio
        if self.portfolio_file.exists():
            with open(self.portfolio_file, 'r') as f:
                self.portfolio = json.load(f)
        else:
            self.portfolio = None
            
        # Load trades
        if self.trades_file.exists():
            self.trades_df = pd.read_csv(self.trades_file)
            self.trades_df['timestamp'] = pd.to_datetime(self.trades_df['timestamp'])
            self.trades_df['kick_off'] = pd.to_datetime(self.trades_df['kick_off'])
        else:
            self.trades_df = pd.DataFrame()
    
    def calculate_returns(self) -> pd.DataFrame:
        """Calculate returns from completed trades."""
        if self.trades_df.empty:
            return pd.DataFrame()
            
        # For now, simulate some results (in production, this would check actual results)
        # This is where you'd integrate with the result checking system
        returns = []
        
        for _, trade in self.trades_df.iterrows():
            # Simulate win/loss based on edge (temporary - replace with actual results)
            win_probability = trade['signal_prob']
            won = np.random.random() < win_probability
            
            if won:
                profit = trade['stake'] * (trade['odds'] - 1)
            else:
                profit = -trade['stake']
                
            returns.append({
                'timestamp': trade['timestamp'],
                'match': trade['match'],
                'stake': trade['stake'],
                'odds': trade['odds'],
                'profit': profit,
                'return_pct': profit / trade['stake'] * 100,
                'cumulative': 0,  # Will calculate after
                'won': won
            })
        
        returns_df = pd.DataFrame(returns)
        
        # Calculate cumulative returns
        if not returns_df.empty:
            returns_df['cumulative'] = returns_df['profit'].cumsum()
            
        return returns_df
    
    def calculate_metrics(self, returns_df: pd.DataFrame) -> Dict:
        """Calculate key performance metrics."""
        if returns_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_stake': 0,
                'avg_odds': 0
            }
        
        total_trades = len(returns_df)
        wins = returns_df['won'].sum()
        win_rate = wins / total_trades * 100
        
        total_profit = returns_df['profit'].sum()
        total_staked = returns_df['stake'].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0
        
        # Calculate Sharpe ratio (simplified - daily)
        if len(returns_df) > 1:
            daily_returns = returns_df.groupby(returns_df['timestamp'].dt.date)['profit'].sum()
            if len(daily_returns) > 1 and daily_returns.std() > 0:
                sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # Calculate max drawdown
        cumulative = returns_df['cumulative']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / (running_max + 10000)  # Initial bankroll
        max_drawdown = drawdown.min() * 100
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'wins': wins,
            'losses': total_trades - wins,
            'total_profit': total_profit,
            'roi': roi,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'avg_stake': returns_df['stake'].mean(),
            'avg_odds': returns_df['odds'].mean(),
            'total_staked': total_staked
        }
    
    def analyze_by_sport(self) -> pd.DataFrame:
        """Analyze performance by sport."""
        if self.trades_df.empty:
            return pd.DataFrame()
            
        # Extract sport from match string (simplified)
        # In production, you'd join with market data
        sports_performance = []
        
        # Group by some criteria
        # This is placeholder - would need actual sport data
        return pd.DataFrame(sports_performance)
    
    def analyze_by_signal_strength(self) -> pd.DataFrame:
        """Analyze performance by signal strength."""
        if self.trades_df.empty:
            return pd.DataFrame()
            
        # Bin trades by edge
        bins = [0, 0.02, 0.05, 0.10, 1.0]
        labels = ['0-2%', '2-5%', '5-10%', '>10%']
        
        self.trades_df['edge_bin'] = pd.cut(
            self.trades_df['edge'], 
            bins=bins, 
            labels=labels
        )
        
        # Group analysis
        edge_analysis = self.trades_df.groupby('edge_bin').agg({
            'stake': ['count', 'sum', 'mean'],
            'edge': 'mean',
            'signal_prob': 'mean',
            'odds': 'mean'
        }).round(2)
        
        return edge_analysis
    
    def generate_report(self, save_plots: bool = True):
        """Generate comprehensive performance report."""
        print("\n" + "="*80)
        print("OMINARI PORTFOLIO ANALYSIS REPORT")
        print("="*80)
        print(f"Report Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        if self.portfolio is None:
            print("\nNo portfolio data found.")
            return
            
        # Current Portfolio Status
        print("\n📊 CURRENT PORTFOLIO STATUS")
        print("-"*50)
        print(f"Cash Balance: ${self.portfolio['cash']:,.2f}")
        print(f"Active Bets: {len(self.portfolio['pending_bets'])}")
        
        at_risk = sum(bet['stake'] for bet in self.portfolio['pending_bets'].values())
        print(f"Amount at Risk: ${at_risk:,.2f}")
        
        # Historical Performance
        returns_df = self.calculate_returns()
        metrics = self.calculate_metrics(returns_df)
        
        print("\n📈 HISTORICAL PERFORMANCE")
        print("-"*50)
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
        print(f"Total Profit/Loss: ${metrics['total_profit']:,.2f}")
        print(f"ROI: {metrics['roi']:.1f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.1f}%")
        print(f"Average Stake: ${metrics['avg_stake']:.2f}")
        print(f"Average Odds: {metrics['avg_odds']:.2f}")
        
        # Trade Analysis
        if not self.trades_df.empty:
            print("\n🎯 TRADE ANALYSIS")
            print("-"*50)
            
            # By time of day
            self.trades_df['hour'] = pd.to_datetime(self.trades_df['kick_off']).dt.hour
            hourly_dist = self.trades_df['hour'].value_counts().sort_index()
            
            print("\nTrades by Hour of Day:")
            for hour, count in hourly_dist.head(5).items():
                print(f"  {hour:02d}:00 - {count} trades")
            
            # By confidence level
            print("\nTrades by Confidence Level:")
            conf_bins = [0, 0.7, 0.8, 0.9, 1.0]
            conf_labels = ['Low (0-70%)', 'Medium (70-80%)', 'High (80-90%)', 'Very High (90-100%)']
            
            if 'confidence' in self.trades_df.columns:
                self.trades_df['conf_bin'] = pd.cut(
                    self.trades_df['confidence'], 
                    bins=conf_bins, 
                    labels=conf_labels
                )
                conf_dist = self.trades_df['conf_bin'].value_counts()
                for conf, count in conf_dist.items():
                    print(f"  {conf}: {count} trades")
            
            # Edge analysis
            edge_analysis = self.analyze_by_signal_strength()
            if not edge_analysis.empty:
                print("\n📊 Performance by Signal Edge:")
                print(edge_analysis)
        
        # Generate plots
        if save_plots and not returns_df.empty:
            self.generate_plots(returns_df, metrics)
            print("\n📊 Performance plots saved to 'portfolio_plots/'")
        
        # Recommendations
        print("\n💡 RECOMMENDATIONS")
        print("-"*50)
        
        if metrics['win_rate'] < 50:
            print("⚠️  Win rate below 50% - Review signal quality")
        
        if metrics['max_drawdown'] < -20:
            print("⚠️  High drawdown detected - Consider reducing position sizes")
            
        if metrics['sharpe_ratio'] < 1:
            print("⚠️  Low Sharpe ratio - Risk-adjusted returns need improvement")
            
        if at_risk > self.portfolio['cash'] * 0.2:
            print("⚠️  High exposure - More than 20% of capital at risk")
        
        print("\n" + "="*80)
    
    def generate_plots(self, returns_df: pd.DataFrame, metrics: Dict):
        """Generate performance visualization plots."""
        # Create plots directory
        plots_dir = Path("portfolio_plots")
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # 1. Cumulative returns plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        initial_capital = 10000
        cum_returns = initial_capital + returns_df['cumulative']
        
        ax.plot(returns_df['timestamp'], cum_returns, linewidth=2)
        ax.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.7)
        ax.fill_between(returns_df['timestamp'], initial_capital, cum_returns, 
                       where=(cum_returns >= initial_capital), 
                       color='green', alpha=0.3, label='Profit')
        ax.fill_between(returns_df['timestamp'], initial_capital, cum_returns, 
                       where=(cum_returns < initial_capital), 
                       color='red', alpha=0.3, label='Loss')
        
        ax.set_title('Portfolio Value Over Time', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'cumulative_returns.png', dpi=300)
        plt.close()
        
        # 2. Win/Loss distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Win rate pie chart
        sizes = [metrics['wins'], metrics['losses']]
        labels = ['Wins', 'Losses']
        colors = ['#28a745', '#dc3545']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Win/Loss Distribution')
        
        # Profit distribution
        if not returns_df.empty:
            ax2.hist(returns_df['profit'], bins=20, color='skyblue', edgecolor='black')
            ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
            ax2.set_title('Profit Distribution')
            ax2.set_xlabel('Profit ($)')
            ax2.set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(plots_dir / 'win_loss_distribution.png', dpi=300)
        plt.close()
        
        # 3. Edge vs Performance
        if 'edge' in self.trades_df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Create scatter plot
            scatter = ax.scatter(self.trades_df['edge'] * 100, 
                               self.trades_df['stake'],
                               c=self.trades_df['odds'],
                               cmap='viridis', 
                               alpha=0.6,
                               s=100)
            
            ax.set_xlabel('Expected Edge (%)')
            ax.set_ylabel('Stake Size ($)')
            ax.set_title('Stake Size vs Expected Edge')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Odds')
            
            plt.tight_layout()
            plt.savefig(plots_dir / 'edge_vs_stake.png', dpi=300)
            plt.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Portfolio Analysis Tool")
    parser.add_argument('--no-plots', action='store_true', 
                       help="Skip generating plots")
    
    args = parser.parse_args()
    
    analyzer = PortfolioAnalyzer()
    analyzer.generate_report(save_plots=not args.no_plots)


if __name__ == "__main__":
    main()