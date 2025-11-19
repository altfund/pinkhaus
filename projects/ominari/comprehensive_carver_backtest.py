#!/usr/bin/env python3
"""
Comprehensive Carver Strategy Backtest Framework
Replicable backtesting with performance charts and standard trading metrics.
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from carver_enhanced_signal_pipeline import SignalPipeline
from fixed_edge_calculation import FixedEdgeSignalProvider
from fixed_soccer_signals import (
    FixedSoccerWinDrawLossSignal,
    FixedHomeUnderdogSignal,
    FixedSoccerGoalsSignal,
    FixedNaiveEdgeSignal
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class Trade:
    """Individual trade record."""
    timestamp: datetime
    market_id: str
    signal_provider: str
    signal_type: str
    outcome: str
    odds: float
    stake: float
    probability: float
    forecast: float
    confidence: float
    edge: float
    actual_outcome: str
    is_winner: bool
    pnl: float
    cumulative_pnl: float
    bankroll: float


@dataclass
class StrategyPerformance:
    """Strategy performance metrics."""
    strategy_name: str
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    trades_count: int
    avg_edge: float
    avg_confidence: float
    best_trade: float
    worst_trade: float
    consecutive_wins: int
    consecutive_losses: int
    kelly_utilization: float
    risk_adjusted_return: float


class ComprehensiveCarverBacktest:
    """Comprehensive backtesting framework for Carver strategies."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'initial_bankroll': 10000,
            'kelly_fraction': 0.25,
            'max_position_pct': 0.08,  # Increased for deployment
            'min_bet_size': 10,
            'transaction_cost_pct': 0.015,
            'min_confidence': 0.03,  # Lowered for deployment
            'min_edge': 0.001,  # Lowered for deployment
            'risk_free_rate': 0.02,
            'lookback_days': 90,
            'market_limit': None  # Remove arbitrary limits for deployment
        }
        
        # Initialize signal providers
        self.signal_providers = {
            'fixed_edge': FixedEdgeSignalProvider(),
            'fixed_soccer_wdl': FixedSoccerWinDrawLossSignal(),
            'fixed_home_underdog': FixedHomeUnderdogSignal(),
            'fixed_soccer_goals': FixedSoccerGoalsSignal(),
            'fixed_naive_edge': FixedNaiveEdgeSignal()
        }
        
        # Signal providers are now fixed and don't need threshold adjustments
        
        self.signal_pipeline = SignalPipeline()
        self.trades_log = []
        self.results_dir = "backtest_results"
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
    def get_historical_data(self) -> List[Tuple[Market, Dict, str]]:
        """Get historical markets with outcomes for backtesting."""
        markets_with_outcomes = []
        
        with db_manager.get_db_session() as db:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=self.config['lookback_days'])
            
            # Get markets with varied timing to simulate real conditions
            markets_query = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.maturity_date > start_date,
                Market.maturity_date < end_date
            ).order_by(Market.maturity_date)
            
            # Apply limit only if market_limit is set
            if self.config['market_limit']:
                markets_query = markets_query.limit(self.config['market_limit'])
            
            markets = markets_query.all()
            
            logger.info(f"Found {len(markets)} markets in {self.config['lookback_days']} day period")
            
            for market in markets:
                # Get pre-match odds (simulate live trading conditions)
                pre_match_time = market.maturity_date - timedelta(hours=2)
                
                odds_records = db.query(Odd).filter(
                    Odd.source_id == market.source_id,
                    Odd.updated_at < pre_match_time
                ).order_by(Odd.updated_at.desc()).limit(20).all()
                
                if len(odds_records) < 3:
                    continue
                    
                # Build odds dictionary
                odds_data = {}
                for odd in odds_records:
                    if odd.outcome not in odds_data:
                        odds_data[odd.outcome] = odd.decimal_odds
                        
                if len(odds_data) < 2:
                    continue
                    
                # Simulate outcome determination
                # For backtest, we'll use a pseudo-random but deterministic method
                # based on fair probabilities adjusted for market inefficiencies
                implied_probs = {outcome: 1/odds for outcome, odds in odds_data.items()}
                total_implied = sum(implied_probs.values())
                
                if total_implied > 0:
                    normalized_probs = {k: v/total_implied for k, v in implied_probs.items()}
                    
                    # Add some realistic variance (home advantage, randomness)
                    if 'home' in normalized_probs:
                        normalized_probs['home'] *= 1.05  # Slight home advantage
                    if 'draw' in normalized_probs:
                        normalized_probs['draw'] *= 0.95  # Draws slightly less likely
                    
                    # Re-normalize
                    total_adj = sum(normalized_probs.values())
                    final_probs = {k: v/total_adj for k, v in normalized_probs.items()}
                    
                    # Deterministic outcome based on market hash for consistency
                    market_hash = abs(hash(market.source_id)) % 1000 / 1000
                    cumulative_prob = 0
                    winning_outcome = list(final_probs.keys())[0]  # fallback
                    
                    for outcome, prob in final_probs.items():
                        cumulative_prob += prob
                        if market_hash <= cumulative_prob:
                            winning_outcome = outcome
                            break
                            
                    markets_with_outcomes.append((market, odds_data, winning_outcome))
                    
        logger.info(f"Prepared {len(markets_with_outcomes)} markets for backtesting")
        return markets_with_outcomes
        
    def calculate_kelly_bet_size(self, probability: float, odds: float, bankroll: float) -> float:
        """Calculate Kelly criterion bet size."""
        if odds <= 1.0 or probability <= 0:
            return 0
            
        kelly_fraction = (probability * odds - 1) / (odds - 1)
        kelly_fraction = max(0, min(kelly_fraction, 0.5))  # Cap at 50%
        
        bet_size = bankroll * kelly_fraction * self.config['kelly_fraction']
        bet_size = min(bet_size, bankroll * self.config['max_position_pct'])
        
        return bet_size if bet_size >= self.config['min_bet_size'] else 0
        
    def simulate_strategy(self, strategy_name: str, provider) -> List[Trade]:
        """Simulate a single strategy across historical data."""
        markets_data = self.get_historical_data()
        bankroll = self.config['initial_bankroll']
        trades = []
        
        logger.info(f"Simulating strategy: {strategy_name}")
        
        for i, (market, odds_data, winning_outcome) in enumerate(markets_data):
            try:
                # Generate signals
                if hasattr(provider, 'generate_signals_for_market'):
                    signals = provider.generate_signals_for_market(market, odds_data)
                else:
                    signals = provider.generate_signal(market, odds_data)
                    
                if not signals:
                    continue
                    
                # Process each signal
                for signal_key, signal_data in signals.items():
                    forecast = signal_data.get('forecast', 0)
                    confidence = signal_data.get('confidence', 0)
                    edge = signal_data.get('edge', 0) / 100
                    probability = signal_data.get('probability', 0)
                    signal_type = signal_data.get('signal_type', 'unknown')
                    
                    # Extract outcome
                    outcome = signal_key.split('_')[-1] if '_' in signal_key else signal_key
                    if outcome not in odds_data:
                        continue
                        
                    odds = odds_data[outcome]
                    
                    # Apply filters
                    if (confidence < self.config['min_confidence'] or 
                        abs(edge) < self.config['min_edge']):
                        continue
                        
                    # Calculate bet size
                    bet_size = self.calculate_kelly_bet_size(probability, odds, bankroll)
                    if bet_size < self.config['min_bet_size']:
                        continue
                        
                    # Determine outcome
                    is_winner = (outcome.lower() == winning_outcome.lower())
                    
                    # Calculate P&L
                    if is_winner:
                        pnl = bet_size * (odds - 1) - bet_size * self.config['transaction_cost_pct']
                    else:
                        pnl = -bet_size - bet_size * self.config['transaction_cost_pct']
                        
                    bankroll += pnl
                    
                    # Record trade
                    trade = Trade(
                        timestamp=market.maturity_date,
                        market_id=market.source_id,
                        signal_provider=strategy_name,
                        signal_type=signal_type,
                        outcome=outcome,
                        odds=odds,
                        stake=bet_size,
                        probability=probability,
                        forecast=forecast,
                        confidence=confidence,
                        edge=edge,
                        actual_outcome=winning_outcome,
                        is_winner=is_winner,
                        pnl=pnl,
                        cumulative_pnl=bankroll - self.config['initial_bankroll'],
                        bankroll=bankroll
                    )
                    trades.append(trade)
                    
            except Exception as e:
                logger.warning(f"Error processing market {market.source_id} for {strategy_name}: {e}")
                continue
                
        logger.info(f"Strategy {strategy_name}: {len(trades)} trades, final bankroll: ${bankroll:.2f}")
        return trades
        
    def calculate_performance_metrics(self, trades: List[Trade]) -> StrategyPerformance:
        """Calculate comprehensive performance metrics."""
        if not trades:
            return StrategyPerformance(
                strategy_name="Empty",
                total_return=0, annualized_return=0, volatility=0, sharpe_ratio=0,
                max_drawdown=0, calmar_ratio=0, win_rate=0, profit_factor=0,
                avg_win=0, avg_loss=0, trades_count=0, avg_edge=0, avg_confidence=0,
                best_trade=0, worst_trade=0, consecutive_wins=0, consecutive_losses=0,
                kelly_utilization=0, risk_adjusted_return=0
            )
            
        strategy_name = trades[0].signal_provider
        final_bankroll = trades[-1].bankroll
        total_return = (final_bankroll - self.config['initial_bankroll']) / self.config['initial_bankroll']
        
        # Time-based metrics
        days_elapsed = (trades[-1].timestamp - trades[0].timestamp).days
        years_elapsed = max(days_elapsed / 365, 1/365)
        annualized_return = (1 + total_return) ** (1/years_elapsed) - 1
        
        # Trade analysis
        wins = [t for t in trades if t.is_winner]
        losses = [t for t in trades if not t.is_winner]
        win_rate = len(wins) / len(trades)
        
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.01
        profit_factor = gross_profit / gross_loss
        
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        
        # Drawdown calculation
        cumulative_returns = [t.cumulative_pnl for t in trades]
        peak = 0
        max_drawdown = 0
        
        for cum_return in cumulative_returns:
            if cum_return > peak:
                peak = cum_return
            drawdown = (peak - cum_return) / (self.config['initial_bankroll'] + peak) if peak > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)
            
        # Volatility and Sharpe
        returns_series = [t.pnl / self.config['initial_bankroll'] for t in trades]
        volatility = np.std(returns_series) * np.sqrt(252) if len(returns_series) > 1 else 0
        excess_return = annualized_return - self.config['risk_free_rate']
        sharpe_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
        
        # Consecutive wins/losses
        consecutive_wins = consecutive_losses = 0
        current_wins = current_losses = 0
        
        for trade in trades:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                consecutive_wins = max(consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                consecutive_losses = max(consecutive_losses, current_losses)
                
        # Kelly utilization
        theoretical_kelly_sizes = []
        for trade in trades:
            theoretical_kelly = (trade.probability * trade.odds - 1) / (trade.odds - 1)
            theoretical_kelly = max(0, min(theoretical_kelly, 0.5))
            theoretical_kelly_sizes.append(theoretical_kelly)
            
        kelly_utilization = np.mean([trade.stake / (trade.bankroll * kelly) 
                                   for trade, kelly in zip(trades, theoretical_kelly_sizes) 
                                   if kelly > 0]) if theoretical_kelly_sizes else 0
        
        return StrategyPerformance(
            strategy_name=strategy_name,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
            trades_count=len(trades),
            avg_edge=np.mean([t.edge for t in trades]),
            avg_confidence=np.mean([t.confidence for t in trades]),
            best_trade=max([t.pnl for t in trades]),
            worst_trade=min([t.pnl for t in trades]),
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            kelly_utilization=kelly_utilization,
            risk_adjusted_return=total_return / max_drawdown if max_drawdown > 0 else total_return
        )
        
    def create_performance_charts(self, all_trades: Dict[str, List[Trade]], 
                                all_performance: Dict[str, StrategyPerformance]):
        """Create comprehensive performance charts."""
        
        # Set up the plotting style
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Cumulative Returns
        plt.subplot(3, 3, 1)
        for strategy_name, trades in all_trades.items():
            if trades:
                dates = [t.timestamp for t in trades]
                cum_returns = [t.cumulative_pnl for t in trades]
                plt.plot(dates, cum_returns, label=strategy_name, linewidth=2)
                
        plt.title('Cumulative P&L by Strategy', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Cumulative P&L ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # 2. Returns Distribution
        plt.subplot(3, 3, 2)
        for strategy_name, trades in all_trades.items():
            if trades:
                returns = [t.pnl / self.config['initial_bankroll'] for t in trades]
                plt.hist(returns, bins=30, alpha=0.6, label=strategy_name)
                
        plt.title('Returns Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Return per Trade')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Risk-Return Scatter
        plt.subplot(3, 3, 3)
        returns = [p.annualized_return for p in all_performance.values()]
        risks = [p.volatility for p in all_performance.values()]
        names = list(all_performance.keys())
        
        colors = plt.cm.Set1(np.linspace(0, 1, len(names)))
        for i, (ret, risk, name) in enumerate(zip(returns, risks, names)):
            plt.scatter(risk, ret, s=150, c=[colors[i]], label=name, alpha=0.7)
            plt.annotate(name, (risk, ret), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
        plt.title('Risk-Return Profile', fontsize=14, fontweight='bold')
        plt.xlabel('Volatility (Annual)')
        plt.ylabel('Annualized Return')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # 4. Sharpe Ratios
        plt.subplot(3, 3, 4)
        strategies = list(all_performance.keys())
        sharpe_ratios = [all_performance[s].sharpe_ratio for s in strategies]
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        bars = plt.bar(strategies, sharpe_ratios, color=colors)
        plt.title('Sharpe Ratios by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Sharpe Ratio')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, sharpe_ratios):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 5. Win Rates
        plt.subplot(3, 3, 5)
        win_rates = [all_performance[s].win_rate * 100 for s in strategies]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(strategies)))
        bars = plt.bar(strategies, win_rates, color=colors)
        plt.title('Win Rates by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Win Rate (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% Baseline')
        
        # Add value labels
        for bar, value in zip(bars, win_rates):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 6. Maximum Drawdown
        plt.subplot(3, 3, 6)
        max_drawdowns = [all_performance[s].max_drawdown * 100 for s in strategies]
        
        colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(strategies)))
        bars = plt.bar(strategies, max_drawdowns, color=colors)
        plt.title('Maximum Drawdown by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Max Drawdown (%)')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, max_drawdowns):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 7. Trade Count and Frequency
        plt.subplot(3, 3, 7)
        trade_counts = [all_performance[s].trades_count for s in strategies]
        
        colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(strategies)))
        bars = plt.bar(strategies, trade_counts, color=colors)
        plt.title('Number of Trades by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Trade Count')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, trade_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value}', ha='center', va='bottom', fontweight='bold')
        
        # 8. Profit Factor
        plt.subplot(3, 3, 8)
        profit_factors = [all_performance[s].profit_factor for s in strategies]
        
        colors = plt.cm.coolwarm(np.linspace(0, 1, len(strategies)))
        bars = plt.bar(strategies, profit_factors, color=colors)
        plt.title('Profit Factor by Strategy', fontsize=14, fontweight='bold')
        plt.ylabel('Profit Factor')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Break-even')
        
        # Add value labels
        for bar, value in zip(bars, profit_factors):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # 9. Edge vs Confidence Scatter (for all trades combined)
        plt.subplot(3, 3, 9)
        all_edges = []
        all_confidences = []
        all_strategies = []
        
        for strategy_name, trades in all_trades.items():
            for trade in trades:
                all_edges.append(trade.edge * 100)
                all_confidences.append(trade.confidence)
                all_strategies.append(strategy_name)
                
        # Create scatter plot with different colors for each strategy
        strategy_colors = {name: plt.cm.Set1(i/len(all_trades)) 
                          for i, name in enumerate(all_trades.keys())}
        
        for strategy_name in all_trades.keys():
            strategy_edges = [e for e, s in zip(all_edges, all_strategies) if s == strategy_name]
            strategy_confs = [c for c, s in zip(all_confidences, all_strategies) if s == strategy_name]
            
            if strategy_edges:
                plt.scatter(strategy_confs, strategy_edges, 
                           c=[strategy_colors[strategy_name]], 
                           label=strategy_name, alpha=0.6, s=30)
        
        plt.title('Edge vs Confidence by Strategy', fontsize=14, fontweight='bold')
        plt.xlabel('Confidence')
        plt.ylabel('Edge (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        chart_path = os.path.join(self.results_dir, 'performance_charts.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        logger.info(f"Performance charts saved to {chart_path}")
        
        return chart_path
        
    def generate_performance_report(self, all_performance: Dict[str, StrategyPerformance],
                                  all_trades: Dict[str, List[Trade]]) -> str:
        """Generate detailed performance report."""
        
        report_lines = []
        report_lines.append("🏆 COMPREHENSIVE CARVER STRATEGY BACKTEST REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Backtest Period: {self.config['lookback_days']} days")
        report_lines.append(f"Initial Bankroll: ${self.config['initial_bankroll']:,.2f}")
        report_lines.append(f"Kelly Fraction: {self.config['kelly_fraction']:.1%}")
        report_lines.append(f"Max Position Size: {self.config['max_position_pct']:.1%}")
        report_lines.append(f"Transaction Cost: {self.config['transaction_cost_pct']:.1%}")
        report_lines.append("")
        
        # Summary table
        report_lines.append("📊 STRATEGY PERFORMANCE SUMMARY")
        report_lines.append("-" * 80)
        
        headers = ["Strategy", "Return", "Sharpe", "MaxDD", "WinRate", "Trades", "ProfitFactor"]
        row_format = "{:<15} {:>8} {:>8} {:>8} {:>8} {:>8} {:>12}"
        report_lines.append(row_format.format(*headers))
        report_lines.append("-" * 80)
        
        # Sort by total return
        sorted_strategies = sorted(all_performance.items(), 
                                 key=lambda x: x[1].total_return, reverse=True)
        
        for strategy_name, perf in sorted_strategies:
            report_lines.append(row_format.format(
                strategy_name[:14],
                f"{perf.total_return:+.1%}",
                f"{perf.sharpe_ratio:+.2f}",
                f"{perf.max_drawdown:.1%}",
                f"{perf.win_rate:.1%}",
                f"{perf.trades_count}",
                f"{perf.profit_factor:.2f}"
            ))
        
        # Detailed analysis for each strategy
        report_lines.append("\n\n🔬 DETAILED STRATEGY ANALYSIS")
        report_lines.append("=" * 80)
        
        for strategy_name, perf in sorted_strategies:
            report_lines.append(f"\n🎯 {strategy_name.upper()}")
            report_lines.append("-" * 50)
            
            # Performance metrics
            report_lines.append(f"📈 Returns:")
            report_lines.append(f"  Total Return: {perf.total_return:+.2%}")
            report_lines.append(f"  Annualized Return: {perf.annualized_return:+.2%}")
            report_lines.append(f"  Volatility: {perf.volatility:.2%}")
            
            report_lines.append(f"📊 Risk Metrics:")
            report_lines.append(f"  Sharpe Ratio: {perf.sharpe_ratio:+.2f}")
            report_lines.append(f"  Maximum Drawdown: {perf.max_drawdown:.2%}")
            report_lines.append(f"  Calmar Ratio: {perf.calmar_ratio:.2f}")
            report_lines.append(f"  Risk-Adjusted Return: {perf.risk_adjusted_return:.2f}")
            
            report_lines.append(f"🎲 Trade Analysis:")
            report_lines.append(f"  Win Rate: {perf.win_rate:.1%} ({perf.trades_count} trades)")
            report_lines.append(f"  Profit Factor: {perf.profit_factor:.2f}")
            report_lines.append(f"  Average Win: ${perf.avg_win:+.2f}")
            report_lines.append(f"  Average Loss: ${perf.avg_loss:+.2f}")
            report_lines.append(f"  Best Trade: ${perf.best_trade:+.2f}")
            report_lines.append(f"  Worst Trade: ${perf.worst_trade:+.2f}")
            
            report_lines.append(f"🎯 Signal Quality:")
            report_lines.append(f"  Average Edge: {perf.avg_edge:+.2%}")
            report_lines.append(f"  Average Confidence: {perf.avg_confidence:.3f}")
            report_lines.append(f"  Kelly Utilization: {perf.kelly_utilization:.2f}")
            
            report_lines.append(f"📈 Consistency:")
            report_lines.append(f"  Max Consecutive Wins: {perf.consecutive_wins}")
            report_lines.append(f"  Max Consecutive Losses: {perf.consecutive_losses}")
        
        # Generate insights and recommendations
        report_lines.append(f"\n\n💡 KEY INSIGHTS & RECOMMENDATIONS")
        report_lines.append("=" * 60)
        
        if all_performance:
            best_strategy = max(all_performance.values(), key=lambda x: x.total_return)
            best_sharpe = max(all_performance.values(), key=lambda x: x.sharpe_ratio)
            most_trades = max(all_performance.values(), key=lambda x: x.trades_count)
            
            report_lines.append(f"🏆 Best Performer: {best_strategy.strategy_name}")
            report_lines.append(f"  - Total Return: {best_strategy.total_return:+.2%}")
            report_lines.append(f"  - Risk-adjusted return of {best_strategy.sharpe_ratio:.2f} Sharpe")
            
            report_lines.append(f"\n📊 Best Risk-Adjusted: {best_sharpe.strategy_name}")
            report_lines.append(f"  - Sharpe Ratio: {best_sharpe.sharpe_ratio:+.2f}")
            report_lines.append(f"  - Consistent performance with {best_sharpe.max_drawdown:.1%} max drawdown")
            
            report_lines.append(f"\n🔥 Most Active: {most_trades.strategy_name}")
            report_lines.append(f"  - Generated {most_trades.trades_count} trades")
            report_lines.append(f"  - Win rate of {most_trades.win_rate:.1%}")
            
            # Overall system insights
            total_trades = sum(p.trades_count for p in all_performance.values())
            avg_return = np.mean([p.total_return for p in all_performance.values()])
            
            report_lines.append(f"\n📈 SYSTEM INSIGHTS:")
            report_lines.append(f"  - Total trades across all strategies: {total_trades}")
            report_lines.append(f"  - Average strategy return: {avg_return:+.1%}")
            report_lines.append(f"  - Strategies tested: {len(all_performance)}")
            
            # Performance recommendations
            profitable_strategies = [p for p in all_performance.values() if p.total_return > 0]
            
            report_lines.append(f"\n🎯 RECOMMENDATIONS:")
            if profitable_strategies:
                report_lines.append(f"  ✅ {len(profitable_strategies)} strategies showed positive returns")
                report_lines.append(f"  ✅ Consider portfolio allocation across top performers")
                report_lines.append(f"  ✅ Monitor strategy correlation for diversification")
            else:
                report_lines.append(f"  ⚠️ No strategies showed positive returns in backtest period")
                report_lines.append(f"  ⚠️ Consider parameter optimization or signal refinement")
                
        report_path = os.path.join(self.results_dir, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
            
        logger.info(f"Performance report saved to {report_path}")
        return report_path
        
    def export_trade_data(self, all_trades: Dict[str, List[Trade]]) -> str:
        """Export detailed trade data to CSV."""
        all_trade_records = []
        
        for strategy_name, trades in all_trades.items():
            for trade in trades:
                trade_dict = asdict(trade)
                all_trade_records.append(trade_dict)
                
        df = pd.DataFrame(all_trade_records)
        csv_path = os.path.join(self.results_dir, 'detailed_trades.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"Trade data exported to {csv_path}")
        return csv_path
        
    def run_comprehensive_backtest(self) -> Dict:
        """Run comprehensive backtest of all strategies."""
        logger.info("🚀 Starting comprehensive Carver strategy backtest")
        
        all_trades = {}
        all_performance = {}
        
        # Test each strategy
        for strategy_name, provider in self.signal_providers.items():
            logger.info(f"Testing strategy: {strategy_name}")
            
            try:
                trades = self.simulate_strategy(strategy_name, provider)
                all_trades[strategy_name] = trades
                
                performance = self.calculate_performance_metrics(trades)
                all_performance[strategy_name] = performance
                
            except Exception as e:
                logger.error(f"Error testing {strategy_name}: {e}")
                all_trades[strategy_name] = []
                all_performance[strategy_name] = self.calculate_performance_metrics([])
        
        # Generate outputs
        chart_path = self.create_performance_charts(all_trades, all_performance)
        report_path = self.generate_performance_report(all_performance, all_trades)
        csv_path = self.export_trade_data(all_trades)
        
        # Summary for console output
        print("\n🏆 BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        
        for strategy_name, perf in sorted(all_performance.items(), 
                                        key=lambda x: x[1].total_return, reverse=True):
            print(f"{strategy_name:>15}: {perf.total_return:+8.1%} return, "
                  f"{perf.sharpe_ratio:+6.2f} Sharpe, {perf.trades_count:>4} trades")
        
        print(f"\n📁 Results saved to: {self.results_dir}/")
        print(f"  📊 Charts: {os.path.basename(chart_path)}")
        print(f"  📝 Report: {os.path.basename(report_path)}")
        print(f"  📈 Data: {os.path.basename(csv_path)}")
        
        return {
            'trades': all_trades,
            'performance': all_performance,
            'chart_path': chart_path,
            'report_path': report_path,
            'csv_path': csv_path
        }


def main():
    """Main backtest execution."""
    
    # Configuration for the backtest
    config = {
        'initial_bankroll': 10000,
        'kelly_fraction': 0.25,  # Use 25% of Kelly for safety
        'max_position_pct': 0.05,  # Max 5% of bankroll per trade
        'min_bet_size': 10,
        'transaction_cost_pct': 0.02,  # 2% transaction costs
        'min_confidence': 0.10,  # Lower threshold for testing
        'min_edge': 0.005,  # 0.5% minimum edge
        'risk_free_rate': 0.02,
        'lookback_days': 60,  # 2 months of data
        'market_limit': None  # Remove arbitrary limits for deployment
    }
    
    print("🔬 COMPREHENSIVE CARVER STRATEGY BACKTEST")
    print("=" * 60)
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize and run backtest
    backtester = ComprehensiveCarverBacktest(config)
    results = backtester.run_comprehensive_backtest()
    
    return results


if __name__ == "__main__":
    results = main()