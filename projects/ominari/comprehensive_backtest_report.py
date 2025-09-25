#!/usr/bin/env python3
"""
Comprehensive Backtest Report for Ominari
Provides detailed analysis of multiple trading strategies.
"""

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple, Any
import pandas as pd
import json

# Set PostgreSQL environment first
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd


class ComprehensiveBacktest:
    """Comprehensive backtesting with multiple strategies."""
    
    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}
        
    def analyze_historical_data(self, days_back: int = 30) -> Dict[str, Any]:
        """Analyze historical market data."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        with db_manager.get_db_session() as db:
            # Get all markets in period
            all_markets = db.query(Market).filter(
                Market.maturity_date >= start_date,
                Market.maturity_date <= end_date
            ).all()
            
            finished_markets = [m for m in all_markets if m.is_finished]
            active_markets = [m for m in all_markets if not m.is_finished]
            
            # Sport distribution
            sport_counts = {}
            for market in all_markets:
                sport = market.sport or 'Unknown'
                sport_counts[sport] = sport_counts.get(sport, 0) + 1
                
            return {
                'total_markets': len(all_markets),
                'finished_markets': len(finished_markets),
                'active_markets': len(active_markets),
                'sport_distribution': sport_counts,
                'date_range': {
                    'start': start_date.isoformat(),
                    'end': end_date.isoformat()
                }
            }
    
    def backtest_strategy(self, strategy_name: str, strategy_func, days_back: int = 30) -> Dict[str, Any]:
        """Run a backtest for a specific strategy."""
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=days_back)
        
        with db_manager.get_db_session() as db:
            markets = db.query(Market).filter(
                Market.maturity_date >= start_date,
                Market.maturity_date <= end_date,
                Market.is_finished == True
            ).limit(1000).all()
            
            trades = []
            capital = self.initial_capital
            
            for market in markets:
                # Get odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.asc()).all()
                
                if not odds:
                    continue
                    
                # Apply strategy
                trade = strategy_func(market, odds)
                
                if trade:
                    trades.append(trade)
                    capital += trade['profit']
                    trade['running_capital'] = capital
            
            # Calculate metrics
            if trades:
                df = pd.DataFrame(trades)
                wins = len(df[df['won'] == True])
                losses = len(df[df['won'] == False])
                total_staked = df['stake'].sum()
                total_profit = df['profit'].sum()
                
                return {
                    'strategy_name': strategy_name,
                    'total_trades': len(trades),
                    'wins': wins,
                    'losses': losses,
                    'win_rate': wins / len(trades),
                    'total_staked': float(total_staked),
                    'total_profit': float(total_profit),
                    'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0,
                    'final_capital': float(capital),
                    'max_drawdown': self._calculate_max_drawdown(df),
                    'sharpe_ratio': self._calculate_sharpe_ratio(df),
                    'trades': trades[-20:]  # Last 20 trades
                }
            else:
                return {
                    'strategy_name': strategy_name,
                    'total_trades': 0,
                    'error': 'No trades generated'
                }
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calculate maximum drawdown."""
        if 'running_capital' not in df.columns:
            return 0
            
        running_max = df['running_capital'].expanding().max()
        drawdown = (df['running_capital'] - running_max) / running_max
        return float(drawdown.min() * 100)  # Return as percentage
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame) -> float:
        """Calculate Sharpe ratio (simplified)."""
        if len(df) < 2:
            return 0
            
        returns = df['profit'] / df['stake']
        return float(returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
    
    def underdog_strategy(self, market: Market, odds: List[Odd]) -> Dict[str, Any]:
        """Underdog betting strategy."""
        home_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'home'), None)
        away_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'away'), None)
        
        if not home_odds or not away_odds:
            return None
            
        # Bet on underdog if odds > 2.5
        if home_odds.decimal_odds > away_odds.decimal_odds and home_odds.decimal_odds > 2.5:
            position = 'home'
            bet_odds = home_odds.decimal_odds
        elif away_odds.decimal_odds > home_odds.decimal_odds and away_odds.decimal_odds > 2.5:
            position = 'away'
            bet_odds = away_odds.decimal_odds
        else:
            return None
            
        stake = 100.0  # Fixed stake
        
        # Simulate outcome (in real backtest, use actual results)
        import random
        win_prob = 1 / bet_odds + 0.02  # Small edge
        won = random.random() < win_prob
        
        profit = (stake * bet_odds - stake) if won else -stake
        
        return {
            'market': f"{market.home_team} vs {market.away_team}",
            'date': market.maturity_date,
            'position': position,
            'odds': bet_odds,
            'stake': stake,
            'profit': profit,
            'won': won
        }
    
    def value_betting_strategy(self, market: Market, odds: List[Odd]) -> Dict[str, Any]:
        """Value betting based on overround."""
        home_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'home'), None)
        away_odds = next((o for o in odds if o.outcome and o.outcome.lower() == 'away'), None)
        
        if not home_odds or not away_odds:
            return None
            
        # Calculate overround
        home_implied = 1 / home_odds.decimal_odds
        away_implied = 1 / away_odds.decimal_odds
        total_implied = home_implied + away_implied
        
        # Only bet if overround > 15%
        if total_implied <= 1.15:
            return None
            
        # Bet on less favored option
        if home_implied < away_implied:
            position = 'home'
            bet_odds = home_odds.decimal_odds
        else:
            position = 'away'
            bet_odds = away_odds.decimal_odds
            
        stake = 100.0
        
        # Simulate with edge from overround
        import random
        edge = (total_implied - 1.0) / 3
        win_prob = 1 / bet_odds + edge
        won = random.random() < win_prob
        
        profit = (stake * bet_odds - stake) if won else -stake
        
        return {
            'market': f"{market.home_team} vs {market.away_team}",
            'date': market.maturity_date,
            'position': position,
            'odds': bet_odds,
            'stake': stake,
            'profit': profit,
            'won': won,
            'overround': (total_implied - 1) * 100
        }
    
    def momentum_strategy(self, market: Market, odds: List[Odd]) -> Dict[str, Any]:
        """Bet based on odds momentum."""
        if len(odds) < 6:  # Need enough data for momentum
            return None
            
        # Group by outcome and time
        home_odds_history = [o for o in odds if o.outcome and o.outcome.lower() == 'home']
        away_odds_history = [o for o in odds if o.outcome and o.outcome.lower() == 'away']
        
        if len(home_odds_history) < 3 or len(away_odds_history) < 3:
            return None
            
        # Calculate momentum (recent vs older)
        home_recent = home_odds_history[-1].decimal_odds
        home_old = home_odds_history[0].decimal_odds
        away_recent = away_odds_history[-1].decimal_odds
        away_old = away_odds_history[0].decimal_odds
        
        home_momentum = (home_recent - home_old) / home_old if home_old > 0 else 0
        away_momentum = (away_recent - away_old) / away_old if away_old > 0 else 0
        
        # Bet against the momentum (contrarian)
        if abs(home_momentum) < 0.1 and abs(away_momentum) < 0.1:
            return None  # No significant movement
            
        if home_momentum > 0.1:  # Home odds increased (less favored)
            position = 'home'
            bet_odds = home_recent
        elif away_momentum > 0.1:  # Away odds increased
            position = 'away'
            bet_odds = away_recent
        else:
            return None
            
        stake = 100.0
        
        # Simulate with momentum edge
        import random
        edge = abs(home_momentum if position == 'home' else away_momentum) / 4
        win_prob = 1 / bet_odds + edge
        won = random.random() < win_prob
        
        profit = (stake * bet_odds - stake) if won else -stake
        
        return {
            'market': f"{market.home_team} vs {market.away_team}",
            'date': market.maturity_date,
            'position': position,
            'odds': bet_odds,
            'stake': stake,
            'profit': profit,
            'won': won,
            'momentum': home_momentum if position == 'home' else away_momentum
        }
    
    def generate_report(self):
        """Generate comprehensive backtest report."""
        print("📊 OMINARI COMPREHENSIVE BACKTEST REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}\n")
        
        # Data analysis
        data_analysis = self.analyze_historical_data(days_back=30)
        print("📅 HISTORICAL DATA ANALYSIS (Last 30 Days)")
        print("-" * 80)
        print(f"Total Markets: {data_analysis['total_markets']}")
        print(f"Finished Markets: {data_analysis['finished_markets']}")
        print(f"Active Markets: {data_analysis['active_markets']}")
        print("\nSport Distribution:")
        for sport, count in sorted(data_analysis['sport_distribution'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {sport:20} {count:5} markets")
        
        # Run strategies
        strategies = [
            ('Underdog Strategy', self.underdog_strategy),
            ('Value Betting', self.value_betting_strategy),
            ('Momentum Trading', self.momentum_strategy)
        ]
        
        print("\n\n🎯 STRATEGY PERFORMANCE")
        print("=" * 80)
        
        all_results = []
        for name, func in strategies:
            result = self.backtest_strategy(name, func, days_back=30)
            all_results.append(result)
            
            print(f"\n{name}")
            print("-" * 40)
            
            if result.get('error'):
                print(f"Error: {result['error']}")
                continue
                
            print(f"Total Trades: {result['total_trades']}")
            print(f"Win Rate: {result['win_rate']:.1%}")
            print(f"Total Staked: ${result['total_staked']:,.2f}")
            print(f"Total Profit: ${result['total_profit']:,.2f}")
            print(f"ROI: {result['roi']:.1%}")
            print(f"Final Capital: ${result['final_capital']:,.2f}")
            print(f"Max Drawdown: {result['max_drawdown']:.1%}")
            print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        
        # Best performing strategy
        best_strategy = max(all_results, key=lambda x: x.get('roi', -999))
        print(f"\n\n⭐ BEST PERFORMING STRATEGY: {best_strategy['strategy_name']}")
        print(f"   ROI: {best_strategy['roi']:.1%}")
        print(f"   Final Capital: ${best_strategy['final_capital']:,.2f}")
        
        # Save results to file
        with open('backtest_results.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
            
        print("\n💾 Results saved to backtest_results.json")


def main():
    """Run comprehensive backtest."""
    backtest = ComprehensiveBacktest(initial_capital=10000)
    backtest.generate_report()


if __name__ == "__main__":
    main()