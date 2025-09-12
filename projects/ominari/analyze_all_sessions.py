#!/usr/bin/env python3
"""
Analyze all betting sessions to generate comprehensive performance statistics.
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def parse_session_report(filepath):
    """Parse a betting session report to extract metrics."""
    metrics = {}
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Extract date from filepath
        date_str = os.path.basename(os.path.dirname(filepath))
        try:
            metrics['date'] = datetime.strptime(date_str[:10], '%Y-%m-%d')
        except:
            metrics['date'] = None
            
        # Extract metrics using regex
        patterns = {
            'sharpe': r'Sharpe[:\s]+([0-9.-]+)',
            'volatility': r'Volatility[:\s]+([0-9.]+)',
            'expected_return': r'Expected Return[:\s\(log\)]+([0-9.-]+)',
            'expected_multiplier': r'Expected Multiplier[:\s]+([0-9.]+)x',
            'total_stake': r'Total Stake[:\s]+([0-9.]+)',
            'num_bets': r'Bets to Place:.*\n((?:[^\n]+\n)+)',  # Count rows
            'kelly_fraction': r'Kelly fraction[:\s]+([0-9.]+)',
            'min_bet': r'Min bets[:\s]+([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                if key == 'num_bets':
                    # Count betting rows
                    bet_lines = match.group(1).strip().split('\n')
                    metrics[key] = len([l for l in bet_lines if l.strip() and not l.startswith(' ')])
                else:
                    metrics[key] = float(match.group(1))
            else:
                metrics[key] = None
                
    except Exception as e:
        print(f"Error parsing {filepath}: {e}")
        
    return metrics

def analyze_all_sessions():
    """Analyze all betting sessions and generate statistics."""
    
    print("🎲 Analyzing all betting sessions...")
    
    # Find all session reports
    session_files = glob.glob('betting_reports/*/betting_session_report.md')
    print(f"Found {len(session_files)} sessions")
    
    # Parse all sessions
    sessions = []
    for filepath in session_files:
        metrics = parse_session_report(filepath)
        if metrics.get('sharpe') is not None:  # Only include valid sessions
            sessions.append(metrics)
    
    print(f"Successfully parsed {len(sessions)} sessions")
    
    if not sessions:
        print("No valid sessions found")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(sessions)
    
    # Filter by date if available
    if 'date' in df.columns:
        df = df[df['date'].notna()]
        df = df.sort_values('date')
    
    # Calculate statistics for different windows
    windows = {
        '7 days': 7,
        '30 days': 30,
        '90 days': 90,
        '365 days': 365,
        'All time': 99999
    }
    
    results = {}
    
    for window_name, days in windows.items():
        cutoff = datetime.now() - timedelta(days=days)
        window_df = df[df['date'] >= cutoff] if 'date' in df.columns else df
        
        if len(window_df) > 0:
            stats = {
                'sessions': len(window_df),
                'sharpe_mean': window_df['sharpe'].mean(),
                'sharpe_std': window_df['sharpe'].std(),
                'sharpe_min': window_df['sharpe'].min(),
                'sharpe_max': window_df['sharpe'].max(),
                'sharpe_median': window_df['sharpe'].median(),
                'volatility_mean': window_df['volatility'].mean() if 'volatility' in window_df else None,
                'return_mean': window_df['expected_return'].mean() if 'expected_return' in window_df else None,
                'stake_total': window_df['total_stake'].sum() if 'total_stake' in window_df else None,
                'stake_mean': window_df['total_stake'].mean() if 'total_stake' in window_df else None,
                'bets_total': window_df['num_bets'].sum() if 'num_bets' in window_df else None,
            }
            
            # Calculate annualized metrics
            if stats['return_mean'] and stats['sessions'] > 0 and stats['stake_mean']:
                # The return_mean is the expected log return on staked amount
                # We need to adjust for the actual bankroll usage
                # Average stake per session / bankroll = fraction of bankroll at risk
                bankroll = 100  # Execution bankroll
                stake_fraction = stats['stake_mean'] / bankroll
                
                # Actual return per session on bankroll
                # If we stake 15% of bankroll and get 14% return on that stake,
                # then return on bankroll = 0.15 * 0.14 = 0.021 (2.1%)
                expected_multiplier = np.exp(stats['return_mean'])  # Convert log return to multiplier
                profit_per_session = stake_fraction * (expected_multiplier - 1)
                
                # Calculate daily and annualized returns
                # 6 sessions per day (24h / 4h)
                sessions_per_day = 6
                sessions_per_year = sessions_per_day * 365
                
                # Daily return: compound 6 sessions
                stats['return_daily'] = (1 + profit_per_session) ** sessions_per_day - 1
                
                # Annualized return: compound 2190 sessions
                stats['return_annualized'] = (1 + profit_per_session) ** sessions_per_year - 1
                
            results[window_name] = stats
    
    # Print results
    print("\n📊 PERFORMANCE STATISTICS BY TIME WINDOW")
    print("=" * 80)
    
    for window_name, stats in results.items():
        print(f"\n{window_name} ({stats['sessions']} sessions)")
        print("-" * 40)
        print(f"Sharpe Ratio: {stats['sharpe_mean']:.3f} (±{stats['sharpe_std']:.3f})")
        print(f"  Range: {stats['sharpe_min']:.3f} to {stats['sharpe_max']:.3f}")
        print(f"  Median: {stats['sharpe_median']:.3f}")
        
        if stats['volatility_mean']:
            print(f"Volatility: {stats['volatility_mean']*100:.2f}%")
            
        if stats['return_mean'] and stats['stake_mean']:
            print(f"Expected Return (log): {stats['return_mean']:.4f} per session")
            expected_multiplier = np.exp(stats['return_mean'])
            print(f"  Expected Multiplier: {expected_multiplier:.3f}x on staked amount")
            stake_fraction = stats['stake_mean'] / 100
            profit_per_session = stake_fraction * (expected_multiplier - 1)
            print(f"  Stake per Session: ${stats['stake_mean']:.2f} ({stake_fraction*100:.1f}% of bankroll)")
            print(f"  Profit per Session: ${profit_per_session*100:.2f} ({profit_per_session*100:.2f}% of bankroll)")
            if 'return_daily' in stats:
                print(f"  Daily Return: {stats['return_daily']*100:.2f}%")
            if 'return_annualized' in stats:
                print(f"  Annualized: {stats['return_annualized']*100:.1f}%")
                
        if stats['stake_mean']:
            print(f"Average Stake: ${stats['stake_mean']:.2f}")
            print(f"Total Staked: ${stats['stake_total']:,.2f}")
            
        if stats['bets_total']:
            print(f"Total Bets: {stats['bets_total']:,}")
    
    # Save detailed results
    output_file = 'performance_analysis.json'
    with open(output_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for window, stats in results.items():
            json_results[window] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for k, v in stats.items() if v is not None}
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    # Calculate rolling statistics
    if 'date' in df.columns and len(df) > 30:
        print("\n📈 ROLLING 30-DAY STATISTICS")
        print("=" * 80)
        
        # Sort by date
        df = df.sort_values('date')
        
        # Calculate rolling metrics
        df['sharpe_30d'] = df['sharpe'].rolling(window=30, min_periods=10).mean()
        df['volatility_30d'] = df['volatility'].rolling(window=30, min_periods=10).mean()
        
        # Show recent trend
        recent = df.tail(10)
        print("\nRecent 10 sessions:")
        for _, row in recent.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d') if pd.notna(row['date']) else 'Unknown'
            sharpe_30d = row['sharpe_30d'] if pd.notna(row['sharpe_30d']) else row['sharpe']
            print(f"{date_str}: Sharpe={row['sharpe']:.2f} (30d avg: {sharpe_30d:.2f})")

if __name__ == "__main__":
    analyze_all_sessions()