#!/usr/bin/env python3
"""
Analyze betting sessions with proper cash flow accounting.
Shows the difference between theoretical Kelly returns and actual cash returns.
"""

import os
import re
import glob
import json
import numpy as np
import pandas as pd
from datetime import datetime

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

def analyze_cash_flow():
    """Analyze betting sessions with proper cash flow accounting."""
    
    print("💰 Cash Flow Analysis for Betting Sessions")
    print("=" * 80)
    
    # Key parameters
    KELLY_BANKROLL = 1.0      # Theoretical Kelly bankroll
    EXECUTION_BANKROLL = 100  # Actual cash bankroll
    SESSIONS_PER_DAY = 6      # 4-hour sessions
    TRADING_DAYS_PER_YEAR = 365  # Sports betting runs every day
    
    print("\nParameters:")
    print(f"  • Kelly Bankroll: ${KELLY_BANKROLL}")
    print(f"  • Execution Bankroll: ${EXECUTION_BANKROLL}")
    print(f"  • Sessions per Day: {SESSIONS_PER_DAY}")
    print(f"  • Trading Days per Year: {TRADING_DAYS_PER_YEAR}")
    
    # Find all session reports
    session_files = glob.glob('betting_reports/*/betting_session_report.md')
    print(f"\nFound {len(session_files)} total sessions")
    
    # Parse sessions
    sessions = []
    for filepath in session_files[:20]:  # Analyze first 20 sessions
        metrics = parse_session_report(filepath)
        if metrics.get('expected_return') is not None and metrics.get('total_stake') is not None:
            sessions.append(metrics)
    
    print(f"Successfully parsed {len(sessions)} sessions with complete data")
    
    if not sessions:
        print("No valid sessions found")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(sessions)
    
    # Calculate key metrics
    print("\n📊 ACTUAL vs THEORETICAL RETURNS")
    print("-" * 80)
    
    # Average metrics
    avg_stake = df['total_stake'].mean()
    avg_log_return = df['expected_return'].mean()
    avg_sharpe = df['sharpe'].mean()
    avg_volatility = df['volatility'].mean() if 'volatility' in df else 0.03
    
    # Calculate returns properly
    print("\n1. Per-Session Analysis:")
    print(f"   Average Stake: ${avg_stake:.2f}")
    print(f"   Stake as % of Bankroll: {avg_stake/EXECUTION_BANKROLL*100:.1f}%")
    print(f"   Expected Log Return: {avg_log_return:.4f} (on staked amount)")
    print(f"   Expected Multiplier: {np.exp(avg_log_return):.3f}x")
    
    # Theoretical profit on staked amount
    expected_profit_on_stake = avg_stake * (np.exp(avg_log_return) - 1)
    print("\n2. Expected Profit per Session:")
    print(f"   On Staked Amount: ${expected_profit_on_stake:.2f}")
    print(f"   As % of Bankroll: {expected_profit_on_stake/EXECUTION_BANKROLL*100:.2f}%")
    
    # Daily and annual compounding
    profit_rate_per_session = expected_profit_on_stake / EXECUTION_BANKROLL
    
    # Simple interest approach (more realistic for cash flow)
    daily_profit_simple = profit_rate_per_session * SESSIONS_PER_DAY
    annual_profit_simple = daily_profit_simple * TRADING_DAYS_PER_YEAR
    
    # Compound interest approach (theoretical maximum)
    daily_return_compound = (1 + profit_rate_per_session) ** SESSIONS_PER_DAY - 1
    annual_return_compound = (1 + profit_rate_per_session) ** (SESSIONS_PER_DAY * TRADING_DAYS_PER_YEAR) - 1
    
    print("\n3. Daily Returns:")
    print(f"   Simple (Linear): {daily_profit_simple*100:.2f}%")
    print(f"   Compound (Geometric): {daily_return_compound*100:.2f}%")
    
    print("\n4. Annual Returns:")
    print(f"   Simple (Linear): {annual_profit_simple*100:.1f}%")
    print(f"   Compound (Geometric): {min(annual_return_compound, 1000)*100:.1f}%{'+ (capped)' if annual_return_compound > 1000 else ''}")
    
    # Risk-adjusted returns
    print("\n5. Risk-Adjusted Metrics:")
    print(f"   Average Sharpe Ratio: {avg_sharpe:.2f}")
    print(f"   Average Volatility: {avg_volatility*100:.1f}%")
    
    # Adjusted Sharpe for actual bankroll returns
    actual_return_volatility = avg_volatility * (avg_stake / EXECUTION_BANKROLL)
    actual_sharpe = profit_rate_per_session / actual_return_volatility if actual_return_volatility > 0 else 0
    print(f"   Bankroll-Adjusted Sharpe: {actual_sharpe:.2f}")
    
    # Cash flow considerations
    print("\n6. Cash Flow Reality Check:")
    print("   ❌ Theoretical assumes: All profits reinvested immediately")
    print("   ✅ Reality includes:")
    print("      • Settlement delays (24-48 hours)")
    print("      • Withdrawal requirements")
    print("      • Platform limits")
    print("      • Bankroll management rules")
    print(f"   → Realistic annual return: {annual_profit_simple*100:.0f}%-{min(annual_return_compound*0.5, 10)*100:.0f}%")
    
    # Save analysis
    results = {
        'parameters': {
            'kelly_bankroll': KELLY_BANKROLL,
            'execution_bankroll': EXECUTION_BANKROLL,
            'sessions_per_day': SESSIONS_PER_DAY,
            'trading_days_per_year': TRADING_DAYS_PER_YEAR
        },
        'per_session': {
            'avg_stake': float(avg_stake),
            'stake_percentage': float(avg_stake/EXECUTION_BANKROLL),
            'expected_log_return': float(avg_log_return),
            'expected_multiplier': float(np.exp(avg_log_return)),
            'expected_profit': float(expected_profit_on_stake),
            'profit_percentage': float(profit_rate_per_session)
        },
        'daily': {
            'simple_return': float(daily_profit_simple),
            'compound_return': float(daily_return_compound)
        },
        'annual': {
            'simple_return': float(annual_profit_simple),
            'compound_return': float(min(annual_return_compound, 1000)),
            'realistic_range_low': float(annual_profit_simple),
            'realistic_range_high': float(min(annual_return_compound*0.5, 10))
        },
        'risk_metrics': {
            'avg_sharpe': float(avg_sharpe),
            'avg_volatility': float(avg_volatility),
            'bankroll_adjusted_sharpe': float(actual_sharpe)
        }
    }
    
    output_file = 'cash_flow_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")

if __name__ == "__main__":
    analyze_cash_flow()