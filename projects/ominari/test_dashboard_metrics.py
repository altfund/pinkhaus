#!/usr/bin/env python3
"""Test that dashboard metrics are correctly showing the updated values."""

import requests
from paper_trading_sessions import PaperTradingSessionManager

def test_dashboard_metrics():
    """Check if dashboard APIs return correct metrics."""
    base_url = "http://localhost:5001/api"
    
    # Get session data directly
    session_manager = PaperTradingSessionManager()
    current_session = session_manager.get_current_session()
    performance = session_manager.get_session_performance(current_session['session_id'])
    
    print("=== Session Data ===")
    print(f"Session ID: {current_session['session_id']}")
    print(f"Portfolio Value: ${current_session['portfolio_value']:.2f}")
    print(f"Cash Available: ${current_session['current_bankroll']:.2f}")
    print(f"Total P&L: ${performance['total_pnl']:.2f}")
    print(f"Win Rate: {performance['win_rate']:.1%}")
    print(f"Winning Trades: {performance['winning_trades']}")
    print(f"Losing Trades: {performance['losing_trades']}")
    print()
    
    # Test portfolio endpoint
    try:
        resp = requests.get(f"{base_url}/trading/portfolio")
        if resp.status_code == 200:
            data = resp.json()
            print("=== Portfolio API ===")
            print(f"Total Value: ${data.get('total_value', 0):.2f}")
            print(f"Cash Available: ${data.get('cash_available', 0):.2f}")
            print(f"Daily Change: ${data.get('daily_change', 0):.2f} ({data.get('daily_change_pct', 0):.1f}%)")
            print(f"Positions Count: {data.get('positions_count', 0)}")
            print()
            
            # Verify values match
            if abs(data['total_value'] - current_session['portfolio_value']) < 0.01:
                print("✓ Portfolio value matches!")
            else:
                print(f"✗ Portfolio value mismatch: API ${data['total_value']:.2f} vs Session ${current_session['portfolio_value']:.2f}")
    except Exception as e:
        print(f"Error testing portfolio API: {e}")
    
    # Test recent trades endpoint
    try:
        resp = requests.get(f"{base_url}/trading/recent")
        if resp.status_code == 200:
            data = resp.json()
            metrics = data.get('metrics', {})
            print("=== Trading Metrics API ===")
            print(f"Today P&L: ${metrics.get('today_pnl', 0):.2f}")
            print(f"Win Rate: {metrics.get('win_rate', 0):.1%}")
            print(f"Total Trades: {metrics.get('total_trades', 0)}")
            print(f"Active Positions: {metrics.get('active_positions', 0)}")
            print(f"Largest Win: ${metrics.get('largest_win', 0):.2f}")
            print(f"Largest Loss: ${metrics.get('largest_loss', 0):.2f}")
            print(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print()
            
            # Verify key metrics
            if abs(metrics['win_rate'] - performance['win_rate']) < 0.001:
                print("✓ Win rate matches!")
            else:
                print(f"✗ Win rate mismatch: API {metrics['win_rate']:.1%} vs Session {performance['win_rate']:.1%}")
    except Exception as e:
        print(f"Error testing recent trades API: {e}")

if __name__ == "__main__":
    test_dashboard_metrics()