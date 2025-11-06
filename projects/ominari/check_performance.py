#!/usr/bin/env python3
"""Check current trading performance"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from datetime import datetime

session_manager = PaperTradingSessionManager()
session_id = session_manager.get_current_session()

# Get session info
session = session_manager.get_session(session_id)
print(f'📊 CURRENT PERFORMANCE SUMMARY')
print('=' * 50)
print(f'Session ID: {session_id}')
print(f'Started: {session["created_at"]}')
print(f'Initial Bankroll: ${session["initial_bankroll"]:,.2f}')
print(f'Current Bankroll: ${session["current_bankroll"]:,.2f}')
print()

# Get performance metrics
try:
    perf = session_manager.get_enhanced_performance_analytics(session_id)
    if perf and 'overview' in perf:
        print('📈 PERFORMANCE METRICS:')
        print(f'Total Trades: {perf["overview"]["total_trades"]}')
        print(f'Win Rate: {perf["overview"]["win_rate"]:.1%}')
        print(f'ROI: {perf["financial"]["roi"]:.2%}')
        print(f'Profit Factor: {perf["financial"]["profit_factor"]:.2f}')
        print(f'Sharpe Ratio: {perf["financial"]["sharpe_ratio"]:.2f}')
        print(f'Max Drawdown: {perf["financial"]["max_drawdown"]:.1%}')
        print()
        
        if 'signal_performance' in perf:
            print('🎯 SIGNAL PERFORMANCE:')
            for sig in perf['signal_performance']:
                print(f'{sig["signal_name"]}: {sig["count"]} trades, {sig["win_rate"]:.1%} win rate, {sig["total_profit"]:.2f} profit')
except:
    # Fallback to basic metrics
    metrics = session_manager.get_performance_metrics(session_id)
    if metrics:
        print('📈 PERFORMANCE METRICS:')
        print(f'Total Bets: {metrics.get("total_bets", 0)}')
        print(f'Win Rate: {metrics.get("win_rate", 0):.1%}')
        print(f'ROI: {metrics.get("roi", 0):.2%}')

# Get current positions
positions = session_manager.get_positions(session_id)
open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
closed_positions = [p for p in positions if p['status'] == 'closed']

print(f'\n📋 POSITIONS:')
print(f'Open Positions: {len(open_positions)}')
print(f'Closed Positions: {len(closed_positions)}')

if open_positions:
    total_exposure = sum(float(p['stake']) for p in open_positions)
    print(f'Current Exposure: ${total_exposure:.2f} ({total_exposure/session["current_bankroll"]*100:.1%})')
    print('\n🎲 CURRENT OPEN POSITIONS:')
    for pos in open_positions[:5]:  # Show first 5
        market_id = pos.get("market_id", pos.get("match_id", "Unknown"))
        bet_type = pos.get("bet_type", pos.get("bet_on", "Unknown"))
        print(f'  • {market_id}: {bet_type} @ {pos["odds"]} - ${pos["stake"]:.2f}')
    if len(open_positions) > 5:
        print(f'  ... and {len(open_positions) - 5} more positions')

# Show recent wins/losses
if closed_positions:
    recent_closed = sorted(closed_positions, key=lambda x: x['updated_at'], reverse=True)[:5]
    print('\n💰 RECENT RESULTS:')
    for pos in recent_closed:
        pnl = float(pos.get('profit', 0))
        result = '✅ WIN' if pnl > 0 else '❌ LOSS'
        market_id = pos.get("market_id", pos.get("match_id", "Unknown"))
        print(f'  {result}: {market_id} - ${pnl:+.2f}')

# Calculate P&L
total_pnl = session['current_bankroll'] - session['initial_bankroll']
pnl_pct = (total_pnl / session['initial_bankroll']) * 100 if session['initial_bankroll'] > 0 else 0

print(f'\n💸 PROFIT/LOSS:')
print(f'Total P&L: ${total_pnl:+,.2f} ({pnl_pct:+.1%})')