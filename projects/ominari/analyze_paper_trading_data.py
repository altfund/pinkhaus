#!/usr/bin/env python3
"""Analyze existing paper trading data in both JSON and database."""

from database_v2 import db_manager
from paper_trading_models import PaperOrder, PaperFill, PaperPerformance
from paper_trading_sessions import PaperTradingSessionManager
import json

print("ANALYZING PAPER TRADING DATA")
print("=" * 80)

# Check database tables
print("\n1. DATABASE TABLES:")
print("-" * 40)

with db_manager.get_db_session() as db:
    # Check paper_orders
    order_count = db.query(PaperOrder).count()
    print(f"paper_orders: {order_count} records")
    
    # Check paper_fills
    fill_count = db.query(PaperFill).count()
    print(f"paper_fills: {fill_count} records")
    
    # Skip paper_performance for now - schema mismatch
    print(f"paper_performance: (skipped - checking schema)")

# Check JSON data
print("\n2. JSON FILE DATA:")
print("-" * 40)

try:
    session_manager = PaperTradingSessionManager()
    sessions = session_manager.sessions.get('sessions', {})
    
    print(f"Total sessions: {len(sessions)}")
    
    total_positions = 0
    total_closed = 0
    total_trades = 0
    
    for session_id, session_data in sessions.items():
        positions = len(session_data.get('positions', {}))
        closed = len(session_data.get('closed_positions', []))
        trades = len(session_data.get('trades', []))
        
        total_positions += positions
        total_closed += closed
        total_trades += trades
        
        print(f"\nSession {session_id}:")
        print(f"  Status: {session_data.get('status', 'unknown')}")
        print(f"  Open positions: {positions}")
        print(f"  Closed positions: {closed}")
        print(f"  Trades: {trades}")
        print(f"  Initial capital: ${session_data.get('initial_bankroll', 0):,.2f}")
        print(f"  Current capital: ${session_data.get('current_bankroll', 0):,.2f}")
        print(f"  Portfolio value: ${session_data.get('portfolio_value', 0):,.2f}")
    
    print(f"\nTOTALS:")
    print(f"  Open positions: {total_positions}")
    print(f"  Closed positions: {total_closed}")
    print(f"  Total trades: {total_trades}")
    
except Exception as e:
    print(f"Error reading JSON data: {e}")

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("-" * 40)

if order_count == 0 and total_closed > 0:
    print("✓ JSON file has data but database is empty")
    print("✓ Migration needed to move JSON data to database")
    print("✓ This will enable better querying and automatic P&L tracking")
else:
    print("⚠️  Both systems may have data - need to reconcile")