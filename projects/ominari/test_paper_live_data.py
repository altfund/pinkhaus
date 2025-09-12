#!/usr/bin/env python3
"""Test paper trading with live market data."""

from database_v2 import db_manager
from models import Market, Odd
from paper_trading_engine import PaperTradingEngine, PaperOrder
from datetime import datetime, timezone
import json
import asyncio

async def test_paper_trading_with_live_data():
    """Test paper trading using real market data."""
    
    # Initialize paper trading engine
    engine = PaperTradingEngine(initial_capital=10000)
    
    print("=== PAPER TRADING TEST WITH LIVE DATA ===")
    print(f"Starting capital: ${engine.current_capital:,.2f}")
    
    with db_manager.get_db_session() as db:
        # Get active soccer markets
        active_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False,
            Market.maturity_date > datetime.now(timezone.utc)
        ).limit(5).all()
        
        print(f"\nFound {len(active_markets)} active markets")
        
        # Create paper orders for testing
        for i, market in enumerate(active_markets[:3]):  # Test with first 3
            print(f"\n{i+1}. {market.home_team} vs {market.away_team}")
            print(f"   Kick-off: {market.maturity_date}")
            
            # Get latest odds
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id,
                Odd.outcome.in_(['option_1', 'option_2', 'option_3'])
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            if odds:
                # Find best odds
                best_odd = max(odds, key=lambda o: o.decimal_odds or 0)
                
                # Create a paper order
                stake = 100.0  # $100 bet
                order = PaperOrder(
                    order_id=f"TEST_{i}_{datetime.now().timestamp()}",
                    timestamp=datetime.now(timezone.utc),
                    source_id=market.source_id,
                    market_type="winner",
                    bet_name=best_odd.outcome,
                    side="buy",
                    size=stake,
                    limit_price=best_odd.decimal_odds,
                    signal_name="test_signal",
                    expected_edge=0.05  # 5% edge
                )
                
                print(f"   Creating order: {best_odd.outcome} @ {best_odd.decimal_odds}")
                print(f"   Stake: ${stake}")
                
                # Submit order
                fills = await engine.submit_order(order)
                
                if fills:
                    fill = fills[0]
                    print("   ✓ Order filled!")
                    print(f"     Execution price: {fill.price}")
                    print(f"     Commission: ${fill.commission:.2f}")
                else:
                    print("   ✗ Order failed")
        
        # Show portfolio status
        print("\n=== PORTFOLIO STATUS ===")
        print(f"Capital remaining: ${engine.current_capital:,.2f}")
        print(f"Total orders: {len(engine.orders)}")
        print(f"Total fills: {len(engine.fills)}")
        
        # Show positions
        if engine.positions:
            print("\nActive positions:")
            for market_id, position in engine.positions.items():
                print(f"  {market_id[:16]}...: ${position:.2f}")
        
        # Test getting portfolio value
        portfolio_value = engine.get_portfolio_value()
        print(f"\nTotal portfolio value: ${portfolio_value:,.2f}")
        
        # Save state
        state = {
            'capital': engine.current_capital,
            'orders': len(engine.orders),
            'fills': len(engine.fills),
            'positions': dict(engine.positions),
            'portfolio_value': portfolio_value
        }
        
        with open('test_paper_trading_result.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
        
        print("\nResults saved to test_paper_trading_result.json")

def check_data_freshness():
    """Check how fresh the market data is."""
    print("\n=== DATA FRESHNESS CHECK ===")
    
    with db_manager.get_db_session() as db:
        # Get most recent update times
        recent = db.execute("""
            SELECT source, MAX(last_update) as latest
            FROM market
            GROUP BY source
        """).fetchall()
        
        now = datetime.now(timezone.utc)
        for source, latest in recent:
            if latest:
                age = (now - latest).total_seconds() / 60  # minutes
                print(f"{source}: Last updated {age:.1f} minutes ago")

async def main():
    await test_paper_trading_with_live_data()
    check_data_freshness()

if __name__ == "__main__":
    asyncio.run(main())