#!/usr/bin/env python3
"""
Test script to verify paper trading is working end-to-end
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
from paper_trading_live import LivePaperTrader


async def test_components():
    """Test individual components"""
    print("🧪 Testing Paper Trading Components")
    print("=" * 50)
    
    # Test 1: Database connection
    print("\n1. Testing database connection...")
    try:
        with db_manager.get_db_session() as db:
            market_count = db.query(Market).count()
            print(f"✅ Database connected. Markets: {market_count}")
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False
        
    # Test 2: Bankroll config
    print("\n2. Testing bankroll configuration...")
    try:
        bankroll = BankrollConfig()
        current = bankroll.get_current_bankroll()
        print(f"✅ Bankroll config loaded. Current: ${current:.2f}")
    except Exception as e:
        print(f"❌ Bankroll error: {e}")
        return False
        
    # Test 3: Find markets with odds
    print("\n3. Testing market and odds data...")
    try:
        with db_manager.get_db_session() as db:
            # Get a market with odds
            market_with_odds = db.query(Market).join(
                Odd, Market.source_id == Odd.source_id
            ).first()
            
            if market_with_odds:
                odds_count = db.query(Odd).filter(
                    Odd.source_id == market_with_odds.source_id
                ).count()
                print(f"✅ Found market: {market_with_odds.home_team} vs {market_with_odds.away_team}")
                print(f"   Odds records: {odds_count}")
            else:
                print("❌ No markets with odds found")
                return False
    except Exception as e:
        print(f"❌ Market query error: {e}")
        return False
        
    return True


async def test_trading_logic():
    """Test the trading logic"""
    print("\n🎯 Testing Trading Logic")
    print("=" * 50)
    
    trader = LivePaperTrader()
    
    # Test finding opportunities
    print("\n1. Finding betting opportunities...")
    opportunities = await trader.find_betting_opportunities()
    
    if opportunities:
        print(f"✅ Found {len(opportunities)} opportunities")
        for i, opp in enumerate(opportunities[:3]):
            print(f"\n   Opportunity {i+1}:")
            print(f"   Match: {opp['market'].home_team} vs {opp['market'].away_team}")
            print(f"   Outcome: {opp['outcome']}")
            print(f"   Odds: {opp['odds']:.2f}")
            print(f"   Edge: {opp['edge']:.2f}%")
            print(f"   Fair Prob: {opp['fair_prob']:.2%}")
    else:
        print("❌ No opportunities found")
        print("   This might be because:")
        print("   - No markets in next 24 hours")
        print("   - No markets with positive edge")
        print("   - Database has no recent odds")
        
    # Test bet calculation
    print("\n2. Testing bet sizing...")
    test_prob = 0.55  # 55% win probability
    test_odds = 2.0   # Even money
    test_bankroll = 10000
    
    bet_size = trader.calculate_kelly_bet(test_prob, test_odds, test_bankroll)
    print(f"✅ Kelly bet calculation:")
    print(f"   Probability: {test_prob:.0%}")
    print(f"   Odds: {test_odds}")
    print(f"   Bankroll: ${test_bankroll}")
    print(f"   Bet size: ${bet_size:.2f} ({bet_size/test_bankroll*100:.1f}% of bankroll)")
    
    return len(opportunities) > 0


async def run_mini_session():
    """Run a mini trading session"""
    print("\n🚀 Running Mini Trading Session (30 seconds)")
    print("=" * 50)
    
    trader = LivePaperTrader()
    
    # Create a test session
    try:
        with db_manager.get_db_session() as db:
            current_bankroll = trader.bankroll_config.get_current_bankroll()
            session = BettingSession(
                as_of=datetime.now(timezone.utc),
                session_type='paper',
                strategy_name="Test Strategy",
                kelly_bankroll=current_bankroll,
                execution_bankroll=current_bankroll,
                kelly_fraction=0.25,
                cap_per_game=current_bankroll * 0.1,
                cap_per_bet=current_bankroll * 0.05,
                cap_per_game_market=current_bankroll * 0.05,
                min_bet_abs=10.0,
                min_bet_pct=0.001,
                abs_game_limit=None,
                min_break_minutes=0.0,
                avg_game_duration_minutes=120.0
            )
            db.add(session)
            db.commit()
            trader.session_id = session.id
            print(f"✅ Created test session: {session.id}")
    except Exception as e:
        print(f"❌ Failed to create session: {e}")
        return False
        
    # Run for 30 seconds
    trader.is_running = True
    start_time = datetime.now()
    
    while (datetime.now() - start_time).seconds < 30:
        try:
            # Find opportunities
            opportunities = await trader.find_betting_opportunities()
            
            if opportunities:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Found {len(opportunities)} opportunities")
                
                # Try to place one bet
                for opp in opportunities[:1]:  # Just first one
                    bet = await trader.place_bet(opp)
                    if bet:
                        print(f"✅ Placed test bet!")
                        break
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] No opportunities found")
                
            await asyncio.sleep(10)
            
        except Exception as e:
            print(f"❌ Error in trading loop: {e}")
            
    # Check results
    print("\n📊 Session Summary:")
    with db_manager.get_db_session() as db:
        bets = db.query(Bet).filter(
            Bet.session_id == trader.session_id
        ).all()
        
        print(f"Total bets placed: {len(bets)}")
        for bet in bets:
            print(f"  - {bet.bet_name}: @ {bet.odds} (${bet.stake:.2f})")
            
    return True


async def main():
    """Run all tests"""
    print("🏁 PAPER TRADING END-TO-END TEST")
    print("=" * 50)
    print(f"Started at: {datetime.now()}")
    print()
    
    # Run component tests
    if not await test_components():
        print("\n❌ Component tests failed. Fix these first!")
        return
        
    # Test trading logic
    if not await test_trading_logic():
        print("\n⚠️  No trading opportunities found")
        print("This is normal if:")
        print("- Database has old data")
        print("- No games in next 24 hours")
        print("- All markets have negative edge")
        
    # Run mini session
    print("\n" + "=" * 50)
    input("Press Enter to run a 30-second trading session...")
    
    await run_mini_session()
    
    print("\n✅ Test complete!")
    print("\nNext steps:")
    print("1. Check config/bankroll.json for bankroll tracking")
    print("2. Run the dashboard to see trades: python main.py")
    print("3. Check /api/trades endpoint for trade display")


if __name__ == "__main__":
    asyncio.run(main())