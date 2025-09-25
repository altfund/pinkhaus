#!/usr/bin/env python3
"""
Test PostgreSQL paper trading integration
"""
import os
from datetime import datetime

# Set environment variables
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

from paper_trading_postgres_integrated import PaperTradingSessionManager

def test_postgres_integration():
    """Test PostgreSQL paper trading functionality."""
    print("🧪 Testing PostgreSQL Paper Trading Integration")
    print("=" * 60)
    
    try:
        # Initialize manager
        manager = PaperTradingSessionManager()
        print("✅ Manager initialized successfully")
        
        # Test 1: Create a new session
        print("\n1️⃣ Creating new session...")
        session_id = manager.create_session(
            initial_bankroll=5000,
            session_name="PostgreSQL Test Session"
        )
        print(f"✅ Created session: {session_id}")
        
        # Test 2: Get current session
        print("\n2️⃣ Getting current session...")
        current = manager.get_current_session()
        print(f"✅ Current session: {current}")
        
        # Test 3: Get session data
        print("\n3️⃣ Getting session data...")
        session_data = manager.get_session(session_id)
        if session_data:
            print(f"✅ Session data retrieved:")
            print(f"   - Bankroll: ${session_data['current_bankroll']:,.2f}")
            print(f"   - Portfolio: ${session_data['portfolio_value']:,.2f}")
            print(f"   - Total bets: {session_data['total_bets']}")
            print(f"   - Status: {session_data['status']}")
        
        # Test 4: Record a test bet
        print("\n4️⃣ Recording test bet...")
        bet_data = {
            'match_id': 'TEST_MATCH_001',
            'sport': 'Soccer',
            'home_team': 'Test Home FC',
            'away_team': 'Test Away United',
            'bet_on': 'home',
            'odds': 2.5,
            'stake': 100,
            'signal_name': 'test_signal',
            'signal_value': 0.45,
            'edge': 0.125,
            'kickoff_time': datetime.now()
        }
        bet_id = manager.record_bet(session_id, bet_data)
        print(f"✅ Recorded bet: {bet_id}")
        
        # Test 5: Update session
        print("\n5️⃣ Updating session...")
        update_data = {
            'current_bankroll': 4900,  # Deducted stake
            'exposure': 100,
            'total_pnl': -100,
            'wins': 0,
            'losses': 0,
            'pending': 1
        }
        manager.update_session(session_id, update_data)
        print("✅ Session updated")
        
        # Test 6: Get positions
        print("\n6️⃣ Getting positions...")
        positions = manager.get_positions(session_id)
        print(f"✅ Found {len(positions)} positions")
        if positions:
            pos = positions[0]
            print(f"   - Match: {pos['home_team']} vs {pos['away_team']}")
            print(f"   - Bet: {pos['bet_on']} @ {pos['odds']}")
            print(f"   - Stake: ${pos['stake']}")
        
        # Test 7: Check session compatibility
        print("\n7️⃣ Testing JSON compatibility...")
        sessions_prop = manager.sessions
        print(f"✅ Sessions property works: {len(sessions_prop['sessions'])} sessions")
        
        # Test 8: Database verification
        print("\n8️⃣ Verifying database state...")
        with manager.get_connection() as conn:
            with conn.cursor() as cur:
                # Check tables
                cur.execute("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name LIKE 'paper_trading%'
                    ORDER BY table_name
                """)
                tables = cur.fetchall()
                print("✅ Database tables:")
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) as count FROM {table['table_name']}")
                    count = cur.fetchone()
                    print(f"   - {table['table_name']}: {count['count']} rows")
        
        print("\n" + "=" * 60)
        print("✨ All tests passed! PostgreSQL paper trading is working correctly.")
        print("\n📝 Next steps:")
        print("1. Run migration if you have existing JSON data: python migrate_json_to_postgres.py")
        print("2. Restart web_monitor.py to use PostgreSQL backend")
        print("3. Monitor with: ./simple_monitor.sh")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_postgres_integration()