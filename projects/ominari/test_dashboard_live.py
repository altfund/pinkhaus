#!/usr/bin/env python3
"""
Test the dashboard with live data
"""
import os
os.environ['PG_PORT'] = '5999'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market
from datetime import datetime
import requests

def test_dashboard():
    """Test dashboard functionality."""
    
    print("🧪 Testing Ominari Dashboard")
    print("="*50)
    
    # 1. Check database
    with db_manager.get_db_session() as db:
        now = datetime.now()  # timezone-naive
        
        total_markets = db.query(Market).count()
        upcoming_markets = db.query(Market).filter(
            Market.is_finished == False,
            Market.maturity_date > now
        ).count()
        
        soccer_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False, 
            Market.maturity_date > now
        ).count()
        
        print(f"📊 Database Status:")
        print(f"   Total markets: {total_markets}")
        print(f"   Upcoming markets: {upcoming_markets}")
        print(f"   Upcoming soccer: {soccer_markets}")
    
    # 2. Test web server
    print(f"\n🌐 Testing Web Server:")
    try:
        response = requests.get('http://localhost:8889/')
        print(f"   Server status: {'✅ Running' if response.status_code == 200 else '❌ Not running'}")
        
        if response.status_code == 200:
            # Check if we have market cards
            if 'market-card' in response.text:
                print(f"   Market cards: ✅ Found")
            else:
                print(f"   Market cards: ❌ Not found")
                
            # Check for our test games
            if 'Manchester United' in response.text:
                print(f"   Test games: ✅ Visible")
            else:
                print(f"   Test games: ❌ Not visible")
                
    except Exception as e:
        print(f"   Server status: ❌ Error - {e}")
    
    # 3. Test API endpoint
    print(f"\n🔌 Testing API:")
    try:
        response = requests.get('http://localhost:8889/api/data')
        if response.status_code == 200:
            data = response.json()
            print(f"   API status: ✅ Working")
            print(f"   Markets returned: {len(data.get('markets', []))}")
            print(f"   Portfolio cash: ${data.get('account', {}).get('balance', 0):,.2f}")
            
            # Show first 3 markets
            markets = data.get('markets', [])
            if markets:
                print(f"\n📋 Sample Markets:")
                for i, market in enumerate(markets[:3]):
                    print(f"   {i+1}. {market['home_team']} vs {market['away_team']}")
                    print(f"      Odds: H:{market.get('home_odds', '-')} D:{market.get('draw_odds', '-')} A:{market.get('away_odds', '-')}")
        else:
            print(f"   API status: ❌ Error {response.status_code}")
    except Exception as e:
        print(f"   API status: ❌ Error - {e}")
    
    print("\n✅ Dashboard test complete!")
    print(f"🌐 Open http://localhost:8889 in your browser to view the dashboard")

if __name__ == "__main__":
    test_dashboard()