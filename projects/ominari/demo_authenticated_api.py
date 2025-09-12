#!/usr/bin/env python3
"""
Demo script for testing authenticated API access
"""

import os
import time
from ominari_api_client import OminariAPIClient, OminariAPIError

def test_authenticated_api():
    """Test API with authentication."""
    print("=== Ominari Authenticated API Demo ===\n")
    
    # Check for API key
    api_key = os.getenv('OMINARI_API_KEY')
    
    if not api_key:
        print("⚠️  No OMINARI_API_KEY found in environment")
        print("\nTo create an API key:")
        print("python api_auth.py create --name 'Demo Key' --permissions read trade")
        print("\nThen export it:")
        print("export OMINARI_API_KEY='your-key-here'\n")
        
        # Try without authentication
        print("Attempting unauthenticated access...\n")
    
    # Initialize client
    client = OminariAPIClient(api_key=api_key)
    
    # Test endpoints
    print("1. Testing Health Check (public endpoint):")
    try:
        if client.health_check():
            print("   ✅ API is healthy\n")
        else:
            print("   ❌ API is not healthy\n")
    except Exception as e:
        print(f"   ❌ Error: {e}\n")
    
    # Test authenticated endpoints
    print("2. Testing System Status (requires auth):")
    try:
        status = client.get_system_status()
        print(f"   ✅ Status: {status.get('status')}")
        print(f"   ✅ Active markets: {status.get('active_markets', 0)}\n")
    except OminariAPIError as e:
        print(f"   ❌ Auth Error: {e}\n")
    
    print("3. Testing Trading Status:")
    try:
        trading = client.get_trading_status()
        print(f"   ✅ Trading active: {trading.get('is_active', False)}")
        print(f"   ✅ Mode: {trading.get('mode', 'unknown')}\n")
    except OminariAPIError as e:
        print(f"   ❌ Error: {e}\n")
    
    print("4. Testing Portfolio Access:")
    try:
        portfolio = client.get_portfolio("default")
        print(f"   ✅ Total value: ${portfolio.total_value:,.2f}")
        print(f"   ✅ Open positions: {portfolio.open_positions}")
        print(f"   ✅ P&L: ${portfolio.total_pnl:,.2f} ({portfolio.total_pnl_pct:.2f}%)\n")
    except OminariAPIError as e:
        print(f"   ❌ Error: {e}\n")
    
    print("5. Testing Rate Limits:")
    if api_key:
        print("   Making rapid requests to test rate limiting...")
        start_time = time.time()
        request_count = 0
        
        try:
            for i in range(10):
                client.get_system_status()
                request_count += 1
                print(f"   Request {i+1}: ✅")
                time.sleep(0.1)
        except OminariAPIError as e:
            if "Rate limit" in str(e):
                print(f"   ⚠️  Rate limit hit after {request_count} requests")
                print(f"   Time elapsed: {time.time() - start_time:.1f}s")
            else:
                print(f"   ❌ Error: {e}")
    else:
        print("   ⏭️  Skipping (requires authentication)\n")
    
    print("\n6. Testing Trade Execution (dry run):")
    try:
        result = client.execute_trades("default", dry_run=True)
        print(f"   ✅ Status: {result.get('status')}")
        print(f"   ✅ Recommendations: {len(result.get('recommendations', []))}")
        
        if result.get('recommendations'):
            rec = result['recommendations'][0]
            print(f"\n   Example recommendation:")
            print(f"   - Market: {rec.get('market_name')}")
            print(f"   - Outcome: {rec.get('outcome')}")
            print(f"   - Odds: {rec.get('odds')}")
            print(f"   - Edge: {rec.get('edge'):.2%}")
            print(f"   - Stake: ${rec.get('stake'):.2f}")
    except OminariAPIError as e:
        print(f"   ❌ Error: {e}")
    
    print("\n=== Demo Complete ===")
    
    if not api_key:
        print("\n💡 To access all features, create an API key and set OMINARI_API_KEY")
    else:
        print(f"\n✅ Successfully authenticated with key: {api_key[:15]}...")
        print("📊 All authenticated endpoints are working!")


if __name__ == "__main__":
    test_authenticated_api()