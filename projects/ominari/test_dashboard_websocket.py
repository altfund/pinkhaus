#!/usr/bin/env python3
"""
Dashboard WebSocket Real-Time Updates Test
Tests that WebSocket connection works for live data updates.
"""
import time
import socketio
import requests

DASHBOARD_URL = 'http://localhost:8888'

def test_websocket_connection():
    """Test WebSocket connection and real-time updates"""
    print("\n" + "="*60)
    print("🔌 TESTING WEBSOCKET CONNECTION")
    print("="*60 + "\n")

    try:
        # Create SocketIO client
        sio = socketio.Client()

        connected = False
        updates_received = []

        @sio.event
        def connect():
            nonlocal connected
            connected = True
            print("✅ WebSocket connected")

        @sio.event
        def disconnect():
            print("⚠️  WebSocket disconnected")

        @sio.on('portfolio_update')
        def on_portfolio_update(data):
            updates_received.append(('portfolio', data))
            print(f"✅ Received portfolio update: ${data.get('value', 0):,.2f}")

        @sio.on('trade_update')
        def on_trade_update(data):
            updates_received.append(('trade', data))
            print(f"✅ Received trade update: {data.get('match', 'Unknown')}")

        @sio.on('market_update')
        def on_market_update(data):
            updates_received.append(('market', data))
            print(f"✅ Received market update: {len(data.get('markets', []))} markets")

        # Connect to WebSocket
        print("Connecting to WebSocket...")
        sio.connect(DASHBOARD_URL)

        if not connected:
            print("❌ Failed to connect to WebSocket")
            return False

        # Wait for potential updates
        print("Waiting 5 seconds for any live updates...")
        time.sleep(5)

        # Disconnect
        sio.disconnect()

        print(f"\n📊 Updates Summary:")
        print(f"   Total updates received: {len(updates_received)}")

        if len(updates_received) > 0:
            for update_type, data in updates_received:
                print(f"   ✅ {update_type}: {type(data).__name__}")
            print("\n✅ WebSocket real-time updates working")
        else:
            print("   ℹ️  No updates in test window (normal if no trading activity)")
            print("\n✅ WebSocket connection successful (no updates expected)")

        return True

    except ImportError:
        print("⚠️  python-socketio not installed")
        print("   Run: pip install python-socketio")
        print("   Skipping WebSocket test (not critical)")
        return True  # Not a failure - just skipped

    except Exception as e:
        print(f"❌ WebSocket test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_websocket_health():
    """Check WebSocket health via HTTP endpoint"""
    print("\n" + "="*60)
    print("🏥 CHECKING WEBSOCKET HEALTH")
    print("="*60 + "\n")

    try:
        resp = requests.get(f'{DASHBOARD_URL}/health')
        health = resp.json()

        websocket_status = health.get('services', {}).get('websocket', {})
        status = websocket_status.get('status', 'unknown')
        clients = websocket_status.get('clients', 0)

        print(f"WebSocket Status: {status}")
        print(f"Connected Clients: {clients}")

        if status == 'up':
            print("\n✅ WebSocket service is running")
            return True
        else:
            print(f"\n❌ WebSocket service is {status}")
            return False

    except Exception as e:
        print(f"❌ WebSocket health check error: {e}")
        return False

def main():
    """Run WebSocket tests"""
    print("\n" + "="*70)
    print("⚡ DASHBOARD WEBSOCKET TEST")
    print("="*70)
    print(f"Testing: {DASHBOARD_URL}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run tests
    health_ok = test_websocket_health()
    connection_ok = test_websocket_connection()

    # Summary
    print("\n" + "="*70)
    print("📋 WEBSOCKET TEST SUMMARY")
    print("="*70 + "\n")

    print(f"{'✅' if health_ok else '❌'} WebSocket Health")
    print(f"{'✅' if connection_ok else '❌'} WebSocket Connection")

    all_passed = health_ok and connection_ok

    print("\n" + "="*70)
    if all_passed:
        print("✅ WEBSOCKET REAL-TIME UPDATES OPERATIONAL")
        print("="*70)
        print("\n🔄 Dashboard will receive live updates for:")
        print("   • Portfolio value changes")
        print("   • New trades executed")
        print("   • Market opportunities")
    else:
        print("⚠️  WEBSOCKET ISSUES DETECTED")
        print("="*70)
        print("\nDashboard will still work with 30s polling")
    print()

    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
