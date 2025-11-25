#!/usr/bin/env python3
"""
Dashboard Content Verification Test
Tests that dashboard elements are populated with actual data, not just present in HTML.
"""
import time
import requests
import re
import json

DASHBOARD_URL = 'http://localhost:8888/'

def test_api_data():
    """Test that APIs return actual data"""
    print("\n" + "="*60)
    print("📡 TESTING API ENDPOINTS")
    print("="*60 + "\n")

    tests_passed = []
    tests_failed = []

    # Test Portfolio API
    try:
        resp = requests.get(f'{DASHBOARD_URL}api/portfolio', timeout=5)
        data = resp.json()

        portfolio_value = data['portfolio']['value']
        open_positions = data['portfolio']['openPositions']
        total_trades = data['portfolio']['totalTrades']
        historical_points = len(data.get('historical', []))

        if portfolio_value > 0:
            tests_passed.append(f"Portfolio API - Value: ${portfolio_value:,.2f}")
        else:
            tests_failed.append("Portfolio API - Value is 0")

        if open_positions >= 0:
            tests_passed.append(f"Portfolio API - Positions: {open_positions}")

        if historical_points > 0:
            tests_passed.append(f"Portfolio API - Historical data: {historical_points} points")
        else:
            tests_failed.append("Portfolio API - No historical data")

    except Exception as e:
        tests_failed.append(f"Portfolio API - Error: {e}")

    # Test Trades API
    try:
        resp = requests.get(f'{DASHBOARD_URL}api/trades', timeout=5)
        data = resp.json()

        if data.get('success'):
            total_trades = data.get('total_trades', 0)
            trades_list = data.get('trades', [])

            if total_trades > 0:
                tests_passed.append(f"Trades API - Total: {total_trades} trades")
            else:
                tests_failed.append("Trades API - No trades")

            if trades_list:
                # Check first trade has required fields
                first_trade = trades_list[0]
                required_fields = ['match', 'outcome', 'stake', 'odds', 'status']
                missing_fields = [f for f in required_fields if f not in first_trade]

                if not missing_fields:
                    tests_passed.append(f"Trades API - Data structure valid")
                    # Show sample trade
                    tests_passed.append(f"Trades API - Sample: {first_trade['match'][:60]}")
                else:
                    tests_failed.append(f"Trades API - Missing fields: {missing_fields}")
        else:
            tests_failed.append("Trades API - Success=false")

    except Exception as e:
        tests_failed.append(f"Trades API - Error: {e}")

    # Test Markets API
    try:
        resp = requests.get(f'{DASHBOARD_URL}api/markets', timeout=5)
        data = resp.json()

        if data.get('status') == 'success':
            markets = data.get('markets', [])

            if markets:
                tests_passed.append(f"Markets API - {len(markets)} markets available")

                # Check first market has required fields
                first_market = markets[0]
                required_fields = ['name', 'homeOdds', 'awayOdds', 'edge', 'startTime']
                missing_fields = [f for f in required_fields if f not in first_market]

                if not missing_fields:
                    tests_passed.append(f"Markets API - Data structure valid")
                    tests_passed.append(f"Markets API - Sample: {first_market['name'][:50]}")
                else:
                    tests_failed.append(f"Markets API - Missing fields: {missing_fields}")
            else:
                # This might be OK if no opportunities
                tests_passed.append("Markets API - No markets (may be normal)")
        else:
            tests_failed.append("Markets API - Status not success")

    except Exception as e:
        tests_failed.append(f"Markets API - Error: {e}")

    # Test Analytics API
    try:
        resp = requests.get(f'{DASHBOARD_URL}api/portfolio/analytics', timeout=5)
        data = resp.json()

        if data.get('status') == 'success':
            sharpe = data.get('sharpeRatio', 0)
            win_rate = data.get('winRate', 0)
            volatility = data.get('volatility', 0)

            tests_passed.append(f"Analytics API - Sharpe: {sharpe:.2f}, Win Rate: {win_rate:.1f}%")

            if data.get('dailyReturns'):
                tests_passed.append(f"Analytics API - {len(data['dailyReturns'])} daily returns")
        else:
            tests_failed.append("Analytics API - Status not success")

    except Exception as e:
        tests_failed.append(f"Analytics API - Error: {e}")

    # Print results
    for test in tests_passed:
        print(f"✅ {test}")

    for test in tests_failed:
        print(f"❌ {test}")

    return len(tests_failed) == 0

def test_html_elements():
    """Test that HTML elements exist"""
    print("\n" + "="*60)
    print("🌐 TESTING HTML ELEMENTS")
    print("="*60 + "\n")

    try:
        resp = requests.get(DASHBOARD_URL, timeout=5)
        html = resp.text

        required_elements = {
            'portfolioValue': 'Portfolio value display',
            'portfolioChange': 'Portfolio change display',
            'openPositions': 'Open positions counter',
            'totalTrades': 'Total trades counter',
            'activeStake': 'Active stake display',
            'marketGrid': 'Market opportunities container',
            'marketCount': 'Market count display',
            'tradesTableBody': 'Trades table body',
            'portfolioChart': 'Portfolio chart canvas',
            'positionChart': 'Position breakdown chart',
            'heartbeatStatus': 'System heartbeat indicator',
            'lastUpdate': 'Last update timestamp'
        }

        tests_passed = []
        tests_failed = []

        for element_id, description in required_elements.items():
            if f'id="{element_id}"' in html:
                tests_passed.append(f"{element_id} ({description})")
            else:
                tests_failed.append(f"{element_id} ({description})")

        # Check JavaScript files
        if 'ominari_dashboard.js' in html:
            tests_passed.append("ominari_dashboard.js loaded")
        else:
            tests_failed.append("ominari_dashboard.js not loaded")

        if 'dashboard_analytics.js' in html:
            tests_passed.append("dashboard_analytics.js loaded")
        else:
            tests_failed.append("dashboard_analytics.js not loaded")

        # Check initialization
        if 'initializeDashboard' in html:
            tests_passed.append("Dashboard initialization present")
        else:
            tests_failed.append("Dashboard initialization missing")

        # Print results
        for test in tests_passed:
            print(f"✅ {test}")

        for test in tests_failed:
            print(f"❌ {test}")

        return len(tests_failed) == 0

    except Exception as e:
        print(f"❌ Error fetching dashboard HTML: {e}")
        return False

def test_data_flow():
    """Test complete data flow - verify APIs return non-default data"""
    print("\n" + "="*60)
    print("🔄 TESTING DATA FLOW & CONTENT")
    print("="*60 + "\n")

    try:
        # Get all API data
        portfolio_resp = requests.get(f'{DASHBOARD_URL}api/portfolio', timeout=5)
        portfolio_data = portfolio_resp.json()

        tests_passed = []
        tests_failed = []

        # Verify portfolio data is realistic (not defaults)
        portfolio_value = portfolio_data['portfolio']['value']
        if portfolio_value != 10000.0:  # Not default value
            tests_passed.append(f"Portfolio has real trading data (${portfolio_value:,.2f} != $10,000 default)")
        elif portfolio_data['portfolio']['totalTrades'] > 0:
            tests_passed.append(f"Portfolio at default value but has {portfolio_data['portfolio']['totalTrades']} trades")
        else:
            tests_failed.append("Portfolio appears to be at default state")

        # Verify chart data exists and has variation
        historical = portfolio_data.get('historical', [])
        if len(historical) >= 10:
            values = [p['value'] for p in historical]
            has_variation = max(values) - min(values) > 1.0
            if has_variation:
                tests_passed.append(f"Chart data shows variation (${min(values):.0f} - ${max(values):.0f})")
            else:
                tests_failed.append("Chart data is flat (no variation)")
        else:
            tests_failed.append(f"Insufficient chart data ({len(historical)} points)")

        # Verify position data
        positions = portfolio_data.get('positions', {})
        cash = positions.get('cash', 0)
        active_stake = positions.get('activeStake', 0)

        if cash > 0:
            tests_passed.append(f"Position breakdown has cash: ${cash:,.2f}")

        if active_stake > 0:
            tests_passed.append(f"Position breakdown has stakes: ${active_stake:,.2f}")
        elif portfolio_data['portfolio']['openPositions'] == 0:
            tests_passed.append("No active stakes (no open positions)")
        else:
            tests_failed.append("Has open positions but no stake data")

        # Print results
        for test in tests_passed:
            print(f"✅ {test}")

        for test in tests_failed:
            print(f"❌ {test}")

        return len(tests_failed) == 0

    except Exception as e:
        print(f"❌ Data flow test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_service_health():
    """Test service health and connectivity"""
    print("\n" + "="*60)
    print("🏥 TESTING SERVICE HEALTH")
    print("="*60 + "\n")

    try:
        resp = requests.get(f'{DASHBOARD_URL}health', timeout=5)
        health = resp.json()

        print(f"Overall Status: {health['status'].upper()}")
        print(f"Timestamp: {health['timestamp']}")
        print(f"\nService Status:")

        all_up = True
        for service_name, service_data in health.get('services', {}).items():
            status = service_data.get('status', 'unknown')
            print(f"   {'✅' if status == 'up' else '❌'} {service_name}: {status}")

            if status != 'up':
                all_up = False

            if service_name == 'database':
                markets = service_data.get('markets', 0)
                print(f"      └─ Markets in DB: {markets:,}")
            elif service_name == 'websocket':
                clients = service_data.get('clients', 0)
                print(f"      └─ Connected clients: {clients}")

        return health['status'] == 'healthy' and all_up

    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Run all dashboard tests"""
    print("\n" + "="*70)
    print("🧪 DASHBOARD COMPREHENSIVE CONTENT TEST")
    print("="*70)
    print(f"Dashboard URL: {DASHBOARD_URL}")
    print(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all tests
    service_ok = test_service_health()
    api_ok = test_api_data()
    html_ok = test_html_elements()
    data_flow_ok = test_data_flow()

    # Final summary
    print("\n" + "="*70)
    print("📋 FINAL TEST SUMMARY")
    print("="*70 + "\n")

    results = [
        ("Service Health", service_ok),
        ("API Data", api_ok),
        ("HTML Elements", html_ok),
        ("Data Flow", data_flow_ok)
    ]

    for test_name, passed in results:
        print(f"{'✅' if passed else '❌'} {test_name}")

    all_passed = all(r[1] for r in results)

    print("\n" + "="*70)
    if all_passed:
        print("✅ DASHBOARD FULLY OPERATIONAL")
        print("="*70)
        print("\n🌐 Access dashboard at: http://localhost:8888/")
        print("📊 All data loading correctly")
        print("📈 Charts should be rendering with real data")
        print("🔄 Auto-refresh every 30 seconds")
    else:
        print("⚠️  DASHBOARD HAS ISSUES")
        print("="*70)
        print("\nReview failed tests above for details")
    print()

    return all_passed

if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
