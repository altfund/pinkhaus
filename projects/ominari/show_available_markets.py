#!/usr/bin/env python3
"""Show available soccer markets for trading"""

import os
import asyncio
from datetime import datetime, timezone, timedelta

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from unified_data_fetcher import UnifiedDataFetcher

async def main():
    """Show available markets"""
    fetcher = UnifiedDataFetcher()
    
    print("⚽ Fetching available soccer markets...\n")
    
    # Fetch all markets
    markets = await fetcher.fetch_all_markets()
    
    print(f"Found {len(markets)} total markets\n")
    
    # Group by time
    now = datetime.now(timezone.utc)
    live_soon = []
    today = []
    tomorrow = []
    later = []
    
    for market in markets:
        maturity = market['maturity_date']
        if isinstance(maturity, str):
            maturity = datetime.fromisoformat(maturity.replace('Z', '+00:00'))
        
        # Ensure timezone aware
        if maturity.tzinfo is None:
            maturity = maturity.replace(tzinfo=timezone.utc)
        
        hours_away = (maturity - now).total_seconds() / 3600
        
        if hours_away < 0:
            continue  # Skip past
        elif hours_away < 2:
            live_soon.append((market, hours_away))
        elif hours_away < 24:
            today.append((market, hours_away))
        elif hours_away < 48:
            tomorrow.append((market, hours_away))
        else:
            later.append((market, hours_away))
    
    # Show markets by category
    if live_soon:
        print(f"🔴 LIVE/STARTING SOON ({len(live_soon)} markets):")
        for market, hours in sorted(live_soon, key=lambda x: x[1])[:10]:
            print(f"\n   {market['home_team']} vs {market['away_team']}")
            print(f"   League: {market.get('league', 'Unknown')}")
            print(f"   Starts in: {hours:.1f} hours")
            print(f"   Odds: H:{market['odds']['home']:.2f} D:{market['odds'].get('draw', 'N/A')} A:{market['odds']['away']:.2f}")
            sources = ', '.join(market.get('data_sources', ['unknown']))
            print(f"   Source: {sources}")
    
    if today:
        print(f"\n📅 TODAY ({len(today)} markets):")
        for market, hours in sorted(today, key=lambda x: x[1])[:5]:
            print(f"\n   {market['home_team']} vs {market['away_team']}")
            print(f"   League: {market.get('league', 'Unknown')}")
            print(f"   Starts in: {hours:.1f} hours")
            print(f"   Odds: H:{market['odds']['home']:.2f} D:{market['odds'].get('draw', 'N/A')} A:{market['odds']['away']:.2f}")
    
    if tomorrow:
        print(f"\n📆 TOMORROW ({len(tomorrow)} markets):")
        for market, hours in sorted(tomorrow, key=lambda x: x[1])[:5]:
            print(f"\n   {market['home_team']} vs {market['away_team']}")
            print(f"   Starts in: {hours:.1f} hours")
    
    print(f"\n📊 Summary:")
    print(f"   Live/Soon: {len(live_soon)} markets")
    print(f"   Today: {len(today)} markets")
    print(f"   Tomorrow: {len(tomorrow)} markets")
    print(f"   Later: {len(later)} markets")
    
    # Show data source breakdown
    blockchain_count = sum(1 for m in markets if 'blockchain' in m.get('data_sources', []))
    api_count = sum(1 for m in markets if 'api' in m.get('data_sources', []))
    
    print(f"\n📡 Data Sources:")
    print(f"   From blockchain: {blockchain_count}")
    print(f"   From API: {api_count}")

if __name__ == "__main__":
    asyncio.run(main())