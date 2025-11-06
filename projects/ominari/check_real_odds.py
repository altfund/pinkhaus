#!/usr/bin/env python3
"""Check what odds are actually in the data"""

import os
import asyncio

# Set up environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

async def check_odds():
    from unified_data_fetcher import UnifiedDataFetcher
    
    fetcher = UnifiedDataFetcher(blockchain_first=True)
    markets = await fetcher.fetch_all_markets()
    trading_markets = fetcher.format_for_trading(markets)
    
    print(f"\nTotal markets: {len(trading_markets)}")
    print("\nFirst 10 markets with odds:")
    
    for i, market in enumerate(trading_markets[:10]):
        print(f"\n{i+1}. {market.get('home_team', 'Unknown')} vs {market.get('away_team', 'Unknown')}")
        print(f"   Position: {market.get('position', market.get('outcome', 'Unknown'))}")
        print(f"   Odds: {market.get('odds', 'None')}")
        print(f"   All keys: {list(market.keys())}")
        
    # Group by match to see all positions
    print("\n\nFirst 3 matches with all positions:")
    matches = {}
    for market in trading_markets:
        match_key = f"{market.get('home_team')}-{market.get('away_team')}"
        if match_key not in matches:
            matches[match_key] = []
        matches[match_key].append(market)
    
    for i, (match_key, positions) in enumerate(list(matches.items())[:3]):
        print(f"\n{i+1}. {match_key}")
        for pos in positions:
            print(f"   {pos.get('position', pos.get('outcome'))}: odds={pos.get('odds')}")

if __name__ == "__main__":
    asyncio.run(check_odds())