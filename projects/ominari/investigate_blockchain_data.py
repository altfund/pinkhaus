#!/usr/bin/env python3
"""Investigate blockchain data availability for Overtime markets"""

import os
import sys

# Set up database environment
os.environ['USE_POSTGRESQL'] = '1'
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func, and_, or_
from datetime import datetime, timezone, timedelta

print("🔍 Investigating blockchain data availability...")

with db_manager.get_db_session() as db:
    # Check total markets in database
    total_markets = db.query(Market).count()
    print(f"\n📊 Total markets in database: {total_markets}")
    
    # Check by source
    print("\n📈 Markets by source:")
    sources = db.query(Market.source, func.count(Market.source_id))\
        .group_by(Market.source)\
        .order_by(func.count(Market.source_id).desc())\
        .all()
    
    for source, count in sources:
        print(f"  {source}: {count}")
    
    # Check recent markets
    now = datetime.now(timezone.utc)
    recent_cutoff = now - timedelta(hours=24)
    
    recent_markets = db.query(Market)\
        .filter(Market.updated_at > recent_cutoff)\
        .count()
    
    print(f"\n⏰ Markets updated in last 24 hours: {recent_markets}")
    
    # Check unfinished markets
    unfinished = db.query(Market)\
        .filter(Market.is_finished == False)\
        .count()
    
    print(f"🎯 Unfinished markets: {unfinished}")
    
    # Sample blockchain markets with odds
    print("\n🔗 Sample blockchain markets with odds:")
    
    blockchain_markets = db.query(Market)\
        .filter(
            and_(
                Market.source.in_(['blockchain', 'overtime_blockchain']),
                Market.is_finished == False
            )
        )\
        .order_by(Market.maturity_date.asc())\
        .limit(10)\
        .all()
    
    if blockchain_markets:
        for market in blockchain_markets:
            print(f"\n📍 {market.home_team} vs {market.away_team}")
            print(f"   Sport: {market.sport}")
            print(f"   League: {market.league}")
            print(f"   Maturity: {market.maturity_date}")
            print(f"   Source: {market.source}")
            print(f"   Market ID: {market.source_id}")
            
            # Get odds for this market
            odds = db.query(Odd)\
                .filter(Odd.source_id == market.source_id)\
                .order_by(Odd.updated_at.desc())\
                .limit(3)\
                .all()
            
            if odds:
                print("   Odds:")
                for odd in odds:
                    print(f"     {odd.outcome}: {odd.odds} (source: {odd.source})")
            else:
                print("   ❌ No odds found")
    else:
        print("❌ No blockchain markets found")
    
    # Check for markets with both API and blockchain data
    print("\n🔄 Markets with multiple sources:")
    
    # Find markets that appear in multiple sources
    duplicate_markets = db.query(
        Market.home_team,
        Market.away_team,
        Market.maturity_date,
        func.array_agg(func.distinct(Market.source)).label('sources'),
        func.count(func.distinct(Market.source)).label('source_count')
    ).group_by(
        Market.home_team,
        Market.away_team,  
        Market.maturity_date
    ).having(
        func.count(func.distinct(Market.source)) > 1
    ).limit(5).all()
    
    if duplicate_markets:
        for match in duplicate_markets:
            print(f"\n  {match.home_team} vs {match.away_team}")
            print(f"    Sources: {', '.join(match.sources)}")
            print(f"    Maturity: {match.maturity_date}")
    else:
        print("  No markets found in multiple sources")
    
    # Check odds data
    print("\n💰 Odds data analysis:")
    
    total_odds = db.query(Odd).count()
    print(f"  Total odds records: {total_odds}")
    
    # Odds by source
    print("\n  Odds by source:")
    sources_odds = db.query(Odd.source, func.count(Odd.id))\
        .group_by(Odd.source)\
        .order_by(func.count(Odd.id).desc())\
        .all()
    
    for source, count in sources_odds:
        print(f"    {source}: {count}")
    
    # Recent odds updates
    recent_odds = db.query(Odd)\
        .filter(Odd.updated_at > recent_cutoff)\
        .count()
    
    print(f"\n  Odds updated in last 24 hours: {recent_odds}")
    
    # Check for live/upcoming soccer markets
    print("\n⚽ Live/Upcoming Soccer Markets:")
    
    soccer_markets = db.query(Market)\
        .filter(
            and_(
                Market.sport.ilike('%soccer%'),
                Market.is_finished == False,
                Market.maturity_date > now - timedelta(hours=2),
                Market.maturity_date < now + timedelta(hours=48)
            )
        )\
        .order_by(Market.maturity_date.asc())\
        .limit(10)\
        .all()
    
    if soccer_markets:
        for market in soccer_markets:
            print(f"\n  {market.home_team} vs {market.away_team}")
            print(f"    Time: {market.maturity_date}")
            print(f"    Source: {market.source}")
            
            # Get latest odds
            odds = db.query(Odd)\
                .filter(Odd.source_id == market.source_id)\
                .order_by(Odd.updated_at.desc())\
                .limit(3)\
                .all()
            
            if odds:
                odds_str = " / ".join([f"{o.outcome}: {o.odds}" for o in odds])
                print(f"    Odds: {odds_str}")
    else:
        print("  No upcoming soccer markets found")

print("\n\n💡 Summary:")
print("The blockchain data should contain:")
print("- Real-time odds updates")
print("- Market addresses/IDs for trading")
print("- Settlement information")
print("- All active markets on Overtime protocol")
print("\nWe need to ensure we're syncing both API and blockchain data!")