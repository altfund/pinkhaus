#!/usr/bin/env python3
"""Debug why markets aren't being found"""

import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
from load_env import load_dotenv
load_dotenv()

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd

with db_manager.get_db_session() as db:
    # Get future markets
    now = datetime.now(timezone.utc)
    future_time = now + timedelta(hours=24)
    
    print(f"Current time: {now}")
    print(f"Looking for markets between now and: {future_time}")
    print()
    
    # Find Portland Thorns specifically
    pt_market = db.query(Market).filter(
        Market.home_team == "Portland Thorns FC"
    ).first()
    
    if pt_market:
        print("Portland Thorns market found:")
        print(f"  Home: {pt_market.home_team}")
        print(f"  Away: {pt_market.away_team}")
        print(f"  Sport: {pt_market.sport}")
        print(f"  Maturity date: {pt_market.maturity_date}")
        print(f"  Is future: {pt_market.maturity_date > now}")
        print(f"  Within 24h: {pt_market.maturity_date < future_time}")
        print(f"  Sport in filter: {pt_market.sport in ['Soccer', 'Tennis', 'American Football', 'Basketball']}")
        
        if pt_market.maturity_date <= now:
            print(f"  ❌ Market is in the past! Maturity: {pt_market.maturity_date}, Now: {now}")
        
        # Get odds
        odds = db.query(Odd).filter(Odd.source_id == pt_market.source_id).all()
        if odds:
            print(f"\n  Odds found: {len(odds)}")
            total_prob = 0
            for odd in odds:
                prob = 1/odd.decimal_odds
                total_prob += prob
                print(f"    {odd.outcome}: {odd.decimal_odds} (prob: {prob:.3f})")
            print(f"  Total probability: {total_prob:.3f}")
            if total_prob < 1.0:
                edge = ((1/total_prob) - 1) * 100
                print(f"  ✅ ARBITRAGE! Edge: {edge:.2f}%")
        else:
            print("  ❌ No odds found for this market")
            
    # Count all future markets
    future_markets = db.query(Market).filter(
        Market.maturity_date > now,
        Market.maturity_date < future_time
    ).count()
    
    print(f"\nTotal future markets in next 24h: {future_markets}")
    
    # Count by sport
    sports = db.query(Market.sport, db.func.count(Market.id)).filter(
        Market.maturity_date > now,
        Market.maturity_date < future_time
    ).group_by(Market.sport).all()
    
    print("\nMarkets by sport:")
    for sport, count in sports:
        in_filter = sport in ['Soccer', 'Tennis', 'American Football', 'Basketball']
        print(f"  {sport}: {count} {'✅' if in_filter else '❌ (not in filter)'}")