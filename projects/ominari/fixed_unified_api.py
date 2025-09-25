#!/usr/bin/env python3
"""Test fixed unified API endpoint"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import func
from datetime import datetime, timezone
import json

def get_markets_for_dashboard(sport_filter='all', limit=100):
    """Get markets with odds for dashboard display"""
    markets = []
    
    try:
        with db_manager.get_db_session() as db:
            # Query active markets
            query = db.query(Market).filter(Market.is_finished == False)
            
            # Apply sport filter if requested
            if sport_filter and sport_filter != 'all':
                query = query.filter(Market.sport == sport_filter)
            
            # Order by maturity date and limit
            active_markets = query.order_by(Market.maturity_date).limit(limit).all()
            
            print(f"Found {len(active_markets)} active markets")
            
            for market in active_markets:
                # Get odds for this market
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(3).all()
                
                # Group odds by outcome
                home_odds = None
                draw_odds = None
                away_odds = None
                
                for odd in odds:
                    outcome_lower = str(odd.outcome).lower() if odd.outcome else ''
                    if 'home' in outcome_lower or outcome_lower == 'option_1':
                        home_odds = odd.decimal_odds
                    elif 'away' in outcome_lower or outcome_lower == 'option_2':
                        away_odds = odd.decimal_odds
                    elif 'draw' in outcome_lower or 'tie' in outcome_lower or outcome_lower == 'option_3':
                        draw_odds = odd.decimal_odds
                
                # Calculate time until
                now_utc = datetime.now(timezone.utc)
                if market.maturity_date:
                    if market.maturity_date.tzinfo is None:
                        maturity_aware = market.maturity_date.replace(tzinfo=timezone.utc)
                    else:
                        maturity_aware = market.maturity_date
                    
                    delta = maturity_aware - now_utc
                    if delta.total_seconds() > 0:
                        hours = int(delta.total_seconds() // 3600)
                        mins = int((delta.total_seconds() % 3600) // 60)
                        
                        if hours > 24:
                            time_until = f"{hours // 24}d {hours % 24}h"
                            status = "Future"
                        elif hours > 0:
                            time_until = f"{hours}h {mins}m"
                            status = "Starting Soon"
                        else:
                            time_until = f"{mins}m"
                            status = "Imminent"
                    else:
                        time_until = "Started"
                        status = "Live"
                else:
                    time_until = "Unknown"
                    status = "Unknown"
                
                # Build market data
                market_data = {
                    'id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'source': market.source,
                    'maturity_date': market.maturity_date.isoformat() if market.maturity_date else None,
                    'time_until': time_until,
                    'status': status,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'has_odds': bool(home_odds and away_odds)
                }
                
                markets.append(market_data)
                
    except Exception as e:
        print(f"Error getting markets: {e}")
        import traceback
        traceback.print_exc()
    
    return markets

# Test the function
if __name__ == "__main__":
    print("\n🔍 Testing market retrieval for dashboard...")
    
    # Test all sports
    all_markets = get_markets_for_dashboard()
    print(f"\n📊 All sports: {len(all_markets)} markets")
    
    if all_markets:
        print("\n🏆 First 3 markets:")
        for i, m in enumerate(all_markets[:3]):
            print(f"\nMarket {i+1}:")
            print(f"  {m['home_team']} vs {m['away_team']}")
            print(f"  Sport: {m['sport']}")
            print(f"  Time: {m['time_until']} ({m['status']})")
            print(f"  Odds: H={m['home_odds']} D={m['draw_odds']} A={m['away_odds']}")
            print(f"  Has odds: {m['has_odds']}")
    
    # Test soccer filter
    soccer_markets = get_markets_for_dashboard(sport_filter='soccer')
    print(f"\n⚽ Soccer only: {len(soccer_markets)} markets")
    
    # Count markets by sport
    sport_counts = {}
    for m in all_markets:
        sport = m['sport']
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    
    print(f"\n📊 Markets by sport:")
    for sport, count in sorted(sport_counts.items()):
        print(f"  {sport}: {count}")