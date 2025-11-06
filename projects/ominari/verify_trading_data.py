#!/usr/bin/env python3
"""Verify trading system is using real data"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
from paper_trading_postgres_integrated import PaperTradingSessionManager
from edge_calculator import EdgeCalculator

# Check markets being analyzed
with db_manager.get_db_session() as db:
    markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=48),
        Market.is_finished == False
    ).order_by(Market.maturity_date).limit(10).all()
    
    print(f"🏟️ MARKETS AVAILABLE FOR TRADING:")
    print("=" * 80)
    
    market_data = []
    for market in markets:
        # Get odds
        odds = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).order_by(Odd.updated_at.desc()).limit(3).all()
        
        if len(odds) >= 3:
            home_odd = next((o for o in odds if 'home' in o.outcome.lower()), None)
            draw_odd = next((o for o in odds if 'draw' in o.outcome.lower()), None)
            away_odd = next((o for o in odds if 'away' in o.outcome.lower()), None)
            
            if home_odd and draw_odd and away_odd:
                print(f"\n{market.home_team} vs {market.away_team}")
                print(f"  Sport: {market.sport}, League: {market.league_name}")
                print(f"  Kickoff: {market.maturity_date}")
                print(f"  Odds: Home {home_odd.decimal_odds} | Draw {draw_odd.decimal_odds} | Away {away_odd.decimal_odds}")
                
                market_data.append({
                    'market_id': market.source_id,
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'sport': market.sport,
                    'maturity_date': market.maturity_date,
                    'home_odds': float(home_odd.decimal_odds),
                    'draw_odds': float(draw_odd.decimal_odds),
                    'away_odds': float(away_odd.decimal_odds),
                    'source': market.source
                })
    
    if market_data:
        print(f"\n\n🎯 CALCULATING EDGES FOR {len(market_data)} MARKETS:")
        print("=" * 80)
        
        edge_calculator = EdgeCalculator()
        signals = edge_calculator.calculate_edges(market_data)
        
        for i, signal in enumerate(signals):
            market = market_data[i]
            edges = signal.get('edge', {})
            
            print(f"\n{market['home_team']} vs {market['away_team']}:")
            print(f"  Home edge: {edges.get('home', 0):.1f}%")
            print(f"  Draw edge: {edges.get('draw', 0):.1f}%")  
            print(f"  Away edge: {edges.get('away', 0):.1f}%")
            
            # Show if any positive edges
            positive_edges = [(k, v) for k, v in edges.items() if v > 0]
            if positive_edges:
                print(f"  ✅ POSITIVE EDGES FOUND: {positive_edges}")
    
    # Check current session
    print(f"\n\n📊 CURRENT TRADING SESSION:")
    print("=" * 80)
    
    sm = PaperTradingSessionManager()
    session_id = sm.get_current_session()
    positions = sm.get_positions(session_id)
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    
    print(f"Session ID: {session_id}")
    print(f"Open Positions: {len(open_positions)}")
    
    if open_positions:
        print("\nCurrent positions:")
        for pos in open_positions:
            print(f"  • {pos.get('match_id', 'Unknown')}: {pos.get('bet_type', 'Unknown')} @ {pos['odds']} - ${pos['stake']:.2f}")