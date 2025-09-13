#!/usr/bin/env python3
"""Check why certain markets are not found in the database."""

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from database_v2 import db_manager
from models import Market
from paper_trading_sessions import PaperTradingSessionManager

# Load paper trading sessions
session_manager = PaperTradingSessionManager()
current_session = session_manager.get_current_session()

if current_session:
    print("CHECKING PAPER TRADING POSITIONS")
    print("=" * 80)
    
    # Get closed positions without results
    closed_positions = current_session.get('closed_positions', [])
    print(f"Total closed positions: {len(closed_positions)}")
    
    # Look for positions with 0 P&L (likely missing results)
    positions_without_results = [pos for pos in closed_positions if pos.get('pnl', 0) == 0]
    print(f"Positions with 0 P&L: {len(positions_without_results)}")
    
    if positions_without_results:
        print("\nExamples of positions without results:")
        print("-" * 80)
        
        with db_manager.get_db_session() as db:
            for pos in positions_without_results[:5]:
                market_id = pos.get('market_id')
                bet_name = pos.get('bet_name', 'Unknown')
                
                print(f"\nPosition: {bet_name}")
                print(f"Market ID: {market_id}")
                print(f"Entry time: {pos.get('entry_time', 'Unknown')}")
                
                # Try to find the market
                if market_id:
                    # Direct ID search
                    market = db.query(Market).filter(Market.source_id == market_id).first()
                    
                    if market:
                        print(f"✓ Market found!")
                        print(f"  Teams: {market.home_team} vs {market.away_team}")
                        print(f"  Finished: {market.is_finished}")
                        print(f"  Score: {market.home_score}-{market.away_score}")
                        print(f"  Result: {market.resolved_outcome}")
                    else:
                        print(f"✗ Market NOT found with ID: {market_id}")
                        
                        # Try to extract team names from bet_name
                        if ' vs ' in bet_name:
                            parts = bet_name.split(' vs ')
                            if len(parts) == 2:
                                team1, team2 = parts
                                
                                # Search by team names
                                print(f"  Searching by teams: {team1} vs {team2}")
                                
                                # Try home vs away
                                matches = db.query(Market).filter(
                                    Market.home_team.like(f'%{team1}%'),
                                    Market.away_team.like(f'%{team2}%')
                                ).limit(5).all()
                                
                                if not matches:
                                    # Try away vs home  
                                    matches = db.query(Market).filter(
                                        Market.home_team.like(f'%{team2}%'),
                                        Market.away_team.like(f'%{team1}%')
                                    ).limit(5).all()
                                
                                if matches:
                                    print(f"  Found {len(matches)} similar markets:")
                                    for m in matches[:2]:
                                        print(f"    - {m.source_id}: {m.home_team} vs {m.away_team} ({m.maturity_date})")
                                else:
                                    print(f"  No markets found with these teams")
                
    # Check how old these positions are
    print("\n" + "=" * 80)
    print("AGE ANALYSIS")
    print("=" * 80)
    
    oldest_time = None
    newest_time = None
    
    for pos in closed_positions:
        if 'entry_time' in pos:
            entry_time = datetime.fromisoformat(pos['entry_time'].replace('Z', '+00:00'))
            
            if oldest_time is None or entry_time < oldest_time:
                oldest_time = entry_time
            if newest_time is None or entry_time > newest_time:
                newest_time = entry_time
    
    if oldest_time:
        oldest_age = datetime.now(timezone.utc) - oldest_time
        newest_age = datetime.now(timezone.utc) - newest_time
        
        print(f"\nOldest position: {oldest_time}")
        print(f"Age: {oldest_age.days} days, {oldest_age.seconds//3600} hours")
        
        print(f"\nNewest position: {newest_time}")
        print(f"Age: {newest_age.days} days, {newest_age.seconds//3600} hours")
        
        # Recommendation
        if oldest_age.total_seconds() > 6 * 3600:
            hours_needed = int(oldest_age.total_seconds() / 3600) + 1
            print(f"\n⚠️  RECOMMENDATION:")
            print(f"Run extended catch-up to cover all positions:")
            print(f"  python results_catchup_service.py --mode once --lookback {hours_needed}")
else:
    print("No active paper trading session found!")