#!/usr/bin/env python3
"""Test the position formatting"""

import os
os.environ['PG_PORT'] = '5999'

from paper_trading_postgres_integrated import PaperTradingSessionManager
from database_v2 import db_manager
from models import Market
import json

manager = PaperTradingSessionManager()
session_id = manager.get_current_session()

if session_id:
    positions = manager.get_positions(session_id)
    open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
    
    print(f"Found {len(open_positions)} open positions")
    
    if open_positions:
        # Show first position raw format
        print("\nFirst position raw format:")
        pos = open_positions[0]
        for key, value in pos.items():
            print(f"  {key}: {value}")
        
        # Test formatting
        formatted_open = {}
        for i, pos in enumerate(open_positions[:3]):  # Just first 3
            # Get market info if team names are empty
            market_name = f"{pos.get('home_team', '')} vs {pos.get('away_team', '')}"
            
            if market_name == " vs " or market_name.strip() == "vs":  # Empty names
                print(f"\nPosition {i} has empty team names, looking up from database...")
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == pos['match_id']
                    ).first()
                    if market:
                        market_name = f"{market.home_team} vs {market.away_team}"
                        print(f"  Found: {market_name}")
                    else:
                        print(f"  No market found for {pos['match_id']}")
            
            formatted_open[str(i)] = {
                'market_name': market_name,
                'outcome': pos.get('bet_on', ''),
                'total_stake': float(pos.get('stake', 0)),
                'avg_odds': float(pos.get('odds', 0)),
                'current_value': float(pos.get('stake', 0)),
                'pnl': 0,
                'status': 'open'
            }
        
        print("\nFormatted positions:")
        print(json.dumps(formatted_open, indent=2))