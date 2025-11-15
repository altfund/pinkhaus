#!/usr/bin/env python3
"""Add a few markets with positive edges for demonstration"""

import os
import sys
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd

with db_manager.get_db_session() as db:
    # Find a few soccer markets to update
    markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport == 'Soccer'
    ).limit(5).all()
    
    print(f"Creating positive edge opportunities on {len(markets)} markets...")
    
    for i, market in enumerate(markets):
        # Create different scenarios
        if i == 0:
            # Scenario 1: Overpriced favorite (value on underdog)
            print(f"\n1. Value on underdog: {market.home_team} vs {market.away_team}")
            odds_values = {
                'home': 1.5,   # Too short for favorite
                'away': 3.8,   # Good value on underdog  
                'draw': 4.2
            }
        elif i == 1:
            # Scenario 2: Draw value
            print(f"\n2. Draw value: {market.home_team} vs {market.away_team}")
            odds_values = {
                'home': 2.1,
                'away': 3.4,
                'draw': 4.0    # High draw odds
            }
        elif i == 2:
            # Scenario 3: Arbitrage opportunity
            print(f"\n3. Arbitrage opportunity: {market.home_team} vs {market.away_team}")
            odds_values = {
                'home': 2.3,
                'away': 3.9,
                'draw': 3.8
            }
        elif i == 3:
            # Scenario 4: Slight positive edge
            print(f"\n4. Small edge: {market.home_team} vs {market.away_team}")
            odds_values = {
                'home': 1.85,
                'away': 4.5,
                'draw': 3.6
            }
        else:
            # Scenario 5: Negative edge (normal market)
            print(f"\n5. Normal market: {market.home_team} vs {market.away_team}")
            odds_values = {
                'home': 1.8,
                'away': 4.2,
                'draw': 3.4
            }
        
        # Calculate edges
        total_prob = sum(1/o for o in odds_values.values())
        print(f"   Total probability: {total_prob:.3f}")
        
        for outcome, odds in odds_values.items():
            fair_prob = (1/odds) / total_prob
            fair_odds = 1 / fair_prob
            edge = ((odds / fair_odds) - 1) * 100
            print(f"   {outcome}: {odds} (edge: {edge:+.2f}%)")
            
            # Update in database
            existing = db.query(Odd).filter(
                Odd.source_id == market.source_id,
                Odd.outcome == outcome
            ).first()
            
            if existing:
                existing.decimal_odds = odds
                existing.updated_at = datetime.now(timezone.utc)
            else:
                new_odd = Odd(
                    source_id=market.source_id,
                    outcome=outcome,
                    decimal_odds=odds,
                    created_at=datetime.now(timezone.utc),
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(new_odd)
    
    db.commit()
    print("\n✅ Added positive edge markets! The paper trading system should pick these up.")
    print("Check your Discord for notifications when trades are placed!")