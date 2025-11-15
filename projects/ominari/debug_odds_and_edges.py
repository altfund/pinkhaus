#!/usr/bin/env python3
"""Debug odds and edge calculations to see what's really happening"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd
from datetime import datetime, timezone, timedelta
import random

with db_manager.get_db_session() as db:
    # Get some recent markets
    markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport == 'Soccer'
    ).limit(5).all()
    
    print("Analyzing market odds and edges...\n")
    
    for market in markets:
        print(f"\n{'='*60}")
        print(f"Market: {market.home_team} vs {market.away_team}")
        print(f"Source: {market.source}")
        
        # Get odds for this market
        odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
        
        if len(odds) < 2:
            print("  Not enough odds")
            continue
            
        print(f"\nRaw odds from database:")
        total_implied_prob = 0
        for odd in odds:
            implied_prob = 1 / odd.decimal_odds
            total_implied_prob += implied_prob
            print(f"  {odd.outcome}: {odd.decimal_odds:.3f} (implied prob: {implied_prob:.3%})")
        
        print(f"\nTotal implied probability: {total_implied_prob:.3%}")
        overround = (total_implied_prob - 1) * 100
        print(f"Overround/Margin: {overround:.2f}%")
        
        # Current edge calculation (the problematic one)
        print(f"\nCurrent edge calculation (redistributing margin):")
        for odd in odds:
            fair_prob = (1/odd.decimal_odds) / total_implied_prob
            fair_odds = 1 / fair_prob
            edge = ((odd.decimal_odds / fair_odds) - 1) * 100
            print(f"  {odd.outcome}: Fair odds={fair_odds:.3f}, Edge={edge:.2f}%")
            
        # Alternative edge calculation methods
        print(f"\nAlternative calculations:")
        
        # Method 1: Remove margin proportionally
        margin_per_outcome = overround / len(odds) / 100
        print(f"\nMethod 1 - Remove margin equally:")
        for odd in odds:
            implied_prob = 1 / odd.decimal_odds
            fair_prob = implied_prob - margin_per_outcome/len(odds)
            fair_odds = 1 / fair_prob
            edge = ((odd.decimal_odds / fair_odds) - 1) * 100
            print(f"  {odd.outcome}: Fair prob={fair_prob:.3%}, Fair odds={fair_odds:.3f}, Edge={edge:.2f}%")
            
        # Method 2: Use external reference (e.g., pinnacle closing odds)
        print(f"\nMethod 2 - Sharp book reference (simulated):")
        # In reality, we'd fetch from Pinnacle or another sharp book
        # For now, simulate with slightly different margins
        for odd in odds:
            # Simulate a sharp book with lower margin
            sharp_margin = 0.02  # 2% margin for sharp books
            implied_prob = 1 / odd.decimal_odds
            sharp_implied_prob = implied_prob * (1 + sharp_margin) / total_implied_prob
            sharp_odds = 1 / sharp_implied_prob
            edge = ((odd.decimal_odds / sharp_odds) - 1) * 100
            print(f"  {odd.outcome}: Sharp odds={sharp_odds:.3f}, Edge={edge:.2f}%")
            
        # Method 3: Look for arbitrage opportunities
        print(f"\nMethod 3 - Cross-market arbitrage check:")
        # Check if we can find different odds from different sources
        other_odds = db.query(Odd).join(Market).filter(
            Market.home_team == market.home_team,
            Market.away_team == market.away_team,
            Market.source != market.source,
            Market.maturity_date > datetime.now(timezone.utc)
        ).all()
        
        if other_odds:
            print(f"  Found {len(other_odds)} odds from other sources")
            for other in other_odds[:3]:
                print(f"    {other.outcome} @ {other.decimal_odds} (source: {other.source_id[:20]}...)")
        else:
            print(f"  No cross-market opportunities found")
            
    # Check variety in odds
    print(f"\n{'='*60}")
    print("Checking odds variety across all markets...")
    all_odds = db.query(Odd.decimal_odds).join(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport == 'Soccer'
    ).limit(100).all()
    
    unique_odds = set(o[0] for o in all_odds)
    print(f"Found {len(unique_odds)} unique odds values in {len(all_odds)} total odds")
    print(f"Sample unique odds: {sorted(list(unique_odds))[:10]}")
    
    # Check if all markets have the same total probability
    print(f"\nChecking total probabilities across markets...")
    market_probs = []
    for market in db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.sport == 'Soccer'
    ).limit(10):
        odds = db.query(Odd).filter(Odd.source_id == market.source_id).all()
        if len(odds) >= 2:
            total_prob = sum(1/o.decimal_odds for o in odds)
            market_probs.append(total_prob)
            
    if market_probs:
        print(f"Total probabilities: min={min(market_probs):.3f}, max={max(market_probs):.3f}, avg={sum(market_probs)/len(market_probs):.3f}")
        unique_probs = set(round(p, 3) for p in market_probs)
        print(f"Unique total probabilities: {len(unique_probs)}")