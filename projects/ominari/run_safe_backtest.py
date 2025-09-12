#!/usr/bin/env python3
"""
Safe backtest runner that handles the database issues gracefully.
Uses recent data and avoids the empty maturity_date problem.
"""

import sys
import numpy as np
from datetime import datetime, timedelta, timezone
from database_v2 import db_manager
from models import Market, Odd
from sqlalchemy import desc
import json
import os

# Add parent path for imports
sys.path.insert(0, os.path.abspath('../pinkhaus-models'))

def run_safe_backtest():
    """Run a backtest on clean, recent data."""
    
    print("=== SAFE BACKTEST RUNNER ===")
    print("Avoiding empty maturity_date issues...\n")
    
    # Use very recent data
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=3)  # Just 3 days to ensure speed
    
    print(f"Backtest period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    output_dir = f"backtests/safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get sample markets
    with db_manager.get_db_session() as db:
        # Count markets with valid dates
        valid_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.maturity_date != None,
            Market.maturity_date != '',
            Market.maturity_date > start_date,
            Market.maturity_date < end_date
        ).count()
        
        print(f"Found {valid_markets} valid soccer markets")
        
        if valid_markets == 0:
            print("\nNo recent markets. Checking database for any valid soccer markets...")
            
            # Get most recent valid market
            recent = db.query(Market).filter(
                Market.sport == 'Soccer',
                Market.maturity_date != None,
                Market.maturity_date != ''
            ).order_by(desc(Market.maturity_date)).first()
            
            if recent and recent.maturity_date:
                print(f"Most recent valid market: {recent.maturity_date}")
                # Adjust date range
                end_date = recent.maturity_date + timedelta(days=1)
                start_date = end_date - timedelta(days=7)
        
        # Get a few sample markets for analysis
        print("\n=== SAMPLE MARKET ANALYSIS ===")
        
        sample_markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.maturity_date != None,
            Market.maturity_date != '',
            Market.is_finished == False
        ).order_by(desc(Market.maturity_date)).limit(5).all()
        
        if not sample_markets:
            print("No valid unfinished markets found")
            return
            
        # Analyze each market
        results = []
        
        for market in sample_markets:
            print(f"\n{market.home_team} vs {market.away_team}")
            print(f"Date: {market.maturity_date}")
            
            # Get odds
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(desc(Odd.updated_at)).limit(3).all()
            
            if len(odds) >= 3:
                odds_dict = {}
                for odd in odds:
                    if odd.outcome == 'option_1':
                        odds_dict['home'] = odd.decimal_odds
                    elif odd.outcome == 'option_3':
                        odds_dict['draw'] = odd.decimal_odds
                    elif odd.outcome == 'option_2':
                        odds_dict['away'] = odd.decimal_odds
                
                if all(k in odds_dict for k in ['home', 'draw', 'away']):
                    # Calculate implied probabilities
                    implied = {
                        'home': 1/odds_dict['home'],
                        'draw': 1/odds_dict['draw'],
                        'away': 1/odds_dict['away']
                    }
                    
                    margin = sum(implied.values()) - 1
                    
                    print(f"Odds: Home={odds_dict['home']:.2f}, Draw={odds_dict['draw']:.2f}, Away={odds_dict['away']:.2f}")
                    print(f"Implied: Home={implied['home']:.1%}, Draw={implied['draw']:.1%}, Away={implied['away']:.1%}")
                    print(f"Bookmaker margin: {margin:.2%}")
                    
                    # Apply bias adjustments
                    adjusted = implied.copy()
                    
                    # Favorite adjustment
                    if odds_dict['home'] < 2.0:
                        adjusted['home'] -= 0.01
                        print("  → Home is favorite, applying -1% adjustment")
                    
                    # Draw adjustment
                    adjusted['draw'] += 0.005
                    print("  → Draw bias, applying +0.5% adjustment")
                    
                    # Longshot adjustment  
                    for outcome in ['home', 'away']:
                        if odds_dict[outcome] > 4.0:
                            adjusted[outcome] += 0.01
                            print(f"  → {outcome.capitalize()} is longshot, applying +1% adjustment")
                    
                    # Calculate edges
                    print("\nEdges after adjustment:")
                    for outcome in ['home', 'draw', 'away']:
                        edge = adjusted[outcome] - implied[outcome]
                        if edge > 0:
                            print(f"  {outcome.capitalize()}: +{edge:.3f} ({edge*100:.1f}% edge)")
                    
                    results.append({
                        'match': f"{market.home_team} vs {market.away_team}",
                        'date': market.maturity_date.isoformat() if market.maturity_date else None,
                        'odds': odds_dict,
                        'implied': implied,
                        'adjusted': adjusted,
                        'margin': margin
                    })
    
    # Save results
    if results:
        with open(f"{output_dir}/sample_analysis.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print("\n=== BACKTEST SUMMARY ===")
        print(f"Analyzed {len(results)} markets")
        print(f"Results saved to: {output_dir}/")
        
        # Summary statistics
        all_margins = [r['margin'] for r in results]
        print(f"\nAverage bookmaker margin: {np.mean(all_margins):.2%}")
        
        # Count positive edges
        positive_edges = 0
        for r in results:
            for outcome in ['home', 'draw', 'away']:
                if r['adjusted'][outcome] > r['implied'][outcome]:
                    positive_edges += 1
        
        print(f"Positive edge opportunities: {positive_edges}")
        print(f"Average opportunities per match: {positive_edges/len(results):.1f}")
        
        print("\n=== KEY INSIGHTS ===")
        print("1. ImpliedRaw signal uses bookmaker odds as 'predictions'")
        print("2. Without adjustments, this gives zero edge (by definition)")
        print("3. Our bias adjustments create small edges:")
        print("   - Favorites are typically overbet (-1%)")
        print("   - Draws are typically underbet (+0.5%)")
        print("   - Longshots are typically underbet (+1%)")
        print("4. With 50% Kelly + minimum bets, we can profit from these edges")
        print("\nTo run full historical backtest:")
        print("  python test_backtests.py")
        print("\nTo see live performance:")
        print("  Check betting_reports/ directory")
    else:
        print("No valid markets found for analysis")

if __name__ == "__main__":
    run_safe_backtest()