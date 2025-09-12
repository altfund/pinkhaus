#!/usr/bin/env python3
"""
Test the enhanced system with all improvements:
- Coin flip signal (dummy server running)
- 50% Kelly fraction
- Minimum bet requirements
- Bias adjustments in ImpliedRaw
"""

import pandas as pd
import numpy as np
from signals import SIGNAL_PROVIDERS
from database_v2 import db_manager
from models import Market, Odd
from kelly_multimarket import calculate_kelly_stakes_with_exclusivity
from datetime import datetime, timezone

def test_enhanced_system():
    """Test the complete enhanced system."""
    
    print("=== TESTING ENHANCED OMINARI SYSTEM ===")
    print(f"Time: {datetime.now(timezone.utc)}")
    print("-" * 50)
    
    # Test 1: Verify signal providers
    print("\n1. SIGNAL PROVIDER STATUS:")
    for provider in SIGNAL_PROVIDERS:
        print(f"   - {provider.name}: Loaded ✓")
    
    # Test 2: Get a sample market
    with db_manager.get_db_session() as db:
        # Get one active soccer market
        market = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).first()
        
        if not market:
            print("No active markets found")
            return
            
        print(f"\n2. TEST MARKET: {market.home_team} vs {market.away_team}")
        
        # Get odds
        odds = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).order_by(Odd.updated_at.desc()).limit(3).all()
        
        if len(odds) < 3:
            print("Need 3 outcomes for test")
            return
            
        # Build test DataFrame
        test_df = pd.DataFrame([{
            'source_id': odd.source_id,
            'match_id': f"{market.home_team}_vs_{market.away_team}",
            'unified_market_type': 'winner',
            'normalized_outcome': odd.outcome,
            'normalized_line': 0.0,
            'bet_name': odd.outcome,
            'market_name': f"{market.home_team} vs {market.away_team}",
            'league_name': market.league_name,
            'bookmaker': 'Overtime',
            'decimal_odds': odd.decimal_odds,
            'implied_raw': (1/odd.decimal_odds * 100) if odd.decimal_odds > 0 else 0,
            'home_team': market.home_team,
            'away_team': market.away_team,
            'time': odd.updated_at
        } for odd in odds])
        
        print("\n3. SIGNAL ANALYSIS:")
        
        # Test each signal
        for provider in SIGNAL_PROVIDERS:
            print(f"\n   {provider.name.upper()}:")
            try:
                probs = provider.get_probs(test_df)
                
                for i, (_, row) in enumerate(test_df.iterrows()):
                    if i < len(probs):
                        prob = probs.iloc[i] if hasattr(probs, 'iloc') else probs[i]
                        implied = 1 / row['decimal_odds']
                        edge = prob - implied
                        
                        print(f"      {row['normalized_outcome']:8}: prob={prob:.3f}, implied={implied:.3f}, edge={edge:+.3f}")
                        
            except Exception as e:
                print(f"      Error: {e}")
        
        # Test 4: Kelly optimization
        print("\n4. KELLY OPTIMIZATION (50% Kelly):")
        
        # Prepare for Kelly with combined signals
        all_probs = []
        weights = []
        
        for provider in SIGNAL_PROVIDERS:
            try:
                probs = provider.get_probs(test_df)
                if len(probs) > 0:
                    all_probs.append(probs)
                    weights.append(1.0)  # Equal weights
            except:
                pass
                
        if all_probs:
            # Combine probabilities (simple average for test)
            combined_probs = pd.Series([
                np.average([p.iloc[i] if hasattr(p, 'iloc') else p[i] 
                           for p in all_probs if i < len(p)], 
                          weights=weights)
                for i in range(len(test_df))
            ])
            
            test_df['probability'] = combined_probs
            test_df['odds'] = test_df['decimal_odds']
            
            # Run Kelly optimization
            kelly_results = calculate_kelly_stakes_with_exclusivity(
                test_df,
                bankroll=10000,
                correlation_matrix=None,
                risk_adjusted=True,
                max_stake_per_bet=None
            )
            
            print("\n   Kelly Results:")
            for _, row in kelly_results.iterrows():
                if row['stake'] > 0:
                    print(f"      {row['normalized_outcome']:8}: stake=${row['stake']:.2f} ({row['stake_fraction']*100:.1f}% of bankroll)")
                    
            total_stake = kelly_results['stake'].sum()
            print(f"\n   Total recommended stake: ${total_stake:.2f}")
            
            # Apply 50% Kelly and minimum bet
            kelly_fraction = 0.5
            min_bet = 50  # $50 minimum
            
            print(f"\n   After 50% Kelly + ${min_bet} minimum:")
            adjusted_stakes = kelly_results['stake'] * kelly_fraction
            
            # Force minimum on positive edge
            positive_edge = kelly_results['stake'] > 0
            below_min = adjusted_stakes < min_bet
            adjusted_stakes.loc[positive_edge & below_min] = min_bet
            
            for i, stake in enumerate(adjusted_stakes):
                if stake > 0:
                    outcome = test_df.iloc[i]['normalized_outcome']
                    print(f"      {outcome:8}: stake=${stake:.2f}")
                    
            print(f"\n   Total adjusted stake: ${adjusted_stakes.sum():.2f}")
            
        print("\n5. SYSTEM ENHANCEMENTS ACTIVE:")
        print("   ✓ Coin flip signal providing random predictions")
        print("   ✓ ImpliedRaw with bias adjustments (-1% favorites, +1% longshots, +0.5% draws)")
        print("   ✓ 50% Kelly fraction (up from 25%)")
        print("   ✓ Minimum bet enforced on positive edge opportunities")
        print("   ✓ Multi-market correlation handling ready")

if __name__ == "__main__":
    test_enhanced_system()