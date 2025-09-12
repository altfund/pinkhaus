#!/usr/bin/env python3
"""
Analysis of why ImpliedRaw signal shows no edge.
Demonstrates the fundamental issue with using bookmaker odds as predictions.
"""

import pandas as pd
from database_v2 import db_manager
from models import Market, Odd
from signals import ImpliedRawSignal

def analyze_implied_raw_signal():
    """Analyze why ImpliedRaw signal produces no betting edge."""
    
    print("=== Analysis: Why ImpliedRaw Signal Shows No Edge ===\n")
    
    # Get sample market data
    with db_manager.get_db_session() as db:
        # Get one recent soccer market with all 3 outcomes
        market = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.is_finished == False
        ).first()
        
        if not market:
            print("No active markets found")
            return
            
        odds = db.query(Odd).filter(
            Odd.source_id == market.source_id
        ).order_by(Odd.updated_at.desc()).limit(3).all()
        
        if len(odds) < 3:
            print("Need 3 outcomes for analysis")
            return
            
        print(f"Analyzing market: {market.home_team} vs {market.away_team}")
        print(f"League: {market.league_name}")
        print(f"Match time: {market.maturity_date}\n")
        
        # Build DataFrame
        market_df = pd.DataFrame([{
            'source_id': odd.source_id,
            'outcome': odd.outcome,
            'decimal_odds': odd.decimal_odds,
            'implied_raw': (1/odd.decimal_odds * 100) if odd.decimal_odds > 0 else 0,
            'home_team': market.home_team,
            'away_team': market.away_team
        } for odd in odds])
        
        print("Raw Odds Data:")
        for _, row in market_df.iterrows():
            print(f"  {row['outcome']:10}: {row['decimal_odds']:.3f} odds = {row['implied_raw']:.1f}% implied")
        
        # Show bookmaker margin
        total_prob = sum(1/row['decimal_odds'] for _, row in market_df.iterrows())
        margin = (total_prob - 1) * 100
        print(f"\nBookmaker margin: {margin:.2f}%")
        print(f"Sum of implied probabilities: {total_prob:.3f} (should be 1.0 for fair odds)")
        
        # Test ImpliedRaw signal
        signal = ImpliedRawSignal()
        signal_probs = signal.get_probs(market_df)
        
        print("\n=== ImpliedRaw Signal Analysis ===")
        print("Signal simply converts implied_raw (%) to probability:")
        
        for i, (_, row) in enumerate(market_df.iterrows()):
            prob = signal_probs.iloc[i]
            implied = 1 / row['decimal_odds']
            edge = prob - implied
            expected_value = prob * (row['decimal_odds'] - 1) - (1 - prob)
            
            print(f"  {row['outcome']:10}: signal={prob:.3f}, implied={implied:.3f}")
            print(f"                edge={edge:.3f}, EV={expected_value:.3f}")
        
        print("\n=== Why No Edge? ===")
        print("1. ImpliedRaw signal just uses bookmaker's own odds")
        print("2. Bookmaker odds already include margin/vig")  
        print("3. Signal probability ≈ Implied probability")
        print("4. Therefore: Edge ≈ 0 for all outcomes")
        print("5. Kelly system correctly identifies no positive expected value")
        
        print("\n=== What Creates Edge? ===")
        print("1. External predictions that differ from bookmaker odds")
        print("2. Arbitrage opportunities between bookmakers") 
        print("3. Information advantages (injuries, weather, etc.)")
        print("4. Market inefficiencies or mispricing")
        
        print("\n=== Signal Provider Status ===")
        print("• ImpliedRaw: Working but no edge by design")
        print("• ExternalGrpc: Connection refused (offline)")
        print("• Grant LLM: Connection refused (offline)")
        print("\nRecommendation: Fix gRPC services or add new predictive signals")
        
        return {
            'market_name': f"{market.home_team} vs {market.away_team}",
            'margin': margin,
            'signal_probs': signal_probs.tolist(),
            'implied_probs': [1/row['decimal_odds'] for _, row in market_df.iterrows()],
            'expected_values': [
                prob * (row['decimal_odds'] - 1) - (1 - prob) 
                for prob, (_, row) in zip(signal_probs, market_df.iterrows())
            ]
        }

def compare_kelly_vs_carver():
    """Compare our Kelly system vs pure Carver framework."""
    
    print("\n=== Kelly vs Carver Framework Comparison ===")
    
    print("\n🏆 Why Keep Soccer-Specific Kelly Logic:")
    print("1. Mutual Exclusivity: Soccer has exactly 3 exclusive outcomes")
    print("   - Kelly handles Home/Draw/Away constraints perfectly") 
    print("   - Carver assumes independent futures markets")
    print("")
    print("2. Correlation Structure: Soccer outcomes are perfectly correlated")
    print("   - If Home wins, Draw and Away lose (correlation = -1)")
    print("   - Kelly optimization accounts for this directly")
    print("   - Carver correlation matrix would be complex to specify")
    print("")
    print("3. Log Utility: Kelly maximizes log growth for betting")
    print("   - Optimal for discrete win/lose outcomes")
    print("   - Carver optimizes for volatility targeting (futures)")
    print("")
    print("4. Probability Constraints: Soccer probabilities must sum to 1.0")
    print("   - Kelly handles this with exclusivity constraints")
    print("   - Carver doesn't enforce probability conservation")
    
    print("\n✨ What We Can Adopt from Carver:")
    print("1. Forecast Combination: Multiple signals → single forecast")
    print("2. Diversification Multipliers: Account for signal correlation")
    print("3. Forecast Scaling: Convert to standardized [-20, +20] range")
    print("4. No Arbitrary Cutoffs: Let optimization decide everything")
    
    print("\n🎯 Our Hybrid Approach:")
    print("1. Use Carver forecast combination for signal blending")
    print("2. Keep Kelly optimization for soccer mutual exclusivity")
    print("3. Remove all arbitrary edge/margin cutoffs")
    print("4. Show ALL opportunities to risk management system")

if __name__ == "__main__":
    analysis = analyze_implied_raw_signal()
    compare_kelly_vs_carver()
    
    print("\n=== Bottom Line ===")
    print("ImpliedRaw signal has no edge because it uses bookmaker odds as predictions.")
    print("To generate edge, we need external predictions that outperform the market.")
    print("The Kelly system is working correctly - it won't bet when EV ≤ 0.")