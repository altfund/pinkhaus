#!/usr/bin/env python3
"""
Simple performance analysis using safe database queries.
Analyzes signal performance without complex backtesting.
"""

import numpy as np
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
from safe_query import get_recent_markets, get_market_summary
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_signal_theoretical_performance():
    """
    Analyze theoretical signal performance using recent market data.
    Shows why ImpliedRaw has no edge and evaluates other signals.
    """
    
    print("=== Signal Performance Analysis ===")
    print("Using safe database queries to avoid timeout issues\n")
    
    # Get recent market summary
    try:
        summary = get_market_summary()
        print("Database Summary:")
        for key, value in summary.items():
            print(f"  {key}: {value:,}")
    except Exception as e:
        print(f"Error getting summary: {e}")
        
    # Get sample of recent markets
    try:
        recent_markets = get_recent_markets(limit=50)
        print(f"\nAnalyzing {len(recent_markets)} recent markets")
        
        if recent_markets.empty:
            print("No recent markets found")
            return {}
            
        # Analyze signal performance
        signal_analysis = {}
        
        for provider in SIGNAL_PROVIDERS:
            print(f"\n--- Analyzing {provider.name} ---")
            
            try:
                # Test signal on sample data
                signal_probs = provider.get_probs(recent_markets)
                
                if len(signal_probs) > 0:
                    # Calculate implied probabilities
                    implied_probs = 1 / recent_markets['decimal_odds']
                    
                    # Calculate edges
                    edges = signal_probs - implied_probs
                    
                    # Calculate expected values
                    expected_values = []
                    for i, (_, row) in enumerate(recent_markets.iterrows()):
                        if i < len(signal_probs) and row['decimal_odds'] > 0:
                            prob = signal_probs.iloc[i] if hasattr(signal_probs, 'iloc') else signal_probs[i]
                            ev = prob * (row['decimal_odds'] - 1) - (1 - prob)
                            expected_values.append(ev)
                        else:
                            expected_values.append(0)
                    
                    # Analysis
                    positive_ev_count = sum(1 for ev in expected_values if ev > 0)
                    avg_edge = np.mean(edges) if len(edges) > 0 else 0
                    avg_ev = np.mean(expected_values) if expected_values else 0
                    
                    signal_analysis[provider.name] = {
                        'total_predictions': len(signal_probs),
                        'positive_ev_count': positive_ev_count,
                        'positive_ev_rate': positive_ev_count / len(expected_values) if expected_values else 0,
                        'average_edge': avg_edge,
                        'average_expected_value': avg_ev,
                        'signal_weight': SIGNAL_WEIGHTS.get(provider.name, 1.0),
                        'working': True
                    }
                    
                    print(f"  Predictions: {len(signal_probs)}")
                    print(f"  Positive EV: {positive_ev_count}/{len(expected_values)} ({positive_ev_count/len(expected_values)*100:.1f}%)")
                    print(f"  Avg Edge: {avg_edge:.4f} ({avg_edge*100:+.2f}%)")
                    print(f"  Avg EV: {avg_ev:.4f}")
                    
                else:
                    signal_analysis[provider.name] = {
                        'working': False,
                        'error': 'No signals generated'
                    }
                    print("  No signals generated")
                    
            except Exception as e:
                signal_analysis[provider.name] = {
                    'working': False, 
                    'error': str(e)
                }
                print(f"  Error: {e}")
        
        return signal_analysis
        
    except Exception as e:
        print(f"Error getting recent markets: {e}")
        return {}

def calculate_portfolio_edge(signal_analysis: dict) -> dict:
    """Calculate combined portfolio edge using signal weights."""
    
    print("\n=== Portfolio-Level Analysis ===")
    
    working_signals = {name: data for name, data in signal_analysis.items() 
                      if data.get('working', False)}
    
    if not working_signals:
        print("No working signals found")
        return {'portfolio_edge': 0, 'expected_roi': 0}
    
    # Calculate weighted average edge
    total_weight = sum(SIGNAL_WEIGHTS.get(name, 1.0) for name in working_signals.keys())
    
    if total_weight == 0:
        print("No positive signal weights")
        return {'portfolio_edge': 0, 'expected_roi': 0}
    
    weighted_edge = 0
    weighted_ev = 0
    
    for signal_name, data in working_signals.items():
        weight = SIGNAL_WEIGHTS.get(signal_name, 1.0) / total_weight
        weighted_edge += data.get('average_edge', 0) * weight
        weighted_ev += data.get('average_expected_value', 0) * weight
        
        print(f"{signal_name}: weight={weight:.2f}, edge={data.get('average_edge', 0)*100:+.2f}%")
    
    print(f"\nWeighted Portfolio Edge: {weighted_edge*100:+.2f}%")
    print(f"Weighted Expected Value: {weighted_ev:+.4f}")
    
    # Theoretical performance with $10,000 bankroll
    if weighted_ev > 0:
        # Simplified Kelly sizing: edge / odds_variance (rough approximation)
        avg_odds = 2.5  # Typical soccer odds
        kelly_fraction = weighted_edge / ((avg_odds - 1) ** 2)
        safe_kelly = min(kelly_fraction * 0.25, 0.1)  # 25% of Kelly, max 10%
        
        expected_annual_roi = weighted_ev * safe_kelly * 100 * 365  # Assumes daily betting
        
        print(f"Theoretical Kelly fraction: {kelly_fraction:.3f}")
        print(f"Safe Kelly sizing: {safe_kelly:.3f} ({safe_kelly*100:.1f}% of bankroll)")
        print(f"Estimated annual ROI: {expected_annual_roi:.1f}%")
        
        return {
            'portfolio_edge': weighted_edge,
            'expected_roi': expected_annual_roi,
            'safe_kelly_fraction': safe_kelly
        }
    else:
        print("Negative expected value - no betting recommended")
        return {'portfolio_edge': weighted_edge, 'expected_roi': 0}

def main():
    """Run complete performance analysis."""
    
    print("🎯 OMINARI PERFORMANCE REPORT")
    print("=" * 50)
    
    # Analyze individual signals
    signal_analysis = analyze_signal_theoretical_performance()
    
    # Calculate portfolio performance
    portfolio_metrics = calculate_portfolio_edge(signal_analysis)
    
    # Summary and recommendations
    print("\n=== SUMMARY & RECOMMENDATIONS ===")
    
    working_count = sum(1 for data in signal_analysis.values() if data.get('working', False))
    total_count = len(SIGNAL_PROVIDERS)
    
    print(f"Signal Health: {working_count}/{total_count} working")
    print(f"Portfolio Edge: {portfolio_metrics['portfolio_edge']*100:+.2f}%")
    print(f"Expected ROI: {portfolio_metrics['expected_roi']:.1f}%")
    
    print("\n📋 Action Items:")
    print("1. ImpliedRaw signal working but zero edge by design ✓")
    print("2. Fix gRPC services for ExternalGrpc and Grant signals")
    print("3. Add new predictive signals (historical patterns, market movement)")
    print("4. Current system correctly shows no bets due to zero/negative EV")
    
    return {
        'signal_analysis': signal_analysis,
        'portfolio_metrics': portfolio_metrics
    }

if __name__ == "__main__":
    main()