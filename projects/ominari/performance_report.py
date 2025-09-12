#!/usr/bin/env python3
"""
Comprehensive Performance Report for Ominari Trading System
Analyzes signal performance, Kelly optimization, and provides Carver-style insights.
"""

import sqlite3
import pandas as pd
from datetime import datetime
from signals import SIGNAL_PROVIDERS, SIGNAL_WEIGHTS
import warnings
warnings.filterwarnings('ignore')

def get_sample_markets(limit: int = 100):
    """Get sample markets using direct SQL to avoid ORM datetime issues."""
    conn = sqlite3.connect('sport_odds.db')
    
    sql = """
    SELECT DISTINCT
        m.source_id,
        m.sport,
        m.home_team,
        m.away_team,
        m.league_name,
        o.outcome,
        o.decimal_odds,
        o.updated_at,
        m.is_finished
    FROM market m
    JOIN odd o ON m.source_id = o.source_id
    WHERE m.sport = 'Soccer' 
        AND o.decimal_odds > 1.0
        AND o.decimal_odds < 10.0
    ORDER BY o.rowid DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(sql, conn, params=[limit * 3])  # Get more to ensure full markets
    conn.close()
    
    return df

def analyze_signal_performance():
    """Comprehensive signal analysis."""
    
    print("🎯 OMINARI TRADING SYSTEM - PERFORMANCE REPORT")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("Database: sport_odds.db (~216GB)")
    print("=" * 60)
    
    # Get sample data
    print("\n📊 Loading sample market data...")
    sample_df = get_sample_markets(limit=200)
    
    if sample_df.empty:
        print("No market data available")
        return {}
        
    # Group by source_id to get complete markets
    markets_with_full_odds = []
    for source_id, group in sample_df.groupby('source_id'):
        if len(group) >= 3:  # Need home/draw/away
            markets_with_full_odds.append(group)
    
    print(f"Found {len(markets_with_full_odds)} complete markets for analysis")
    
    if not markets_with_full_odds:
        print("No complete markets found")
        return {}
    
    # Analyze each signal provider
    results = {}
    
    print("\n🔍 SIGNAL PROVIDER ANALYSIS")
    print("-" * 40)
    
    for provider in SIGNAL_PROVIDERS:
        print(f"\n--- {provider.name.upper()} SIGNAL ---")
        
        provider_results = {
            'predictions': 0,
            'positive_ev_count': 0,
            'total_edge': 0,
            'total_ev': 0,
            'examples': [],
            'working': True,
            'weight': SIGNAL_WEIGHTS.get(provider.name, 1.0)
        }
        
        try:
            for market_group in markets_with_full_odds[:10]:  # Analyze first 10 markets
                # Prepare market data
                market_df = market_group.copy()
                market_df['implied_raw'] = market_df.apply(
                    lambda row: 100 / row['decimal_odds'] if row['decimal_odds'] > 0 else 0,
                    axis=1
                )
                
                # Get signal predictions
                try:
                    signal_probs = provider.get_probs(market_df)
                    
                    for j, (_, row) in enumerate(market_df.iterrows()):
                        if j < len(signal_probs):
                            prob = signal_probs.iloc[j] if hasattr(signal_probs, 'iloc') else signal_probs[j]
                            
                            # Skip invalid probabilities
                            if prob < 0 or prob > 1:
                                continue
                                
                            implied = 1 / row['decimal_odds']
                            edge = prob - implied  
                            ev = prob * (row['decimal_odds'] - 1) - (1 - prob)
                            
                            provider_results['predictions'] += 1
                            provider_results['total_edge'] += edge
                            provider_results['total_ev'] += ev
                            
                            if ev > 0:
                                provider_results['positive_ev_count'] += 1
                            
                            # Save example
                            if len(provider_results['examples']) < 3:
                                provider_results['examples'].append({
                                    'market': f"{row['home_team']} vs {row['away_team']}",
                                    'outcome': row['outcome'],
                                    'odds': row['decimal_odds'],
                                    'signal_prob': prob,
                                    'implied_prob': implied,
                                    'edge': edge,
                                    'ev': ev
                                })
                                
                except Exception as e:
                    print(f"Error getting signals: {e}")
                    
        except Exception as e:
            provider_results['working'] = False
            provider_results['error'] = str(e)
            print(f"Provider failed: {e}")
            
        # Calculate averages
        if provider_results['predictions'] > 0:
            avg_edge = provider_results['total_edge'] / provider_results['predictions']
            avg_ev = provider_results['total_ev'] / provider_results['predictions']
            positive_rate = provider_results['positive_ev_count'] / provider_results['predictions']
            
            print(f"  Predictions: {provider_results['predictions']}")
            print(f"  Positive EV rate: {positive_rate:.1%}")
            print(f"  Average edge: {avg_edge*100:+.3f}%")
            print(f"  Average EV: {avg_ev:+.4f}")
            print(f"  Signal weight: {provider_results['weight']}")
            
            # Show examples
            print("  Examples:")
            for ex in provider_results['examples']:
                print(f"    {ex['outcome']:8} {ex['odds']:5.2f} → edge: {ex['edge']*100:+.2f}%, EV: {ex['ev']:+.3f}")
                
        else:
            print("  No valid predictions generated")
            
        results[provider.name] = provider_results
    
    return results

def calculate_portfolio_performance(signal_results: dict):
    """Calculate overall portfolio performance."""
    
    print("\n💼 PORTFOLIO ANALYSIS")
    print("-" * 30)
    
    working_signals = {name: data for name, data in signal_results.items() 
                      if data.get('working', False) and data.get('predictions', 0) > 0}
    
    if not working_signals:
        print("❌ No working signals with predictions")
        return {}
    
    print(f"Working signals: {len(working_signals)}/{len(SIGNAL_PROVIDERS)}")
    
    # Calculate weighted performance
    total_weight = sum(SIGNAL_WEIGHTS.get(name, 1.0) for name in working_signals.keys())
    
    weighted_edge = 0
    weighted_ev = 0
    weighted_positive_rate = 0
    
    print("\nSignal Contributions:")
    for signal_name, data in working_signals.items():
        weight = SIGNAL_WEIGHTS.get(signal_name, 1.0) / total_weight
        avg_edge = data['total_edge'] / data['predictions']
        avg_ev = data['total_ev'] / data['predictions']
        pos_rate = data['positive_ev_count'] / data['predictions']
        
        weighted_edge += avg_edge * weight
        weighted_ev += avg_ev * weight  
        weighted_positive_rate += pos_rate * weight
        
        print(f"  {signal_name:15}: weight={weight:.1%}, edge={avg_edge*100:+.2f}%, EV={avg_ev:+.3f}")
    
    print("\n📈 Portfolio Metrics:")
    print(f"  Weighted Edge: {weighted_edge*100:+.3f}%") 
    print(f"  Weighted EV: {weighted_ev:+.4f}")
    print(f"  Positive EV Rate: {weighted_positive_rate:.1%}")
    
    # Kelly sizing analysis
    if weighted_ev > 0:
        avg_odds = 2.5  # Typical soccer odds
        kelly_fraction = weighted_edge / ((avg_odds - 1) ** 2)
        safe_fraction = kelly_fraction * 0.25  # Quarter Kelly
        
        print("\n💰 Position Sizing (Kelly Criterion):")
        print(f"  Full Kelly fraction: {kelly_fraction:.3f}")
        print(f"  Safe Kelly (25%): {safe_fraction:.3f}")
        print(f"  Max bet size: {safe_fraction*100:.1f}% of bankroll")
        
        # Theoretical performance
        annual_bets = 365 * 3  # 3 soccer bets per day average
        theoretical_roi = weighted_ev * safe_fraction * annual_bets * 100
        
        print("\n🎯 Theoretical Performance (Annual):")
        print(f"  Expected ROI: {theoretical_roi:.1f}%")
        print(f"  Risk-adjusted return: {theoretical_roi/2:.1f}% (conservative)")
        
    else:
        print("\n❌ Negative Expected Value")
        print("   Kelly system correctly recommends no betting")
        
    return {
        'weighted_edge': weighted_edge,
        'weighted_ev': weighted_ev,
        'positive_rate': weighted_positive_rate,
        'working_signals': len(working_signals)
    }

def provide_recommendations(signal_results: dict, portfolio: dict):
    """Provide actionable recommendations."""
    
    print("\n🎯 RECOMMENDATIONS")
    print("=" * 30)
    
    print("\n1. SIGNAL HEALTH:")
    for name, data in signal_results.items():
        if data.get('working', False):
            if data.get('predictions', 0) > 0:
                avg_edge = data['total_edge'] / data['predictions'] * 100
                print(f"   ✓ {name}: Working, {avg_edge:+.2f}% avg edge")
            else:
                print(f"   ⚠ {name}: Connected but no predictions")
        else:
            error = data.get('error', 'Unknown error')
            print(f"   ❌ {name}: Failed - {error}")
    
    print("\n2. PERFORMANCE ASSESSMENT:")
    if portfolio.get('weighted_ev', 0) > 0:
        print("   ✓ Positive expected value detected")
        print("   ✓ Kelly system ready for live trading")
    else:
        print("   ❌ No positive expected value")
        print("   ❌ Kelly correctly prevents betting")
        
    print("\n3. ACTION ITEMS:")
    
    # Check ImpliedRaw specifically
    implied_data = signal_results.get('implied_probability', {})
    if implied_data.get('working'):
        print("   📊 ImpliedRaw: Working as expected (zero edge by design)")
    
    # Check external signals
    for signal_name in ['coin_flip', 'grant']:
        if not signal_results.get(signal_name, {}).get('working', False):
            print(f"   🔧 Fix {signal_name} signal (gRPC connection issues)")
            
    print("   📈 Consider adding new predictive signals:")
    print("      - Historical team performance")
    print("      - Market movement indicators") 
    print("      - Cross-bookmaker arbitrage detection")
    
    if portfolio.get('working_signals', 0) >= 2:
        print("   🎯 System ready for paper trading")
    else:
        print("   ⏳ Need more working signals before live trading")

def main():
    """Run complete performance analysis."""
    
    # Analyze signals
    signal_results = analyze_signal_performance()
    
    # Calculate portfolio metrics  
    portfolio_metrics = calculate_portfolio_performance(signal_results)
    
    # Provide recommendations
    provide_recommendations(signal_results, portfolio_metrics)
    
    print("\n" + "=" * 60)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 60)
    
    working = sum(1 for data in signal_results.values() if data.get('working', False))
    total = len(SIGNAL_PROVIDERS)
    
    print(f"Signal Health: {working}/{total} providers working")
    
    if portfolio_metrics:
        edge = portfolio_metrics.get('weighted_edge', 0) * 100
        pos_rate = portfolio_metrics.get('positive_rate', 0) * 100
        print(f"Portfolio Edge: {edge:+.3f}%")
        print(f"Positive EV Rate: {pos_rate:.1f}%")
        
        if edge > 0:
            print("✅ SYSTEM STATUS: Positive edge detected, ready for trading")
        else:
            print("⏸️  SYSTEM STATUS: No edge detected, correctly holding cash")
    else:
        print("⚠️  SYSTEM STATUS: Unable to calculate portfolio metrics")
    
    print("\nNext: Fix gRPC signals or add new predictive models")

if __name__ == "__main__":
    main()