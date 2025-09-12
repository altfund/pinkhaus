#!/usr/bin/env python3
"""
Focused backtest analysis using enhanced Kelly system.
Runs on recent data with all signal providers.
"""

import pandas as pd
from datetime import datetime, timezone, timedelta
from enhanced_kelly_system import EnhancedKellySystem
from signals import SIGNAL_PROVIDERS
from database_v2 import db_manager
from models import Market, Odd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_focused_backtest(days_back: int = 7) -> dict:
    """Run backtest on recent data to analyze performance."""
    
    print(f"=== Running Focused Backtest ({days_back} days) ===\n")
    
    # Time window
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(days=days_back)
    
    print(f"Period: {start_time.strftime('%Y-%m-%d')} to {end_time.strftime('%Y-%m-%d')}")
    
    kelly_system = EnhancedKellySystem(bankroll=10000)
    
    with db_manager.get_db_session() as db:
        # Get markets from the time period
        markets = db.query(Market).filter(
            Market.sport == 'Soccer',
            Market.maturity_date >= start_time,
            Market.maturity_date <= end_time
        ).limit(100).all()  # Reasonable limit
        
        print(f"Found {len(markets)} markets in time window")
        
        if not markets:
            print("No markets found in time window")
            return {}
            
        all_results = []
        signal_breakdown = {provider.name: [] for provider in SIGNAL_PROVIDERS}
        
        for i, market in enumerate(markets[:20]):  # Process first 20 for speed
            print(f"Processing market {i+1}/20: {market.home_team} vs {market.away_team}")
            
            # Get odds for this market
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).order_by(Odd.updated_at.desc()).limit(3).all()
            
            if len(odds) < 3:
                continue
                
            # Build market DataFrame
            market_df = pd.DataFrame([{
                'source_id': odd.source_id,
                'market_type': 'winner',
                'normalized_outcome': odd.outcome,
                'normalized_line': 0.0,
                'bet_name': odd.outcome,
                'decimal_odds': odd.decimal_odds,
                'implied_raw': (1/odd.decimal_odds * 100) if odd.decimal_odds > 0 else 0,
                'home_team': market.home_team,
                'away_team': market.away_team,
                'league_name': market.league_name,
                'time': odd.updated_at,
                'maturity_date': market.maturity_date
            } for odd in odds])
            
            # Test each signal individually
            for provider in SIGNAL_PROVIDERS:
                try:
                    signal_probs = provider.get_probs(market_df)
                    
                    for j, (_, row) in enumerate(market_df.iterrows()):
                        if j < len(signal_probs):
                            prob = signal_probs.iloc[j]
                            implied = 1 / row['decimal_odds']
                            edge = prob - implied
                            ev = prob * (row['decimal_odds'] - 1) - (1 - prob)
                            
                            signal_breakdown[provider.name].append({
                                'market': f"{market.home_team} vs {market.away_team}",
                                'outcome': row['normalized_outcome'],
                                'odds': row['decimal_odds'],
                                'signal_prob': prob,
                                'implied_prob': implied,
                                'edge': edge,
                                'expected_value': ev,
                                'positive_ev': ev > 0
                            })
                            
                except Exception as e:
                    logger.error(f"Error with {provider.name}: {e}")
            
            # Run Kelly optimization on this market
            try:
                opportunities = kelly_system.evaluate_market_opportunities(market_df)
                if not opportunities.empty:
                    all_results.append(opportunities)
                    
            except Exception as e:
                logger.error(f"Kelly optimization error: {e}")
    
    # Analyze results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        kelly_report = kelly_system.generate_trading_report(combined_results)
    else:
        kelly_report = {'summary': 'No Kelly opportunities found'}
    
    return {
        'kelly_report': kelly_report,
        'signal_breakdown': signal_breakdown,
        'markets_analyzed': len(markets),
        'timeframe': f"{start_time.date()} to {end_time.date()}"
    }

def generate_signal_performance_report(signal_breakdown: dict) -> None:
    """Generate detailed signal performance report."""
    
    print("\n=== Individual Signal Performance ===")
    
    for signal_name, results in signal_breakdown.items():
        if not results:
            print(f"\n{signal_name}: No results")
            continue
            
        df = pd.DataFrame(results)
        
        total_bets = len(df)
        positive_ev_bets = len(df[df['positive_ev'] == True])
        avg_edge = df['edge'].mean()
        avg_ev = df['expected_value'].mean()
        
        print(f"\n{signal_name}:")
        print(f"  Total predictions: {total_bets}")
        print(f"  Positive EV bets: {positive_ev_bets} ({positive_ev_bets/total_bets*100:.1f}%)")
        print(f"  Average edge: {avg_edge:.4f} ({avg_edge*100:+.2f}%)")
        print(f"  Average EV: {avg_ev:.4f}")
        
        if positive_ev_bets > 0:
            pos_df = df[df['positive_ev'] == True]
            print(f"  Best positive EV: {pos_df['expected_value'].max():.4f}")
            print(f"  Avg positive edge: {pos_df['edge'].mean()*100:+.2f}%")

if __name__ == "__main__":
    backtest_results = run_focused_backtest(days_back=30)  # 30 day analysis
    
    print("\n=== Backtest Summary ===")
    print(f"Timeframe: {backtest_results['timeframe']}")
    print(f"Markets analyzed: {backtest_results['markets_analyzed']}")
    
    # Kelly system report
    kelly = backtest_results['kelly_report']
    print("\nKelly System Results:")
    print(f"  {kelly['summary']}")
    if 'expected_roi' in kelly:
        print(f"  Expected ROI: {kelly['expected_roi']:.2f}%")
    
    # Signal breakdown
    generate_signal_performance_report(backtest_results['signal_breakdown'])