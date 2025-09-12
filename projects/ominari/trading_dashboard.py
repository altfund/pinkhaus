#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Dashboard for Ominari System
Displays upcoming matches, prices, signals, and portfolio status.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
import json
from pathlib import Path
from tabulate import tabulate

from database_v2 import db_manager
from models import Market, Odd
from paper_trading_engine import PaperTradingEngine, PaperOrder


class TradingDashboard:
    """Comprehensive trading dashboard for Ominari system."""
    
    def __init__(self):
        self.paper_engine = PaperTradingEngine()
        # Only use ImpliedRawSignal for now since gRPC servers are not running
        from signals import ImpliedRawSignal
        self.signal_providers = {"implied_probability": ImpliedRawSignal()}
        self.portfolio_file = Path("paper_portfolio.json")
        self.trades_file = Path("paper_trades.csv")
        self.load_portfolio()
        
    def load_portfolio(self):
        """Load portfolio from file."""
        if self.portfolio_file.exists():
            with open(self.portfolio_file, 'r') as f:
                self.portfolio = json.load(f)
        else:
            self.portfolio = {
                'cash': 10000.0,  # Starting capital
                'positions': {},
                'total_value': 10000.0,
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'pending_bets': {}
            }
            
    def save_portfolio(self):
        """Save portfolio to file."""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2, default=str)
            
    def get_upcoming_matches(self, hours_ahead: int = 48, sport_filter: Optional[str] = None) -> pd.DataFrame:
        """Get upcoming matches with odds and signals."""
        with db_manager.get_db_session() as db:
            # Time window
            now = datetime.now(timezone.utc)
            future_time = now + timedelta(hours=hours_ahead)
            
            # Query markets
            query = db.query(Market).filter(
                Market.maturity_date >= now,
                Market.maturity_date <= future_time,
                Market.is_finished == False
            )
            
            if sport_filter:
                query = query.filter(Market.sport.like(f'%{sport_filter}%'))
                
            markets = query.order_by(Market.maturity_date).limit(100).all()
            
            # Build match data
            match_data = []
            for market in markets:
                # Get latest odds
                odds = db.query(Odd).filter(
                    Odd.source_id == market.source_id
                ).order_by(Odd.updated_at.desc()).limit(10).all()
                
                # Group odds by position/outcome
                home_odds = []
                away_odds = []
                draw_odds = []
                
                for odd in odds:
                    if odd.outcome == market.home_team or odd.position == 0:
                        home_odds.append(odd.decimal_odds)
                    elif odd.outcome == market.away_team or odd.position == 1:
                        away_odds.append(odd.decimal_odds)
                    elif 'draw' in odd.outcome.lower() or odd.position == 2:
                        draw_odds.append(odd.decimal_odds)
                
                # Calculate best odds and implied probabilities
                best_home = min(home_odds) if home_odds else None
                best_away = min(away_odds) if away_odds else None
                best_draw = min(draw_odds) if draw_odds else None
                
                match_info = {
                    'match_id': market.source_id,
                    'sport': market.sport,
                    'league': market.league_name or 'Unknown',
                    'home_team': market.home_team,
                    'away_team': market.away_team,
                    'kick_off': market.maturity_date,
                    'hours_until': (market.maturity_date.replace(tzinfo=timezone.utc) - now).total_seconds() / 3600 if market.maturity_date else 0,
                    'home_odds': best_home,
                    'away_odds': best_away,
                    'draw_odds': best_draw,
                    'home_prob': 1/best_home if best_home else None,
                    'away_prob': 1/best_away if best_away else None,
                    'draw_prob': 1/best_draw if best_draw else None,
                    'market_margin': None,
                    'best_bookmaker': odds[0].bookmaker if odds else None
                }
                
                # Calculate market margin (overround)
                probs = [p for p in [match_info['home_prob'], match_info['away_prob'], match_info['draw_prob']] if p]
                if probs:
                    match_info['market_margin'] = (sum(probs) - 1) * 100
                
                match_data.append(match_info)
                
        return pd.DataFrame(match_data)
    
    def calculate_signals(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate signal recommendations for matches."""
        if matches_df.empty:
            return matches_df
            
        # Prepare data for signal providers
        signal_data = []
        for _, match in matches_df.iterrows():
            # Create rows for each outcome
            for outcome, odds, prob in [
                (match['home_team'], match['home_odds'], match['home_prob']),
                (match['away_team'], match['away_odds'], match['away_prob']),
                ('Draw', match['draw_odds'], match['draw_prob'])
            ]:
                if odds:
                    signal_data.append({
                        'source_id': match['match_id'],
                        'sport': match['sport'],
                        'home_team': match['home_team'],
                        'away_team': match['away_team'],
                        'normalized_outcome': outcome,
                        'decimal_odds': odds,
                        'implied_raw': prob * 100 if prob else 0,
                        'time': datetime.now(timezone.utc),
                        'league_name': match['league'],
                        'maturity_date': match['kick_off']
                    })
        
        if not signal_data:
            return matches_df
            
        signal_df = pd.DataFrame(signal_data)
        
        # Get predictions from each signal provider
        all_signals = {}
        for name, provider in self.signal_providers.items():
            try:
                predictions = provider.get_probs(signal_df)
                if not predictions.empty:
                    all_signals[name] = predictions
            except Exception as e:
                print(f"Error getting signal from {name}: {e}")
        
        # Aggregate signals
        if all_signals:
            # Create match-level recommendations
            recommendations = []
            for match_id in matches_df['match_id'].unique():
                match = matches_df[matches_df['match_id'] == match_id].iloc[0]
                
                # Get signals for this match
                match_signals = {}
                for signal_name, predictions in all_signals.items():
                    match_preds = predictions[signal_df['source_id'] == match_id]
                    if not match_preds.empty:
                        # Find best outcome according to this signal
                        if match_preds.isna().all():
                            continue
                        best_idx = match_preds.idxmax()
                        if pd.isna(best_idx):
                            continue
                        best_outcome = signal_df.loc[best_idx, 'normalized_outcome']
                        best_prob = match_preds.loc[best_idx]
                        match_signals[signal_name] = {
                            'outcome': best_outcome,
                            'probability': best_prob,
                            'edge': best_prob - signal_df.loc[best_idx, 'implied_raw'] / 100
                        }
                
                # Calculate consensus
                if match_signals:
                    outcomes = [s['outcome'] for s in match_signals.values()]
                    probs = [s['probability'] for s in match_signals.values()]
                    edges = [s['edge'] for s in match_signals.values()]
                    
                    # Most common outcome
                    consensus_outcome = max(set(outcomes), key=outcomes.count)
                    consensus_prob = np.mean([p for o, p in zip(outcomes, probs) if o == consensus_outcome])
                    consensus_edge = np.mean([e for o, e in zip(outcomes, edges) if o == consensus_outcome])
                    
                    recommendations.append({
                        'match_id': match_id,
                        'recommended_bet': consensus_outcome,
                        'signal_probability': consensus_prob,
                        'expected_edge': consensus_edge,
                        'confidence': len([o for o in outcomes if o == consensus_outcome]) / len(outcomes),
                        'signals_agree': len([o for o in outcomes if o == consensus_outcome]),
                        'total_signals': len(outcomes)
                    })
            
            # Merge recommendations with matches
            rec_df = pd.DataFrame(recommendations)
            matches_df = matches_df.merge(rec_df, on='match_id', how='left')
        
        return matches_df
    
    def calculate_kelly_size(self, probability: float, odds: float, kelly_fraction: float = 0.25) -> float:
        """Calculate Kelly criterion bet size."""
        if probability <= 0 or odds <= 1:
            return 0.0
            
        # Kelly formula: f = (p * o - 1) / (o - 1)
        # where p = probability, o = decimal odds
        edge = probability * odds - 1
        if edge <= 0:
            return 0.0
            
        kelly = edge / (odds - 1)
        
        # Apply fractional Kelly for safety
        return min(kelly * kelly_fraction, 0.05)  # Max 5% of bankroll
    
    def execute_recommendations(self, matches_df: pd.DataFrame, 
                              min_edge: float = 0.02,
                              max_bets: int = 10) -> List[Dict]:
        """Execute paper trades based on recommendations."""
        if 'expected_edge' not in matches_df.columns:
            return []
            
        # Filter by minimum edge and sort by edge
        good_bets = matches_df[
            (matches_df['expected_edge'] > min_edge) & 
            (matches_df['confidence'] >= 0.6)
        ].nlargest(max_bets, 'expected_edge')
        
        executed_trades = []
        
        for _, bet in good_bets.iterrows():
            # Get the odds for recommended outcome
            if bet['recommended_bet'] == bet['home_team']:
                odds = bet['home_odds']
            elif bet['recommended_bet'] == bet['away_team']:
                odds = bet['away_odds']
            else:
                odds = bet['draw_odds']
                
            if not odds:
                continue
                
            # Calculate bet size
            bet_size = self.calculate_kelly_size(
                bet['signal_probability'], 
                odds
            ) * self.portfolio['cash']
            
            # Minimum bet size
            if bet_size < 10:
                continue
                
            # Execute paper trade
            order = PaperOrder(
                order_id=f"BET_{datetime.now().timestamp()}",
                timestamp=datetime.now(timezone.utc),
                source_id=bet['match_id'],
                market_type='moneyline',
                bet_name=bet['recommended_bet'],
                side='back',
                size=round(bet_size, 2),
                limit_price=odds,
                signal_name='consensus',
                expected_edge=bet['expected_edge']
            )
            
            # Record the trade
            trade_record = {
                'timestamp': datetime.now(timezone.utc),
                'match_id': bet['match_id'],
                'match': f"{bet['home_team']} vs {bet['away_team']}",
                'bet_on': bet['recommended_bet'],
                'odds': odds,
                'stake': bet_size,
                'potential_return': bet_size * odds,
                'signal_prob': bet['signal_probability'],
                'market_prob': 1/odds,
                'edge': bet['expected_edge'],
                'confidence': bet['confidence'],
                'kick_off': bet['kick_off']
            }
            
            # Update portfolio
            self.portfolio['cash'] -= bet_size
            self.portfolio['pending_bets'][bet['match_id']] = trade_record
            self.portfolio['trades'] += 1
            
            executed_trades.append(trade_record)
            
        self.save_portfolio()
        
        # Save trades to CSV
        if executed_trades:
            trades_df = pd.DataFrame(executed_trades)
            if self.trades_file.exists():
                existing = pd.read_csv(self.trades_file)
                trades_df = pd.concat([existing, trades_df], ignore_index=True)
            trades_df.to_csv(self.trades_file, index=False)
            
        return executed_trades
    
    def display_dashboard(self, sport_filter: Optional[str] = None):
        """Display comprehensive trading dashboard."""
        print("\n" + "="*100)
        print("OMINARI TRADING DASHBOARD")
        print("="*100)
        print(f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        
        # Portfolio Summary
        print("\n📊 PORTFOLIO SUMMARY")
        print("-"*50)
        total_at_risk = sum(bet['stake'] for bet in self.portfolio['pending_bets'].values())
        print(f"Cash Balance: ${self.portfolio['cash']:,.2f}")
        print(f"At Risk: ${total_at_risk:,.2f}")
        print(f"Total Trades: {self.portfolio['trades']}")
        if self.portfolio['trades'] > 0:
            win_rate = self.portfolio['wins'] / (self.portfolio['wins'] + self.portfolio['losses']) * 100 if (self.portfolio['wins'] + self.portfolio['losses']) > 0 else 0
            print(f"Win Rate: {win_rate:.1f}% ({self.portfolio['wins']}W / {self.portfolio['losses']}L)")
        
        # Get upcoming matches
        print("\n🔄 Fetching upcoming matches...")
        matches_df = self.get_upcoming_matches(hours_ahead=48, sport_filter=sport_filter)
        
        if matches_df.empty:
            print("No upcoming matches found.")
            return
            
        # Calculate signals
        print("🧮 Calculating signals...")
        matches_df = self.calculate_signals(matches_df)
        
        # Display matches with recommendations
        print(f"\n⚽ UPCOMING MATCHES ({len(matches_df)} found)")
        print("-"*100)
        
        # Format for display
        display_df = matches_df[[
            'sport', 'league', 'home_team', 'away_team', 'kick_off',
            'home_odds', 'draw_odds', 'away_odds', 'market_margin'
        ]].copy()
        
        # Add recommendations if available
        if 'recommended_bet' in matches_df.columns:
            display_df['recommendation'] = matches_df.apply(
                lambda x: f"{x['recommended_bet']} ({x['expected_edge']*100:.1f}%)" 
                if pd.notna(x.get('recommended_bet')) else '-',
                axis=1
            )
        
        # Format datetime
        display_df['kick_off'] = pd.to_datetime(display_df['kick_off']).dt.strftime('%m-%d %H:%M')
        
        # Display table
        print(tabulate(display_df.head(20), headers='keys', tablefmt='grid', floatfmt='.2f'))
        
        # Best betting opportunities
        if 'expected_edge' in matches_df.columns:
            best_bets = matches_df[matches_df['expected_edge'] > 0.02].nlargest(10, 'expected_edge')
            
            if not best_bets.empty:
                print("\n🎯 BEST BETTING OPPORTUNITIES")
                print("-"*100)
                
                bet_display = best_bets[[
                    'home_team', 'away_team', 'recommended_bet', 
                    'signal_probability', 'expected_edge', 'confidence'
                ]].copy()
                
                bet_display['signal_prob_%'] = bet_display['signal_probability'] * 100
                bet_display['edge_%'] = bet_display['expected_edge'] * 100
                bet_display['confidence_%'] = bet_display['confidence'] * 100
                
                bet_display = bet_display[[
                    'home_team', 'away_team', 'recommended_bet',
                    'signal_prob_%', 'edge_%', 'confidence_%'
                ]]
                
                print(tabulate(bet_display, headers='keys', tablefmt='grid', floatfmt='.1f'))
        
        # Active bets
        if self.portfolio['pending_bets']:
            print("\n📋 ACTIVE BETS")
            print("-"*100)
            
            active_bets = []
            for match_id, bet in self.portfolio['pending_bets'].items():
                active_bets.append({
                    'Match': bet['match'],
                    'Bet On': bet['bet_on'],
                    'Odds': bet['odds'],
                    'Stake': f"${bet['stake']:.2f}",
                    'Potential': f"${bet['potential_return']:.2f}",
                    'Edge': f"{bet['edge']*100:.1f}%",
                    'Kick Off': pd.to_datetime(bet['kick_off']).strftime('%m-%d %H:%M')
                })
            
            print(tabulate(pd.DataFrame(active_bets), headers='keys', tablefmt='grid'))
        
        print("\n" + "="*100)
    
    def execute_auto_trades(self, sport_filter: Optional[str] = None):
        """Automatically execute trades based on signals."""
        print("\n🤖 AUTO-TRADING MODE")
        print("-"*50)
        
        # Get matches and signals
        matches_df = self.get_upcoming_matches(hours_ahead=24, sport_filter=sport_filter)
        if matches_df.empty:
            print("No upcoming matches found.")
            return
            
        matches_df = self.calculate_signals(matches_df)
        
        # Execute recommendations
        trades = self.execute_recommendations(matches_df)
        
        if trades:
            print(f"\n✅ Executed {len(trades)} trades:")
            for trade in trades:
                print(f"  - {trade['match']}: {trade['bet_on']} @ {trade['odds']:.2f} (${trade['stake']:.2f})")
        else:
            print("No trades meet the criteria.")
    
    def update_results(self):
        """Update results for completed matches."""
        print("\n📈 Updating match results...")
        
        with db_manager.get_db_session() as db:
            completed_bets = []
            
            for match_id, bet in list(self.portfolio['pending_bets'].items()):
                # Check if match is finished
                market = db.query(Market).filter(
                    Market.source_id == match_id
                ).first()
                
                if market and market.is_finished:
                    # Determine if bet won
                    won = False
                    if market.home_score is not None and market.away_score is not None:
                        if bet['bet_on'] == market.home_team and market.home_score > market.away_score:
                            won = True
                        elif bet['bet_on'] == market.away_team and market.away_score > market.home_score:
                            won = True
                        elif bet['bet_on'] == 'Draw' and market.home_score == market.away_score:
                            won = True
                    
                    # Update portfolio
                    if won:
                        self.portfolio['cash'] += bet['potential_return']
                        self.portfolio['wins'] += 1
                        profit = bet['potential_return'] - bet['stake']
                        print(f"  ✅ WON: {bet['match']} - {bet['bet_on']} (+${profit:.2f})")
                    else:
                        self.portfolio['losses'] += 1
                        print(f"  ❌ LOST: {bet['match']} - {bet['bet_on']} (-${bet['stake']:.2f})")
                    
                    completed_bets.append(match_id)
            
            # Remove completed bets
            for match_id in completed_bets:
                del self.portfolio['pending_bets'][match_id]
            
            self.save_portfolio()
            
            if completed_bets:
                print(f"\nUpdated {len(completed_bets)} results.")
            else:
                print("No completed matches to update.")


def main():
    """Main function to run the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ominari Trading Dashboard")
    parser.add_argument('--sport', help='Filter by sport (e.g., Soccer, Basketball)')
    parser.add_argument('--auto-trade', action='store_true', help='Execute trades automatically')
    parser.add_argument('--update-results', action='store_true', help='Update results for completed matches')
    
    args = parser.parse_args()
    
    dashboard = TradingDashboard()
    
    if args.update_results:
        dashboard.update_results()
    elif args.auto_trade:
        dashboard.execute_auto_trades(sport_filter=args.sport)
    else:
        dashboard.display_dashboard(sport_filter=args.sport)


if __name__ == "__main__":
    main()