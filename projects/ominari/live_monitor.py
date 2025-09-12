#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Monitoring Interface for Ominari Trading System
Provides real-time updates on matches, signals, and portfolio performance.
"""

import asyncio
from datetime import datetime, timezone, timedelta
import pandas as pd
import os

from database_v2 import db_manager
from models import Market, Odd
from trading_dashboard import TradingDashboard


class LiveMonitor:
    """Real-time monitoring system for trading activity."""
    
    def __init__(self):
        self.dashboard = TradingDashboard()
        self.last_update = datetime.now(timezone.utc)
        self.update_interval = 60  # seconds
        self.running = True
        
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')
        
    def format_time_until(self, dt: datetime) -> str:
        """Format time until an event."""
        if not dt:
            return "Unknown"
            
        # Ensure timezone awareness
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
            
        delta = dt - datetime.now(timezone.utc)
        hours = delta.total_seconds() / 3600
        
        if hours < 0:
            return "Started"
        elif hours < 1:
            minutes = int(delta.total_seconds() / 60)
            return f"{minutes}m"
        elif hours < 24:
            return f"{int(hours)}h {int((hours % 1) * 60)}m"
        else:
            days = int(hours / 24)
            return f"{days}d {int(hours % 24)}h"
            
    def get_live_odds_changes(self, minutes: int = 5) -> pd.DataFrame:
        """Get recent odds movements."""
        with db_manager.get_db_session() as db:
            since = datetime.now(timezone.utc) - timedelta(minutes=minutes)
            
            # Get odds that have changed recently
            recent_odds = db.query(Odd).filter(
                Odd.updated_at >= since
            ).order_by(Odd.updated_at.desc()).limit(100).all()
            
            # Group by market and track changes
            market_changes = {}
            for odd in recent_odds:
                key = (odd.source_id, odd.outcome)
                if key not in market_changes:
                    market_changes[key] = {
                        'source_id': odd.source_id,
                        'outcome': odd.outcome,
                        'current_odds': odd.decimal_odds,
                        'previous_odds': None,
                        'change': 0,
                        'updated': odd.updated_at
                    }
                else:
                    # Track the change
                    if market_changes[key]['previous_odds'] is None:
                        market_changes[key]['previous_odds'] = odd.decimal_odds
                        market_changes[key]['change'] = (
                            market_changes[key]['current_odds'] - odd.decimal_odds
                        )
                        
        return pd.DataFrame(list(market_changes.values()))
    
    def display_live_dashboard(self):
        """Display live dashboard with real-time updates."""
        self.clear_screen()
        
        # Header
        print("="*120)
        print(f"{'OMINARI LIVE TRADING MONITOR':^120}")
        print("="*120)
        print(f"Last Update: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        print(f"Auto-refresh: Every {self.update_interval} seconds | Press Ctrl+C to exit")
        
        # Portfolio Status
        print("\n📊 PORTFOLIO STATUS")
        print("-"*60)
        
        portfolio = self.dashboard.portfolio
        total_at_risk = sum(bet['stake'] for bet in portfolio['pending_bets'].values())
        potential_return = sum(bet['potential_return'] for bet in portfolio['pending_bets'].values())
        
        # Create portfolio summary table
        portfolio_data = [
            ["Cash Balance", f"${portfolio['cash']:,.2f}"],
            ["Active Bets", len(portfolio['pending_bets'])],
            ["Amount at Risk", f"${total_at_risk:,.2f}"],
            ["Potential Return", f"${potential_return:,.2f}"],
            ["Total Trades", portfolio['trades']],
            ["Win Rate", f"{portfolio['wins']}/{portfolio['wins'] + portfolio['losses']}" if portfolio['trades'] > 0 else "0/0"]
        ]
        
        # Print in columns
        for i in range(0, len(portfolio_data), 2):
            left = portfolio_data[i]
            right = portfolio_data[i+1] if i+1 < len(portfolio_data) else ["", ""]
            print(f"{left[0]:<20} {left[1]:<20} | {right[0]:<20} {right[1]:<20}")
        
        # Next matches starting soon
        print("\n⏰ MATCHES STARTING SOON")
        print("-"*120)
        
        with db_manager.get_db_session() as db:
            next_hour = datetime.now(timezone.utc) + timedelta(hours=1)
            
            upcoming = db.query(Market).filter(
                Market.maturity_date >= datetime.now(timezone.utc),
                Market.maturity_date <= next_hour,
                Market.is_finished == False
            ).order_by(Market.maturity_date).limit(10).all()
            
            if upcoming:
                for market in upcoming[:5]:
                    # Get current odds
                    odds = db.query(Odd).filter(
                        Odd.source_id == market.source_id
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                    
                    odds_str = ""
                    if odds:
                        home_odd = next((o.decimal_odds for o in odds if o.position == 0), None)
                        away_odd = next((o.decimal_odds for o in odds if o.position == 1), None)
                        if home_odd and away_odd:
                            odds_str = f"[{home_odd:.2f} / {away_odd:.2f}]"
                    
                    time_str = self.format_time_until(market.maturity_date)
                    print(f"  {time_str:<8} {market.sport:<12} {market.home_team} vs {market.away_team} {odds_str}")
            else:
                print("  No matches starting in the next hour")
        
        # Recent odds movements
        print("\n📈 RECENT ODDS MOVEMENTS")
        print("-"*120)
        
        odds_changes = self.get_live_odds_changes(minutes=10)
        significant_changes = odds_changes[abs(odds_changes['change']) > 0.1].head(10)
        
        if not significant_changes.empty:
            for _, change in significant_changes.iterrows():
                direction = "↑" if change['change'] > 0 else "↓"
                color = "\033[91m" if change['change'] > 0 else "\033[92m"  # Red for worse, green for better
                reset = "\033[0m"
                
                # Get market info
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == change['source_id']
                    ).first()
                    
                    if market:
                        match_str = f"{market.home_team} vs {market.away_team}"
                        print(f"  {direction} {change['outcome']:<20} {change['previous_odds']:.2f} → "
                              f"{color}{change['current_odds']:.2f}{reset} ({change['change']:+.2f}) - {match_str}")
        else:
            print("  No significant odds movements in the last 10 minutes")
        
        # Active bets status
        if portfolio['pending_bets']:
            print("\n🎯 ACTIVE BETS")
            print("-"*120)
            
            # Sort by kick off time
            active_bets = sorted(
                portfolio['pending_bets'].items(),
                key=lambda x: x[1]['kick_off']
            )
            
            for match_id, bet in active_bets[:10]:  # Show max 10
                # Check current status
                with db_manager.get_db_session() as db:
                    market = db.query(Market).filter(
                        Market.source_id == match_id
                    ).first()
                    
                    status = "Pending"
                    if market:
                        if market.is_finished:
                            status = "Finished"
                        elif market.maturity_date and market.maturity_date.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
                            status = "In-Play"
                    
                    # Get current odds for the bet
                    current_odd = db.query(Odd).filter(
                        Odd.source_id == match_id,
                        Odd.outcome == bet['bet_on']
                    ).order_by(Odd.updated_at.desc()).first()
                    
                    current_price = current_odd.decimal_odds if current_odd else bet['odds']
                    price_move = current_price - bet['odds']
                    
                    print(f"  [{status:<10}] {bet['match']:<40} {bet['bet_on']:<15} "
                          f"${bet['stake']:>8.2f} @ {bet['odds']:.2f} "
                          f"(now {current_price:.2f} {price_move:+.2f})")
        
        # Signal performance
        print("\n📊 SIGNAL PERFORMANCE (Last 24h)")
        print("-"*60)
        
        # This would show actual signal performance if we had results
        print("  Signal performance tracking will be available after matches complete")
        
        print("\n" + "="*120)
    
    async def run_live_monitor(self):
        """Run the live monitoring loop."""
        print("Starting live monitor... Press Ctrl+C to exit")
        
        try:
            while self.running:
                self.display_live_dashboard()
                
                # Check for completed matches every 5 updates
                if (datetime.now(timezone.utc) - self.last_update).total_seconds() > 300:
                    print("\n🔄 Checking for completed matches...")
                    self.dashboard.update_results()
                    self.last_update = datetime.now(timezone.utc)
                
                await asyncio.sleep(self.update_interval)
                
        except KeyboardInterrupt:
            print("\n\nStopping live monitor...")
            self.running = False
    
    def run(self):
        """Run the monitor."""
        asyncio.run(self.run_live_monitor())


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Live Trading Monitor")
    parser.add_argument('--interval', type=int, default=60, 
                       help='Update interval in seconds (default: 60)')
    
    args = parser.parse_args()
    
    monitor = LiveMonitor()
    monitor.update_interval = args.interval
    monitor.run()


if __name__ == "__main__":
    main()