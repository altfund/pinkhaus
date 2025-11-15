#!/usr/bin/env python3
"""
Demo paper trading with notifications
Shows how the system works without real money
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up environment
os.environ['PG_PORT'] = '5999'

from database_v2 import db_manager
from models import Market, Odd, Bet, BettingSession
from config.bankroll_config import BankrollConfig
# from integrated_trading_system import IntegratedTradingSystem

def show_portfolio_status():
    """Display current portfolio status"""
    config = BankrollConfig()
    print("\n💰 Portfolio Status")
    print("=" * 40)
    print(f"Current Bankroll: ${config.get_current_bankroll():,.2f}")
    print(f"Initial Bankroll: $10,000.00")
    
    try:
        stats = config.get_performance_stats()
        total_trades = len(config.trades) if hasattr(config, 'trades') else 0
        print(f"Total Trades: {total_trades}")
        if total_trades > 0:
            wins = sum(1 for t in config.trades if t.get('result') == 'won')
            win_rate = (wins / total_trades) * 100
            print(f"Win Rate: {win_rate:.1f}%")
        current = config.get_current_bankroll()
        roi = ((current - 10000) / 10000) * 100
        print(f"ROI: {roi:.2f}%")
        print(f"Total P&L: ${current - 10000:.2f}")
    except:
        # No trades yet
        print(f"Total Trades: 0")
        print(f"Win Rate: 0.0%")
        print(f"ROI: 0.00%")
        print(f"Total P&L: $0.00")
    
def show_recent_trades():
    """Display recent trades"""
    print("\n📊 Recent Trades")
    print("=" * 40)
    
    with db_manager.get_db_session() as db:
        recent_bets = db.query(Bet).order_by(
            Bet.created_at.desc()
        ).limit(5).all()
        
        if not recent_bets:
            print("No trades yet - the system will place trades when it finds positive edge opportunities!")
            return
            
        for bet in recent_bets:
            # Get market info from source_id
            market = db.query(Market).filter(Market.source_id == bet.source_id).first()
            if not market:
                continue
                
            status_icon = "⏳"  # Bet model doesn't have status field
            print(f"{status_icon} {market.home_team} vs {market.away_team}")
            print(f"   Bet: {bet.normalized_outcome} @ {bet.odds:.2f}")
            print(f"   Stake: ${bet.stake:.2f}")
            print()

def show_market_opportunities():
    """Display current market opportunities"""
    print("\n🔥 Market Opportunities")
    print("=" * 40)
    
    with db_manager.get_db_session() as db:
        # Get markets with positive edge
        markets = db.query(Market).filter(
            Market.is_active == True
        ).order_by(Market.maturity_date.asc()).limit(10).all()
        
        opportunities = 0
        for market in markets:
            odds = db.query(Odd).filter(
                Odd.source_id == market.source_id
            ).all()
            
            if len(odds) >= 2:
                # Simple edge check
                total_prob = sum(1/odd.decimal_odds for odd in odds)
                
                for odd in odds:
                    fair_prob = (1/odd.decimal_odds) / total_prob
                    fair_odds = 1 / fair_prob
                    edge = ((odd.decimal_odds / fair_odds) - 1) * 100
                    
                    if edge > 2:  # Positive edge threshold
                        print(f"✅ {market.home_team} vs {market.away_team}")
                        print(f"   {odd.outcome}: {odd.decimal_odds} (Edge: {edge:.2f}%)")
                        opportunities += 1
                        break
                        
        if opportunities == 0:
            print("No positive edge opportunities at the moment.")
            print("The system continuously scans for profitable bets...")

def main():
    print("🎮 Ominari Paper Trading Demo")
    print("=" * 50)
    print()
    print("This demo shows how the paper trading system works:")
    print("• Real-time market data from blockchain")
    print("• Automated edge calculation")
    print("• Kelly criterion bet sizing")
    print("• Portfolio tracking")
    print("• Risk management")
    print()
    
    # Check Discord config
    webhook_configured = bool(os.getenv('DISCORD_WEBHOOK_URL'))
    if os.path.exists('config/discord_config.json'):
        import json
        with open('config/discord_config.json', 'r') as f:
            config = json.load(f)
            webhook_configured = bool(config.get('webhook_url'))
    
    if webhook_configured:
        print("✅ Discord notifications configured")
        print("   You'll receive alerts for:")
        print("   • New trades placed")
        print("   • Trade results (wins/losses)")
        print("   • Daily portfolio summaries")
        print("   • High opportunity markets")
    else:
        print("⚠️  Discord not configured")
        print("   Run: ./scripts/setup_discord.sh")
        print("   to receive trade notifications")
    
    # Show current status
    show_portfolio_status()
    show_recent_trades()
    show_market_opportunities()
    
    print("\n💡 The system is running at: http://localhost:8888")
    print("   • Dashboard shows live markets and edges")
    print("   • Trades are placed automatically when edges > 2%")
    print("   • All trades use paper money (no real funds)")
    print("   • Portfolio updates in real-time")
    
    print("\n📈 What happens next:")
    print("   1. System scans markets every minute")
    print("   2. Calculates edge for each outcome")
    print("   3. Places bets on positive edge markets")
    print("   4. Tracks results and updates portfolio")
    print("   5. Sends Discord notifications (if configured)")
    
    print("\n🎯 Ready for real trading?")
    print("   1. Configure wallet: ./scripts/setup_wallet.sh")
    print("   2. Fund with USDC/sUSD")
    print("   3. System automatically switches to real mode")
    print("   4. All safety features activate")
    print("   5. Start with testnet for practice!")

if __name__ == "__main__":
    main()