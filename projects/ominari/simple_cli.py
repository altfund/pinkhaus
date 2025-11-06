#!/usr/bin/env python3
"""
Simple Ominari Trading System CLI
Basic command-line interface for managing the trading system
"""

import os
import sys
import subprocess
import signal
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

def print_header():
    """Print CLI header"""
    print("🏛️  OMINARI TRADING SYSTEM CLI")
    print("=" * 50)

def start_trading(continuous=False):
    """Start the terminal trading system"""
    script_name = 'continuous_terminal_trading.py' if continuous else 'simple_terminal_trading.py'
    print(f"🚀 Starting {'continuous' if continuous else 'terminal'} trading system...")
    
    # Check if already running
    try:
        result = subprocess.run(['pgrep', '-f', script_name], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("⚠️  Trading system is already running!")
            return
    except:
        pass
    
    # Start the trading system
    cmd = [
        'uv', 'run', '--no-project', 
        '--with', 'pandas', 
        '--with', 'psycopg2-binary', 
        '--with', 'numpy', 
        '--with', 'sqlalchemy', 
        '--with', 'alembic',
        'python3', script_name
    ]
    
    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    
    print(f"Command: {' '.join(cmd)}")
    print("✅ Started trading system in background")
    print("💡 Use 'python3 simple_cli.py status' to check if it's running")
    print("💡 Use 'python3 simple_cli.py stop' to stop the system")
    
    # Start in background
    subprocess.Popen(cmd, env=env)

def stop_trading():
    """Stop the terminal trading system"""
    print("🛑 Stopping terminal trading system...")
    
    try:
        stopped_any = False
        # Find the process (both simple and continuous)
        for script in ['simple_terminal_trading', 'continuous_terminal_trading']:
            result = subprocess.run(['pgrep', '-f', script], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                pids = result.stdout.strip().split('\n')
                for pid in pids:
                    if pid:
                        print(f"Stopping process {pid} ({script})...")
                        os.kill(int(pid), signal.SIGTERM)
                        stopped_any = True
                        time.sleep(2)
                        
                        # Check if still running
                        try:
                            os.kill(int(pid), 0)  # Check if process exists
                            print(f"Force killing process {pid}...")
                            os.kill(int(pid), signal.SIGKILL)
                        except OSError:
                            pass  # Process already dead
        
        if stopped_any:
            print("✅ Trading system stopped")
        else:
            print("❌ No trading system process found")
        
    except Exception as e:
        print(f"❌ Error stopping trading system: {e}")

def show_status():
    """Show trading system status"""
    print("📊 Trading System Status")
    print("-" * 30)
    
    # Check if process is running
    try:
        result = subprocess.run(['pgrep', '-f', 'simple_terminal_trading'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            print(f"✅ Trading system is RUNNING (PID: {', '.join(pids)})")
        else:
            print("❌ Trading system is STOPPED")
    except:
        print("❌ Unable to check process status")
    
    # Try to get session info
    try:
        sys.path.insert(0, '.')
        from paper_trading_postgres_integrated import PaperTradingSessionManager
        
        session_manager = PaperTradingSessionManager()
        session_id = session_manager.get_current_session()
        
        if session_id:
            session = session_manager.get_session(session_id)
            print(f"\n📋 Current Session: {session_id[:12]}...")
            print(f"💰 Bankroll: ${session['current_bankroll']:,.2f}")
            print(f"📈 Open Positions: {session['open_positions']}")
            print(f"💸 Exposure: ${session['exposure']:.2f}")
        else:
            print("\n❌ No active session found")
            
    except Exception as e:
        print(f"\n❌ Error getting session info: {e}")

def show_positions():
    """Show current positions"""
    print("📈 Current Positions")
    print("-" * 30)
    
    try:
        sys.path.insert(0, '.')
        from paper_trading_postgres_integrated import PaperTradingSessionManager
        
        session_manager = PaperTradingSessionManager()
        session_id = session_manager.get_current_session()
        
        if not session_id:
            print("❌ No active session found")
            return
        
        positions = session_manager.get_positions(session_id)
        open_positions = [p for p in positions if p['status'] in ['pending', 'open']]
        
        if not open_positions:
            print("📊 No open positions")
            return
        
        total_stake = sum(float(p['stake']) for p in open_positions)
        print(f"Total Open: {len(open_positions)} | Total Staked: ${total_stake:.2f}\n")
        
        for i, pos in enumerate(open_positions, 1):
            home_team = pos.get('home_team', 'Unknown')
            away_team = pos.get('away_team', 'Unknown')
            bet_on = pos.get('bet_on', '').upper()
            odds = pos.get('odds', 0)
            stake = pos.get('stake', 0)
            placed_at = pos.get('placed_at', '')
            
            print(f"{i:2d}. {bet_on} {home_team} vs {away_team}")
            print(f"    Odds: {odds:.2f} | Stake: ${stake:.2f} | Placed: {placed_at}")
            print()
            
    except Exception as e:
        print(f"❌ Error getting positions: {e}")

def show_stats():
    """Show performance statistics"""
    print("📊 Performance Statistics")
    print("-" * 30)
    
    try:
        sys.path.insert(0, '.')
        from paper_trading_postgres_integrated import PaperTradingSessionManager
        
        session_manager = PaperTradingSessionManager()
        session_id = session_manager.get_current_session()
        
        if not session_id:
            print("❌ No active session found")
            return
        
        performance = session_manager.get_enhanced_performance_analytics(session_id)
        
        if not performance:
            print("❌ No performance data available")
            return
        
        overview = performance.get('overview', {})
        financial = performance.get('financial', {})
        
        print(f"Total Trades: {overview.get('total_trades', 0)}")
        print(f"Win Rate: {overview.get('win_rate', 0):.1%}")
        print(f"ROI: {financial.get('roi', 0):.2%}")
        print(f"Profit Factor: {financial.get('profit_factor', 0):.2f}")
        print(f"Sharpe Ratio: {financial.get('sharpe_ratio', 0):.2f}")
        print(f"Max Drawdown: {financial.get('max_drawdown', 0):.1%}")
        print(f"Average Win: ${financial.get('avg_win', 0):.2f}")
        print(f"Average Loss: ${financial.get('avg_loss', 0):.2f}")
        print(f"Consecutive Wins: {financial.get('consecutive_wins', 0)}")
        print(f"Consecutive Losses: {financial.get('consecutive_losses', 0)}")
        
        # Performance by sport
        trades_by_sport = financial.get('trades_by_sport', {})
        if trades_by_sport:
            print(f"\n📊 Performance by Sport:")
            for sport, data in trades_by_sport.items():
                print(f"  {sport}: {data['count']} trades | {data['win_rate']:.1f}% WR | ${data['total_pnl']:.2f} P&L")
        
        # Performance by outcome
        trades_by_outcome = financial.get('trades_by_outcome', {})
        if trades_by_outcome:
            print(f"\n🎯 Performance by Outcome:")
            for outcome, data in trades_by_outcome.items():
                print(f"  {outcome.upper()}: {data['count']} trades | {data['win_rate']:.1f}% WR | ${data['total_pnl']:.2f} P&L")
        
    except Exception as e:
        print(f"❌ Error getting statistics: {e}")

def tail_logs():
    """Show recent log output"""
    print("📋 Recent Trading Logs")
    print("-" * 30)
    
    log_files = [
        'trading_system.log',
        'terminal_trading.log'
    ]
    
    found_logs = False
    for log_file in log_files:
        if os.path.exists(log_file):
            found_logs = True
            print(f"\n📄 {log_file}:")
            try:
                result = subprocess.run(['tail', '-20', log_file], 
                                      capture_output=True, text=True)
                print(result.stdout)
            except:
                print(f"❌ Error reading {log_file}")
    
    if not found_logs:
        print("❌ No log files found")

def show_markets():
    """Show available markets"""
    print("🎯 Available Markets")
    print("-" * 30)
    
    try:
        sys.path.insert(0, '.')
        
        # Import with dependencies
        import subprocess
        result = subprocess.run([
            'uv', 'run', '--no-project', 
            '--with', 'pandas', '--with', 'psycopg2-binary', 
            '--with', 'numpy', '--with', 'sqlalchemy', '--with', 'alembic',
            'python3', '-c', '''
import os
os.environ["PYTHONPATH"] = "."
os.environ["PG_HOST"] = "localhost"
os.environ["PG_PORT"] = "5999"
os.environ["PG_USER"] = "ominari_user"
os.environ["PG_PASSWORD"] = "ominari_2025_secure"
os.environ["PG_DB"] = "ominari_production"
os.environ["USE_POSTGRESQL"] = "1"

from database_v2 import db_manager
from models import Market
from datetime import datetime, timezone, timedelta

with db_manager.get_db_session() as db:
    markets = db.query(Market).filter(
        Market.maturity_date > datetime.now(timezone.utc),
        Market.maturity_date < datetime.now(timezone.utc) + timedelta(hours=24),
        Market.is_finished == False,
        Market.sport == "Soccer"
    ).order_by(Market.maturity_date).limit(20).all()
    
    if not markets:
        print("❌ No upcoming markets found")
    else:
        print(f"Found {len(markets)} upcoming soccer matches:")
        print()
        
        for i, market in enumerate(markets, 1):
            time_to_match = market.maturity_date - datetime.now(timezone.utc)
            hours = time_to_match.total_seconds() / 3600
            
            if hours < 1:
                time_str = f"{int(hours * 60)}m"
            elif hours < 24:
                time_str = f"{hours:.1f}h"
            else:
                time_str = f"{int(hours / 24)}d"
            
            print(f"{i:2d}. {market.home_team} vs {market.away_team}")
            print(f"    Time: {time_str} | Source: {market.source}")
            print()
            '''
        ], capture_output=True, text=True, env=os.environ.copy())
        
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"❌ Error getting markets: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error getting markets: {e}")

def show_help():
    """Show help information"""
    print("""
🏛️  OMINARI TRADING SYSTEM CLI

COMMANDS:
  start       Start the terminal trading system
  start-c     Start CONTINUOUS trading (time-chunked optimization)
  stop        Stop the trading system
  status      Show system status and session info
  positions   Show current trading positions
  stats       Show performance statistics
  tail        Show recent log output
  markets     Show available markets
  help        Show this help message

EXAMPLES:
  python3 simple_cli.py start         # Basic trading
  python3 simple_cli.py start-c       # Continuous optimization
  python3 simple_cli.py status
  python3 simple_cli.py positions
  python3 simple_cli.py stop

SYSTEM INFO:
  Database: PostgreSQL on localhost:5999
  Database Name: ominari_production
  Strategy: Kelly criterion with 25% fraction
  Max Positions: 20
  Sports Focus: Soccer only
  
CONTINUOUS MODE:
  - Groups markets into 2-hour time chunks
  - Rebalances portfolio when prices change
  - Closes positions for better opportunities
  - Optimizes across multiple time windows
    """)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Ominari Trading System CLI')
    parser.add_argument('command', nargs='?', default='help',
                       choices=['start', 'start-c', 'stop', 'status', 'positions', 'stats', 'tail', 'markets', 'help'],
                       help='Command to execute')
    
    args = parser.parse_args()
    
    print_header()
    
    if args.command == 'start':
        start_trading(continuous=False)
    elif args.command == 'start-c':
        start_trading(continuous=True)
    elif args.command == 'stop':
        stop_trading()
    elif args.command == 'status':
        show_status()
    elif args.command == 'positions':
        show_positions()
    elif args.command == 'stats':
        show_stats()
    elif args.command == 'tail':
        tail_logs()
    elif args.command == 'markets':
        show_markets()
    elif args.command == 'help':
        show_help()
    else:
        show_help()

if __name__ == "__main__":
    main()