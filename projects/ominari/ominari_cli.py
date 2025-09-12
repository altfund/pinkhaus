#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ominari CLI - Command line interface for managing the Ominari trading system
"""

import click
import subprocess
import signal
import sys
import time
import os
from pathlib import Path
import asyncio
import aiofiles
from datetime import datetime
import requests
import json

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
RESET = '\033[0m'
BOLD = '\033[1m'

# Process tracking
processes = {}

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully."""
    print(f"\n{YELLOW}Shutting down services...{RESET}")
    stop_all_services()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def print_banner():
    """Print Ominari CLI banner."""
    banner = f"""
{CYAN}{BOLD}╔═══════════════════════════════════════════╗
║        OMINARI TRADING SYSTEM CLI         ║
╚═══════════════════════════════════════════╝{RESET}
"""
    print(banner)

def log_with_timestamp(message, color=RESET):
    """Print log message with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{color}[{timestamp}] {message}{RESET}")

def start_service(name, command, log_file=None):
    """Start a service with optional log file."""
    try:
        if log_file:
            with open(log_file, 'a') as f:
                proc = subprocess.Popen(
                    command,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    shell=True,
                    preexec_fn=os.setsid
                )
        else:
            proc = subprocess.Popen(
                command,
                shell=True,
                preexec_fn=os.setsid
            )
        
        processes[name] = proc
        log_with_timestamp(f"Started {name} (PID: {proc.pid})", GREEN)
        return proc
    except Exception as e:
        log_with_timestamp(f"Failed to start {name}: {e}", RED)
        return None

def stop_service(name):
    """Stop a specific service."""
    if name in processes:
        proc = processes[name]
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=5)
            log_with_timestamp(f"Stopped {name}", YELLOW)
        except:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            log_with_timestamp(f"Force stopped {name}", RED)
        del processes[name]

def stop_all_services():
    """Stop all running services."""
    for name in list(processes.keys()):
        stop_service(name)

async def tail_log_file(file_path, prefix, color):
    """Asynchronously tail a log file and print with prefix."""
    try:
        async with aiofiles.open(file_path, 'r') as f:
            # Go to end of file
            await f.seek(0, 2)
            while True:
                line = await f.readline()
                if line:
                    print(f"{color}{prefix}{RESET} {line.rstrip()}")
                else:
                    await asyncio.sleep(0.1)
    except FileNotFoundError:
        await asyncio.sleep(1)  # Wait for file to be created
        await tail_log_file(file_path, prefix, color)
    except Exception as e:
        log_with_timestamp(f"Error tailing {file_path}: {e}", RED)

async def stream_logs():
    """Stream logs from all services."""
    log_tasks = []
    
    # Define log files and their display properties
    log_files = [
        ("main.log", "[API]", BLUE),
        ("web_monitor.log", "[WEB]", GREEN),
        ("paper_trading.log", "[PAPER]", CYAN),
        ("scheduler_output.log", "[SCHEDULER]", YELLOW),
    ]
    
    # Create tasks for each log file
    for file_path, prefix, color in log_files:
        task = asyncio.create_task(tail_log_file(file_path, prefix, color))
        log_tasks.append(task)
    
    # Wait for all tasks (they run forever)
    await asyncio.gather(*log_tasks)

@click.group()
def cli():
    """Ominari Trading System CLI"""
    pass

@cli.command()
@click.option('--mode', default='all', help='Services to start: all, api, web, paper')
@click.option('--stream-logs', 'stream_logs_flag', is_flag=True, help='Stream logs to console')
def start(mode, stream_logs_flag):
    """Start Ominari services."""
    print_banner()
    
    # Ensure we're in the right directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    log_with_timestamp("Starting Ominari Trading System", BOLD)
    
    # Start services based on mode
    if mode in ['all', 'api']:
        start_service(
            'api',
            'uv run python main.py',
            'main.log' if not stream_logs_flag else None
        )
        time.sleep(2)  # Give API time to start
    
    if mode in ['all', 'web']:
        start_service(
            'web',
            'uv run python web_monitor.py',
            'web_monitor.log' if not stream_logs_flag else None
        )
        time.sleep(1)
    
    if mode in ['all', 'paper']:
        # Paper trading is handled by main.py, just log its status
        log_with_timestamp("Paper trading enabled (runs within API)", CYAN)
    
    # Show service status
    print(f"\n{BOLD}Active Services:{RESET}")
    for name, proc in processes.items():
        print(f"  • {name}: PID {proc.pid}")
    
    print(f"\n{CYAN}Dashboard: http://localhost:8888{RESET}")
    print(f"{BLUE}API Docs: http://localhost:8000/docs{RESET}")
    
    if stream_logs_flag:
        print(f"\n{BOLD}Streaming logs...{RESET}")
        print("Press Ctrl+C to stop all services\n")
        
        # Run log streaming in async context
        try:
            asyncio.run(stream_logs())
        except KeyboardInterrupt:
            pass
    else:
        print(f"\n{YELLOW}Services running in background.{RESET}")
        print("Use 'ominari logs' to view logs")
        print("Use 'ominari stop' to stop all services")

@cli.command()
def stop():
    """Stop all Ominari services."""
    print_banner()
    log_with_timestamp("Stopping all services", YELLOW)
    
    # Kill any existing processes by name
    services = ['main.py', 'web_monitor.py']
    
    for service in services:
        try:
            subprocess.run(['pkill', '-f', service], check=False)
        except:
            pass
    
    # Also stop our tracked processes
    stop_all_services()
    
    log_with_timestamp("All services stopped", GREEN)

@cli.command()
@click.option('--follow', '-f', is_flag=True, help='Follow log output')
@click.option('--lines', '-n', default=50, help='Number of lines to show')
@click.option('--service', '-s', help='Specific service: api, web, paper, scheduler')
def logs(follow, lines, service):
    """View logs from Ominari services."""
    print_banner()
    
    # Define log files
    log_files = {
        'api': ('main.log', '[API]', BLUE),
        'web': ('web_monitor.log', '[WEB]', GREEN),
        'paper': ('paper_trading.log', '[PAPER]', CYAN),
        'scheduler': ('scheduler_output.log', '[SCHEDULER]', YELLOW),
    }
    
    if service and service in log_files:
        files_to_show = [log_files[service]]
    else:
        files_to_show = list(log_files.values())
    
    if follow:
        # Stream logs
        print(f"{BOLD}Streaming logs...{RESET}")
        print("Press Ctrl+C to stop\n")
        
        try:
            asyncio.run(stream_logs())
        except KeyboardInterrupt:
            print("\nStopped streaming logs")
    else:
        # Show last N lines from each log
        for file_path, prefix, color in files_to_show:
            if Path(file_path).exists():
                print(f"\n{color}{BOLD}{prefix} - Last {lines} lines:{RESET}")
                try:
                    result = subprocess.run(
                        ['tail', '-n', str(lines), file_path],
                        capture_output=True,
                        text=True
                    )
                    for line in result.stdout.split('\n'):
                        if line.strip():
                            print(f"{color}{prefix}{RESET} {line}")
                except Exception as e:
                    log_with_timestamp(f"Error reading {file_path}: {e}", RED)
            else:
                print(f"\n{YELLOW}{prefix} - No log file found{RESET}")

@cli.command()
def status():
    """Check status of Ominari services."""
    print_banner()
    
    log_with_timestamp("Checking service status", BLUE)
    
    # Check if processes are running
    services = {
        'API (main.py)': 'main.py',
        'Web Monitor': 'web_monitor.py',
    }
    
    print(f"\n{BOLD}Service Status:{RESET}")
    
    for display_name, process_name in services.items():
        try:
            # Check if process is running
            result = subprocess.run(
                ['pgrep', '-f', process_name],
                capture_output=True
            )
            
            if result.returncode == 0:
                pids = result.stdout.decode().strip().split('\n')
                print(f"  • {display_name}: {GREEN}RUNNING{RESET} (PIDs: {', '.join(pids)})")
            else:
                print(f"  • {display_name}: {RED}STOPPED{RESET}")
        except:
            print(f"  • {display_name}: {YELLOW}UNKNOWN{RESET}")
    
    # Check ports
    print(f"\n{BOLD}Port Status:{RESET}")
    ports = {
        8000: "API",
        8888: "Web Dashboard"
    }
    
    for port, name in ports.items():
        try:
            result = subprocess.run(
                ['lsof', '-i', f':{port}'],
                capture_output=True
            )
            if result.returncode == 0:
                print(f"  • Port {port} ({name}): {GREEN}LISTENING{RESET}")
            else:
                print(f"  • Port {port} ({name}): {RED}NOT LISTENING{RESET}")
        except:
            print(f"  • Port {port} ({name}): {YELLOW}UNKNOWN{RESET}")

@cli.command()
def restart():
    """Restart all Ominari services."""
    print_banner()
    
    log_with_timestamp("Restarting services", YELLOW)
    
    # Stop all services
    subprocess.run([sys.executable, __file__, 'stop'], check=False)
    time.sleep(2)
    
    # Start all services
    subprocess.run([sys.executable, __file__, 'start'], check=False)

@cli.command()
@click.option('--tail', '-t', default=100, help='Number of recent trades to show')
def trades(tail):
    """View recent paper trading activity."""
    print_banner()
    
    try:
        import sqlite3
        import pandas as pd
        
        # Connect to paper trading database
        conn = sqlite3.connect('paper_trades.db')
        
        # Get recent trades
        query = """
        SELECT 
            pf.timestamp,
            po.bet_name,
            po.side,
            pf.fill_size,
            pf.fill_price,
            po.expected_edge,
            pf.slippage,
            pf.commission,
            po.signal_name
        FROM paper_fills pf
        JOIN paper_orders po ON pf.order_id = po.order_id
        ORDER BY pf.timestamp DESC
        LIMIT ?
        """
        
        df = pd.read_sql_query(query, conn, params=(tail,))
        
        if df.empty:
            print(f"{YELLOW}No paper trades found yet{RESET}")
        else:
            print(f"\n{BOLD}Recent Paper Trades:{RESET}")
            print(df.to_string(index=False))
            
            # Summary statistics
            print(f"\n{BOLD}Summary Statistics:{RESET}")
            print(f"  • Total trades: {len(df)}")
            print(f"  • Total volume: ${df['fill_size'].sum():.2f}")
            print(f"  • Avg edge: {df['expected_edge'].mean():.2%}")
            print(f"  • Total commission: ${df['commission'].sum():.2f}")
            print(f"  • Total slippage: ${df['slippage'].sum():.2f}")
        
        conn.close()
        
    except Exception as e:
        log_with_timestamp(f"Error reading trades: {e}", RED)

@cli.command()
def dashboard():
    """Open the web dashboard in browser."""
    print_banner()
    
    # Check if web monitor is running
    result = subprocess.run(['pgrep', '-f', 'web_monitor.py'], capture_output=True)
    
    if result.returncode != 0:
        log_with_timestamp("Web monitor not running. Starting it now...", YELLOW)
        subprocess.run([sys.executable, __file__, 'start', '--mode', 'web'])
        time.sleep(2)
    
    # Open browser
    import webbrowser
    url = 'http://localhost:8888'
    log_with_timestamp(f"Opening dashboard at {url}", GREEN)
    webbrowser.open(url)

# Session management commands
@cli.group()
def session():
    """Manage paper trading sessions."""
    pass

@session.command('start')
@click.option('--capital', default=10000.0, help='Initial capital for session')
@click.option('--name', help='Session name/description')
def session_start(capital, name):
    """Start a new paper trading session."""
    print_banner()
    log_with_timestamp("Starting new paper trading session", YELLOW)
    
    try:
        response = requests.post('http://localhost:8000/paper/session/start', 
                               json={"capital": capital, "name": name})
        
        if response.ok:
            data = response.json()
            log_with_timestamp(f"Session started: {data['session_id']}", GREEN)
            print(f"  Initial capital: ${capital:,.2f}")
            print(f"  Name: {name or 'Default session'}")
        else:
            log_with_timestamp(f"Failed to start session: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error starting session: {e}", RED)

@session.command('stop')
def session_stop():
    """Stop the current paper trading session."""
    print_banner()
    log_with_timestamp("Stopping paper trading session", YELLOW)
    
    try:
        response = requests.post('http://localhost:8000/paper/session/stop')
        
        if response.ok:
            data = response.json()
            log_with_timestamp(f"Session stopped: {data['session_id']}", GREEN)
            print("\n📊 Session Summary:")
            print(f"  Duration: {data.get('duration', 'N/A')}")
            print(f"  Initial capital: ${data['initial_capital']:,.2f}")
            print(f"  Final capital: ${data['final_capital']:,.2f}")
            print(f"  Total return: {data['total_return']*100:.2f}%")
            
            if 'metrics' in data:
                metrics = data['metrics']
                print("\n📈 Performance Metrics:")
                print(f"  Total trades: {metrics.get('total_trades', 0)}")
                print(f"  Win rate: {metrics.get('win_rate', 0)*100:.1f}%")
                print(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        else:
            log_with_timestamp(f"Failed to stop session: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error stopping session: {e}", RED)

@session.command('list')
@click.option('--status', help='Filter by status: active, completed, archived')
def session_list(status):
    """List all paper trading sessions."""
    print_banner()
    log_with_timestamp("Listing paper trading sessions", YELLOW)
    
    try:
        params = {"status": status} if status else {}
        response = requests.get('http://localhost:8000/paper/sessions', params=params)
        
        if response.ok:
            sessions = response.json()
            
            if not sessions:
                print("No sessions found")
                return
                
            print(f"\n{'Session ID':<15} {'Status':<12} {'Started':<20} {'Capital':<12} {'Return':<10}")
            print("-" * 80)
            
            for session in sessions:
                return_pct = session.get('total_return', 0) * 100
                print(f"{session['session_id']:<15} "
                      f"{session['status']:<12} "
                      f"{session['start_time'][:19]:<20} "
                      f"${session['initial_capital']:<11,.0f} "
                      f"{return_pct:>9.2f}%")
                      
        else:
            log_with_timestamp(f"Failed to list sessions: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error listing sessions: {e}", RED)

@session.command('info')
def session_info():
    """Get current session information."""
    print_banner()
    log_with_timestamp("Getting session info", YELLOW)
    
    try:
        response = requests.get('http://localhost:8000/paper/session')
        
        if response.ok:
            data = response.json()
            
            if data.get('status') == 'No active session':
                log_with_timestamp("No active session", YELLOW)
                return
                
            print(f"\n📊 Current Session: {data['session_id']}")
            print(f"  Status: {data['status']}")
            print(f"  Started: {data['start_time']}")
            print(f"  Initial capital: ${data['initial_capital']:,.2f}")
            print(f"  Current capital: ${data.get('current_capital', 0):,.2f}")
            print(f"  Portfolio value: ${data.get('portfolio_value', 0):,.2f}")
            print(f"  Open positions: {data.get('open_positions', 0)}")
            print(f"  Total trades: {data.get('total_trades', 0)}")
            
        else:
            log_with_timestamp(f"Failed to get session info: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error getting session info: {e}", RED)

@session.command('archive')
@click.argument('session_id')
@click.option('--reason', help='Reason for archiving')
def session_archive(session_id, reason):
    """Archive a completed session."""
    print_banner()
    log_with_timestamp(f"Archiving session {session_id}", YELLOW)
    
    try:
        data = {"archive_reason": reason} if reason else {}
        response = requests.post(f'http://localhost:8000/paper/session/{session_id}/archive', 
                               json=data)
        
        if response.ok:
            result = response.json()
            log_with_timestamp("Session archived successfully", GREEN)
            print("\n📦 Archived Session Details:")
            
            session = result['session']
            print(f"  Session ID: {session['session_id']}")
            print(f"  Status: {session['status']}")
            print(f"  Initial Capital: ${session['initial_capital']:,.2f}")
            
            if session.get('final_capital'):
                print(f"  Final Capital: ${session['final_capital']:,.2f}")
                return_pct = (session['final_capital'] - session['initial_capital']) / session['initial_capital'] * 100
                print(f"  Return: {return_pct:.2f}%")
            
            # Parse metadata for archive info
            if session.get('metadata'):
                metadata = json.loads(session['metadata'])
                print("\n📋 Archive Info:")
                print(f"  Archived at: {metadata.get('archived_at', 'N/A')}")
                print(f"  Reason: {metadata.get('archive_reason', 'N/A')}")
        else:
            log_with_timestamp(f"Failed to archive session: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error archiving session: {e}", RED)

# Analytics commands
@cli.group()
def analytics():
    """View portfolio analytics."""
    pass

@analytics.command('exposure')
def analytics_exposure():
    """Show portfolio exposure analysis."""
    print_banner()
    log_with_timestamp("Analyzing portfolio exposure", YELLOW)
    
    try:
        response = requests.get('http://localhost:8000/paper/analytics/exposure')
        
        if response.ok:
            data = response.json()
            
            if 'message' in data:
                print(data['message'])
                return
                
            print("\n📊 Portfolio Exposure Analysis")
            
            exposure = data.get('total_exposure', {})
            print(f"\n  Total Exposure: {exposure.get('total_exposure', 0)*100:.1f}%")
            print(f"  Position Count: {exposure.get('position_count', 0)}")
            print(f"  Largest Position: {exposure.get('largest_position', 0)*100:.1f}%")
            
            print("\n  By Sport:")
            for sport, info in data.get('by_sport', {}).items():
                print(f"    {sport}: {info['count']} positions, {info['exposure_pct']*100:.1f}% exposure")
            
            print("\n  Risk Utilization:")
            risk = data.get('risk_utilization', {})
            print(f"    Position Limit: {risk.get('position_limit', 0)*100:.0f}%")
            print(f"    Total Limit: {risk.get('total_limit', 0)*100:.0f}%")
            
        else:
            log_with_timestamp(f"Failed to get exposure data: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error getting exposure data: {e}", RED)

@analytics.command('performance')
def analytics_performance():
    """Show performance analytics."""
    print_banner()
    log_with_timestamp("Analyzing performance", YELLOW)
    
    try:
        response = requests.get('http://localhost:8000/paper/analytics/performance')
        
        if response.ok:
            data = response.json()
            
            print("\n📈 Performance Analytics")
            
            metrics = data.get('overall_metrics', {})
            print("\n  Overall Metrics:")
            print(f"    Total P&L: ${metrics.get('total_pnl', 0):,.2f}")
            print(f"    Win Rate: {metrics.get('win_rate', 0)*100:.1f}%")
            print(f"    Total Trades: {metrics.get('total_trades', 0)}")
            
            print("\n  Trading Activity by Hour:")
            for hour_data in data.get('hourly_breakdown', []):
                print(f"    {hour_data['hour']:02d}:00 - {hour_data['trades']} trades")
            
            print("\n  Performance by Signal:")
            for signal in data.get('signal_performance', []):
                print(f"    {signal['signal']}: {signal['trades']} trades, {signal['avg_edge']*100:.2f}% avg edge")
            
        else:
            log_with_timestamp(f"Failed to get performance data: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error getting performance data: {e}", RED)

@analytics.command('rebalance')
@click.option('--execute', is_flag=True, help='Execute the rebalancing trades')
@click.option('--max-trades', type=int, help='Maximum number of trades to execute')
def analytics_rebalance(execute, max_trades):
    """Calculate and optionally execute portfolio rebalancing."""
    print_banner()
    log_with_timestamp("Calculating portfolio rebalancing", YELLOW)
    
    try:
        response = requests.post('http://localhost:8000/paper/analytics/rebalance')
        
        if response.ok:
            data = response.json()
            
            if 'message' in data:
                print(data['message'])
                return
            
            print("\n📊 Portfolio Rebalancing Analysis")
            
            summary = data.get('summary', {})
            print("\n  Summary:")
            print(f"    Positions to close: {summary.get('positions_to_close', 0)}")
            print(f"    Positions to open: {summary.get('positions_to_open', 0)}")
            print(f"    Positions to adjust: {summary.get('positions_to_adjust', 0)}")
            print(f"    Total trades needed: {summary.get('total_rebalancing_trades', 0)}")
            
            trades = data.get('suggested_trades', [])
            if trades:
                print("\n  Suggested Trades (top 10):")
                for i, trade in enumerate(trades[:10]):
                    action_symbol = {'close': '❌', 'open': '➕', 'adjust': '🔄'}.get(trade['action'], '?')
                    print(f"    {i+1}. {action_symbol} {trade['action'].upper()}: {trade['bet_name']}")
                    print(f"       Current: ${trade['current_size']:.2f} → Target: ${trade['target_size']:.2f}")
                    print(f"       Change: ${trade['size_delta']:+.2f} | {trade['reason']}")
                    
                    if trade.get('expected_edge'):
                        print(f"       Expected edge: {trade['expected_edge']*100:.2f}%")
                    print()
            
            if execute:
                print(f"\n{YELLOW}Executing rebalancing trades...{RESET}")
                
                params = {}
                if max_trades:
                    params['max_trades'] = max_trades
                
                exec_response = requests.post('http://localhost:8000/paper/analytics/rebalance/execute', params=params)
                
                if exec_response.ok:
                    exec_data = exec_response.json()
                    
                    print("\n✅ Execution Results:")
                    print(f"  Executed: {exec_data['executed_count']} trades")
                    print(f"  Failed: {exec_data['failed_count']} trades")
                    
                    if exec_data['executed_trades']:
                        print("\n  Executed Trades:")
                        for item in exec_data['executed_trades'][:5]:
                            trade = item['trade']
                            fill = item['fill']
                            print(f"    • {trade['action'].upper()} {trade['bet_name']}")
                            print(f"      Fill: ${fill['size']:.2f} @ {fill['price']:.3f} (commission: ${fill['commission']:.2f})")
                    
                    if exec_data['failed_trades']:
                        print("\n  Failed Trades:")
                        for item in exec_data['failed_trades'][:5]:
                            trade = item['trade']
                            print(f"    • {trade['action'].upper()} {trade['bet_name']} - {item['reason']}")
                    
                    log_with_timestamp(exec_data['message'], GREEN)
                else:
                    log_with_timestamp(f"Failed to execute rebalancing: {exec_response.text}", RED)
            else:
                print(f"\n{YELLOW}Use --execute flag to execute these trades{RESET}")
            
        else:
            log_with_timestamp(f"Failed to calculate rebalancing: {response.text}", RED)
            
    except Exception as e:
        log_with_timestamp(f"Error in rebalancing: {e}", RED)

if __name__ == '__main__':
    cli()