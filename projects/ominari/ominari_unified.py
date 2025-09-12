#!/usr/bin/env python3
"""
Ominari Trading System - Unified CLI

A single command to rule them all. This CLI manages the entire Ominari trading system
including web monitoring, paper trading, and system management.
"""

import os
import sys
import time
import json
import subprocess
import signal
import argparse
from datetime import datetime
from pathlib import Path
import threading
from typing import Dict, Optional

# ANSI color codes
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
CYAN = '\033[96m'
MAGENTA = '\033[95m'
RESET = '\033[0m'
BOLD = '\033[1m'

# PID file for tracking processes
PID_FILE = '.ominari_pids'


class OminariCLI:
    """Main CLI for Ominari Trading System."""
    
    def __init__(self):
        self.pids = self.load_pids()
        self.processes = {}
        self.threads = {}
        
    def load_pids(self):
        """Load saved PIDs from file."""
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_pids(self):
        """Save PIDs to file."""
        with open(PID_FILE, 'w') as f:
            json.dump(self.pids, f)
    
    def print_banner(self):
        """Print ASCII banner."""
        banner = f"""
{CYAN}╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║   {BOLD}OMINARI TRADING SYSTEM{RESET}{CYAN}                                 ║
║   Automated Sports Betting with Blockchain Oracles        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝{RESET}
"""
        print(banner)
    
    def check_prerequisites(self):
        """Check system prerequisites."""
        print(f"\n{BOLD}Checking prerequisites...{RESET}")
        
        checks = []
        
        # Check Python version
        if sys.version_info >= (3, 8):
            checks.append((True, "Python 3.8+"))
        else:
            checks.append((False, f"Python {sys.version} (need 3.8+)"))
        
        # Check database
        if os.path.exists('sport_odds.db'):
            size = os.path.getsize('sport_odds.db') / (1024**3)  # GB
            checks.append((True, f"Database found ({size:.1f}GB)"))
        else:
            checks.append((False, "Database not found"))
        
        # Check API key
        if os.getenv('OMINARI_API_KEY'):
            checks.append((True, "API key configured"))
        else:
            checks.append((False, "API key not set"))
        
        # Check Docker (for monitoring)
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
            checks.append((True, "Docker installed"))
        except:
            checks.append((False, "Docker not found (optional, for monitoring)"))
        
        # Check paper trading database
        if os.path.exists('paper_trading.db'):
            checks.append((True, "Paper trading DB exists"))
        else:
            checks.append((True, "Paper trading DB will be created"))
        
        # Display results
        all_good = True
        for passed, msg in checks:
            if passed:
                print(f"  {GREEN}✓{RESET} {msg}")
            else:
                print(f"  {RED}✗{RESET} {msg}")
                if "optional" not in msg.lower():
                    all_good = False
        
        return all_good
    
    def setup_api_auth(self):
        """Set up API authentication."""
        print(f"\n{BOLD}Setting up API authentication...{RESET}")
        
        # Check if already configured
        if os.path.exists('api_keys.json'):
            print(f"{GREEN}✓{RESET} API keys already configured")
            
            # Check environment
            if not os.getenv('OMINARI_API_KEY'):
                print(f"\n{YELLOW}⚠{RESET}  No API key in environment")
                print("\nSetting it for this session...")
                
                # Load first key
                with open('api_keys.json', 'r') as f:
                    keys = json.load(f)
                    if keys['keys']:
                        first_key = keys['keys'][0]['key']
                        
                        # Set it for this session
                        os.environ['OMINARI_API_KEY'] = first_key
                        print(f"{GREEN}✓{RESET} Set API key for this session")
            return
        
        # Create API keys
        print("Creating API keys...")
        subprocess.run([sys.executable, 'api_auth.py', 'create', 
                       '--name', 'Default Key', 
                       '--permissions', 'read', 'trade', 
                       '--rate-limit', '120'])
    
    def start_web_monitor(self):
        """Start the web monitoring dashboard."""
        print(f"\n{BOLD}Starting web monitor...{RESET}")
        
        # Check if already running
        if 'web_monitor' in self.pids:
            pid = self.pids['web_monitor']
            try:
                os.kill(pid, 0)  # Check if process exists
                print(f"{YELLOW}⚠{RESET}  Web monitor already running (PID: {pid})")
                return
            except:
                # Process not running, remove from pids
                del self.pids['web_monitor']
        
        # Start web monitor with metrics
        if os.path.exists('web_monitor_with_metrics.py'):
            cmd = [sys.executable, 'web_monitor_with_metrics.py']
        else:
            cmd = [sys.executable, 'web_monitor.py']
        
        # Start in background
        process = subprocess.Popen(cmd, 
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        
        self.pids['web_monitor'] = process.pid
        self.processes['web_monitor'] = process
        self.save_pids()
        
        # Wait for it to start
        time.sleep(3)
        
        try:
            os.kill(process.pid, 0)
            print(f"{GREEN}✓{RESET} Web monitor started (PID: {process.pid})")
            print(f"  Dashboard: {BLUE}http://localhost:8888{RESET}")
            print(f"  API Docs: {BLUE}http://localhost:8888/api/docs{RESET}")
        except:
            print(f"{RED}✗{RESET} Failed to start web monitor")
    
    def start_monitoring_stack(self):
        """Start Prometheus and Grafana."""
        print(f"\n{BOLD}Starting monitoring stack...{RESET}")
        
        # Check if Docker is available
        try:
            subprocess.run(['docker', '--version'], capture_output=True, check=True)
        except:
            print(f"{YELLOW}⚠{RESET}  Docker not available, skipping monitoring stack")
            return
        
        if not os.path.exists('monitoring/docker-compose.yml'):
            print(f"{YELLOW}⚠{RESET}  Monitoring not set up. Setting up now...")
            subprocess.run([sys.executable, 'setup_prometheus.py'])
        
        # Start Docker containers
        try:
            os.chdir('monitoring')
            subprocess.run(['docker-compose', 'up', '-d'], check=True)
            os.chdir('..')
            
            print(f"{GREEN}✓{RESET} Monitoring stack started")
            print(f"  Prometheus: {BLUE}http://localhost:9090{RESET}")
            print(f"  Grafana: {BLUE}http://localhost:3000{RESET} (admin/ominari123)")
        except Exception as e:
            os.chdir('..')  # Make sure we're back in the right directory
            print(f"{YELLOW}⚠{RESET}  Could not start monitoring: {e}")
            print("  Continuing without Prometheus/Grafana...")
    
    def setup_paper_trading(self):
        """Set up paper trading session."""
        print(f"\n{BOLD}Setting up paper trading...{RESET}")
        
        # Run setup script
        result = subprocess.run([sys.executable, 'setup_realistic_paper_trading.py'],
                              capture_output=True, text=True)
        
        print(result.stdout)
        
        # Extract session ID from output
        session_id = None
        for line in result.stdout.split('\n'):
            if 'Session ID:' in line:
                session_id = line.split('Session ID:')[1].strip()
                break
        
        if session_id:
            self.pids['paper_session'] = session_id
            self.save_pids()
            return session_id
        else:
            print(f"{RED}✗{RESET} Failed to set up paper trading")
            return None
    
    def start_paper_trading(self, session_id: Optional[str] = None):
        """Start paper trading bot."""
        print(f"\n{BOLD}Starting paper trading bot...{RESET}")
        
        # Get session ID
        if not session_id:
            session_id = self.pids.get('paper_session')
            if not session_id:
                print(f"{RED}✗{RESET} No paper trading session found. Setting up...")
                session_id = self.setup_paper_trading()
                if not session_id:
                    return
        
        # Check if already running
        if 'paper_trader' in self.pids:
            pid = self.pids['paper_trader']
            try:
                os.kill(pid, 0)
                print(f"{YELLOW}⚠{RESET}  Paper trader already running (PID: {pid})")
                return
            except:
                del self.pids['paper_trader']
        
        # Start the trader
        cmd = [sys.executable, 'run_realistic_paper_trading.py']
        process = subprocess.Popen(cmd,
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        
        self.pids['paper_trader'] = process.pid
        self.processes['paper_trader'] = process
        self.save_pids()
        
        print(f"{GREEN}✓{RESET} Paper trading bot started (PID: {process.pid})")
        print(f"  Session: {session_id}")
        
        # Start log monitor thread
        def monitor_logs():
            for line in process.stdout:
                print(f"  {MAGENTA}[TRADER]{RESET} {line.decode().strip()}")
        
        thread = threading.Thread(target=monitor_logs, daemon=True)
        thread.start()
        self.threads['paper_trader_log'] = thread
    
    def show_status(self):
        """Show system status."""
        print(f"\n{BOLD}System Status:{RESET}")
        
        # Check each component
        components = {
            'Web Monitor': 'web_monitor',
            'Paper Trader': 'paper_trader',
        }
        
        for name, key in components.items():
            if key in self.pids:
                pid = self.pids[key]
                try:
                    os.kill(pid, 0)
                    print(f"  {GREEN}●{RESET} {name}: Running (PID {pid})")
                except:
                    print(f"  {RED}●{RESET} {name}: Stopped")
                    del self.pids[key]
                    self.save_pids()
            else:
                print(f"  {YELLOW}●{RESET} {name}: Not started")
        
        # Check Docker containers
        try:
            result = subprocess.run(['docker-compose', '-f', 'monitoring/docker-compose.yml', 'ps'],
                                  capture_output=True, text=True)
            if 'prometheus' in result.stdout and 'Up' in result.stdout:
                print(f"  {GREEN}●{RESET} Monitoring Stack: Running")
            else:
                print(f"  {YELLOW}●{RESET} Monitoring Stack: Not running")
        except:
            print(f"  {YELLOW}●{RESET} Monitoring Stack: Not available")
        
        # Show URLs
        print(f"\n{BOLD}Access URLs:{RESET}")
        print(f"  Dashboard: {BLUE}http://localhost:8888{RESET}")
        print(f"  API Docs: {BLUE}http://localhost:8888/api/docs{RESET}")
        print(f"  Prometheus: {BLUE}http://localhost:9090{RESET}")
        print(f"  Grafana: {BLUE}http://localhost:3000{RESET}")
        
        # Show paper trading session
        if 'paper_session' in self.pids:
            print(f"\n{BOLD}Paper Trading:{RESET}")
            print(f"  Session ID: {self.pids['paper_session']}")
    
    def stop_all(self):
        """Stop all services."""
        print(f"\n{BOLD}Stopping all services...{RESET}")
        
        # Stop processes
        for name, pid in list(self.pids.items()):
            if name == 'paper_session':
                continue  # Skip session ID
                
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"  {GREEN}✓{RESET} Stopped {name} (PID {pid})")
            except:
                print(f"  {YELLOW}⚠{RESET} Process {name} (PID {pid}) not found")
        
        # Stop Docker containers
        try:
            subprocess.run(['docker-compose', '-f', 'monitoring/docker-compose.yml', 'down'],
                         capture_output=True)
            print(f"  {GREEN}✓{RESET} Stopped monitoring stack")
        except:
            pass
        
        # Clear PIDs (except session)
        session_id = self.pids.get('paper_session')
        self.pids = {}
        if session_id:
            self.pids['paper_session'] = session_id
        self.save_pids()
    
    def open_dashboard(self):
        """Open dashboard in browser."""
        import webbrowser
        
        # Make sure web monitor is running
        if 'web_monitor' not in self.pids:
            self.start_web_monitor()
            time.sleep(2)
        
        print(f"\n{BOLD}Opening dashboard...{RESET}")
        webbrowser.open('http://localhost:8888')
    
    def show_live_monitor(self):
        """Show live monitoring dashboard."""
        print(f"\n{BOLD}Live System Monitor{RESET}")
        print("Press Ctrl+C to exit\n")
        
        try:
            while True:
                # Clear screen
                print('\033[2J\033[H', end='')
                
                # Header
                print(f"{CYAN}╔═══════════════════════════════════════════════════════════╗")
                print(f"║  OMINARI LIVE MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}      ║")
                print(f"╚═══════════════════════════════════════════════════════════╝{RESET}\n")
                
                # System status
                self.show_status()
                
                # Paper trading stats
                if 'paper_session' in self.pids:
                    session_id = self.pids['paper_session']
                    try:
                        from paper_trading_sessions import PaperTradingSessionManager
                        manager = PaperTradingSessionManager()
                        session = manager.get_session(session_id)
                        
                        if session:
                            print(f"\n{BOLD}Paper Trading Performance:{RESET}")
                            capital = session['current_capital']
                            initial = session['initial_capital']
                            pnl = capital - initial
                            pnl_pct = (pnl / initial) * 100
                            
                            color = GREEN if pnl >= 0 else RED
                            print(f"  Capital: ${capital:,.2f}")
                            print(f"  P&L: {color}${pnl:,.2f} ({pnl_pct:+.2f}%){RESET}")
                    except:
                        pass
                
                time.sleep(5)
                
        except KeyboardInterrupt:
            print(f"\n\n{YELLOW}Exiting monitor...{RESET}")
    
    def quick_start(self):
        """Quick start everything with one command."""
        print(f"\n{BOLD}🚀 QUICK START MODE{RESET}")
        print("Starting all components...\n")
        
        # 1. Check prerequisites
        if not self.check_prerequisites():
            print(f"\n{RED}Please fix prerequisites first!{RESET}")
            print("\nTo fix:")
            print("  - Database: Run free_data_pull.py to create it")
            print("  - API Key: Will be created automatically")
            return
        
        # 2. Set up API auth
        self.setup_api_auth()
        
        # 3. Start web monitor
        self.start_web_monitor()
        
        # 4. Start monitoring stack (optional)
        self.start_monitoring_stack()
        
        # 5. Set up and start paper trading
        self.start_paper_trading()
        
        # 6. Wait a moment for everything to settle
        time.sleep(2)
        
        # 7. Show final status
        self.show_status()
        
        print(f"\n{GREEN}✨ ALL SYSTEMS OPERATIONAL!{RESET}")
        print(f"\nOpen {BLUE}http://localhost:8888{RESET} to view the dashboard")
        print(f"\nPress Ctrl+C to open live monitor, Ctrl+C again to stop all services")
        
        # Keep running with live updates
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            # First Ctrl+C - show live monitor
            try:
                self.show_live_monitor()
            except KeyboardInterrupt:
                # Second Ctrl+C - shutdown
                print(f"\n\n{YELLOW}Shutting down...{RESET}")
                self.stop_all()
                print(f"{GREEN}✓ All services stopped{RESET}")
    
    def run(self):
        """Main CLI entry point."""
        parser = argparse.ArgumentParser(
            description='Ominari Trading System - Unified CLI',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=f"""
{BOLD}Examples:{RESET}
  {GREEN}ominari{RESET}                # Quick start everything (recommended)
  {GREEN}ominari status{RESET}         # Show system status
  {GREEN}ominari dashboard{RESET}      # Open web dashboard
  {GREEN}ominari monitor{RESET}        # Show live monitoring
  {GREEN}ominari stop{RESET}           # Stop all services
  
{BOLD}Components:{RESET}
  • Web Dashboard - Real-time trading interface
  • Paper Trading - Realistic simulation with blockchain data
  • Monitoring - Prometheus & Grafana metrics
  • API Server - RESTful API with authentication
            """
        )
        
        parser.add_argument('command', nargs='?', default='start',
                          choices=['start', 'stop', 'status', 'restart', 
                                 'dashboard', 'monitor', 'logs'],
                          help='Command to run (default: start)')
        
        args = parser.parse_args()
        
        # Show banner
        self.print_banner()
        
        # Execute command
        if args.command == 'start':
            self.quick_start()
        elif args.command == 'stop':
            self.stop_all()
        elif args.command == 'status':
            self.show_status()
        elif args.command == 'restart':
            self.stop_all()
            time.sleep(2)
            self.quick_start()
        elif args.command == 'dashboard':
            self.open_dashboard()
        elif args.command == 'monitor':
            self.show_live_monitor()
        elif args.command == 'logs':
            # Show recent logs
            print(f"\n{BOLD}Recent Logs:{RESET}")
            log_files = [
                ('Web Monitor', 'web_monitor.log'),
                ('Paper Trading', 'paper_trading.log'),
                ('Migration', 'migration_output.log'),
            ]
            for name, log_file in log_files:
                if os.path.exists(log_file):
                    print(f"\n{BOLD}=== {name} ==={RESET}")
                    subprocess.run(['tail', '-n', '10', log_file])


def main():
    """Main entry point."""
    cli = OminariCLI()
    cli.run()


if __name__ == "__main__":
    main()