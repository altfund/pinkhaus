#!/usr/bin/env python3
"""
Unified entry point for Ominari DApp
"""

import os
import sys
import subprocess
import time
import signal
import webbrowser
from datetime import datetime

class OminariLauncher:
    def __init__(self):
        self.processes = []
        self.setup_environment()
        
    def setup_environment(self):
        """Set up environment variables"""
        os.environ['USE_POSTGRESQL'] = '1'
        os.environ['PG_HOST'] = 'localhost'
        os.environ['PG_PORT'] = '5999'
        os.environ['PG_USER'] = 'ominari_user'
        os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
        os.environ['PG_DB'] = 'ominari_production'
        
    def print_banner(self):
        """Print welcome banner"""
        print("\n" + "="*60)
        print("🚀 OMINARI DAPP - LOCAL ENVIRONMENT")
        print("="*60)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60 + "\n")
        
    def check_dependencies(self):
        """Check if required dependencies are available"""
        issues = []
        
        # Check PostgreSQL
        try:
            import psycopg2
            from database_v2 import db_manager
            with db_manager.get_db_session() as db:
                db.execute("SELECT 1")
            print("✅ PostgreSQL connection: OK")
        except Exception as e:
            issues.append(f"PostgreSQL: {str(e)}")
            print("❌ PostgreSQL connection: FAILED")
            
        # Check if dashboard is already running
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        dashboard_running = sock.connect_ex(('localhost', 8888)) == 0
        sock.close()
        
        if dashboard_running:
            print("ℹ️  Dashboard already running on port 8888")
        
        # Check Node.js
        try:
            subprocess.run(['node', '--version'], capture_output=True, check=True)
            print("✅ Node.js: Available")
            self.node_available = True
        except:
            print("⚠️  Node.js: Not available (blockchain features disabled)")
            self.node_available = False
            
        return len(issues) == 0, dashboard_running
        
    def start_dashboard(self):
        """Start the web dashboard"""
        print("\n📊 Starting Dashboard...")
        try:
            # Check if already running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if sock.connect_ex(('localhost', 8888)) == 0:
                print("   Dashboard already running!")
                sock.close()
                return True
            sock.close()
            
            # Start new dashboard
            process = subprocess.Popen(
                ['python', 'web_dashboard_real_odds.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(process)
            
            # Wait for startup
            time.sleep(3)
            
            if process.poll() is None:
                print("   ✅ Dashboard started at http://localhost:8888")
                return True
            else:
                print("   ❌ Dashboard failed to start")
                return False
                
        except Exception as e:
            print(f"   ❌ Error starting dashboard: {e}")
            return False
            
    def start_blockchain(self):
        """Start local blockchain if Node.js available"""
        if not self.node_available:
            print("\n⚠️  Skipping blockchain (Node.js not available)")
            return False
            
        print("\n⛓️  Starting Local Blockchain...")
        try:
            process = subprocess.Popen(
                ['npm', 'run', 'blockchain:start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            self.processes.append(process)
            print("   ✅ Blockchain starting...")
            return True
        except Exception as e:
            print(f"   ❌ Blockchain error: {e}")
            return False
            
    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "="*60)
        print("📋 OMINARI SERVICES")
        print("="*60)
        print("✅ Dashboard: http://localhost:8888")
        print("✅ Database: PostgreSQL on port 5999")
        if self.node_available:
            print("✅ Blockchain: Ready to deploy contracts")
        else:
            print("⚠️  Blockchain: Install Node.js for full features")
            
        print("\n" + "="*60)
        print("🎮 QUICK ACTIONS")
        print("="*60)
        print("1. Open Dashboard in Browser")
        print("2. Run Portfolio Trading Test")
        print("3. Check Real Odds in Database")
        print("4. Run Paper Trading Session")
        print("5. View System Status")
        if self.node_available:
            print("6. Deploy Smart Contracts")
            print("7. Run Blockchain Tests")
        print("0. Exit")
        print("="*60)
        
    def handle_action(self, choice):
        """Handle menu choice"""
        if choice == '1':
            print("\n🌐 Opening dashboard...")
            webbrowser.open('http://localhost:8888')
            
        elif choice == '2':
            print("\n📊 Running portfolio trading test...")
            subprocess.run(['python', 'portfolio_trading_engine.py'])
            
        elif choice == '3':
            print("\n🔍 Checking real odds...")
            subprocess.run(['python', 'check_real_odds_db.py'])
            
        elif choice == '4':
            print("\n💰 Starting paper trading...")
            subprocess.run(['python', 'paper_trading_postgres_integrated.py'])
            
        elif choice == '5':
            print("\n📈 System Status:")
            subprocess.run(['python', 'test_local_simple.py'])
            
        elif choice == '6' and self.node_available:
            print("\n📜 Deploying contracts...")
            subprocess.run(['npm', 'run', 'deploy:local'])
            
        elif choice == '7' and self.node_available:
            print("\n🧪 Running blockchain tests...")
            subprocess.run(['npm', 'run', 'test:local'])
            
        elif choice == '0':
            return False
            
        return True
        
    def cleanup(self):
        """Clean up processes on exit"""
        print("\n🛑 Shutting down services...")
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        print("✅ All services stopped")
        
    def run(self):
        """Main run loop"""
        self.print_banner()
        
        # Check dependencies
        deps_ok, dashboard_running = self.check_dependencies()
        if not deps_ok:
            print("\n❌ Some dependencies are missing. Please check setup.")
            return
            
        # Start services
        if not dashboard_running:
            if not self.start_dashboard():
                print("\n❌ Failed to start dashboard")
                return
                
        # Show menu
        try:
            while True:
                self.show_menu()
                choice = input("\nSelect action (0-7): ").strip()
                
                if not self.handle_action(choice):
                    break
                    
                input("\n[Press Enter to continue]")
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        finally:
            self.cleanup()

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nReceived interrupt signal...")
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # Check if running in flox environment
    if 'FLOX_ENV' not in os.environ:
        print("⚠️  Not in flox environment. Starting with flox...")
        os.execvp('flox', ['flox', 'activate', '--', 'python', __file__] + sys.argv[1:])
    else:
        launcher = OminariLauncher()
        launcher.run()