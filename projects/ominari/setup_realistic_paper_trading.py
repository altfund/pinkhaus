#!/usr/bin/env python3
"""
Setup script for realistic paper trading with blockchain data

This script configures a paper trading environment that uses real blockchain
data for the most realistic simulation possible without using real money.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_prerequisites():
    """Check if all required components are available."""
    print("=== Checking Prerequisites ===\n")
    
    checks = {
        'API Key': os.getenv('OMINARI_API_KEY') is not None,
        'Database': os.path.exists('sport_odds.db'),
        'Paper Trading DB': os.path.exists('paper_trading.db'),
        'RPC Config': os.path.exists('rpc_config.py'),
        'Blockchain Reader': os.path.exists('blockchain_reader.py'),
    }
    
    all_good = True
    for component, status in checks.items():
        if status:
            print(f"✅ {component}")
        else:
            print(f"❌ {component}")
            all_good = False
    
    if not all_good:
        print("\n❌ Some prerequisites are missing!")
        if not checks['API Key']:
            print("\nTo set up API key:")
            print("  python setup_api_auth.py")
            print("  export OMINARI_API_KEY='your-key'")
        return False
    
    print("\n✅ All prerequisites met!")
    return True


def create_realistic_config():
    """Create configuration for realistic paper trading."""
    config = {
        "mode": "paper",
        "paper_trading": {
            "enabled": True,
            "session_name": f"realistic_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "initial_capital": 10000.0,
            "data_sources": {
                "primary": "blockchain",
                "fallback": "database",
                "update_interval": 60  # seconds
            }
        },
        "blockchain": {
            "networks": ["optimism", "arbitrum"],
            "contracts": {
                "optimism": {
                    "sports_amm": "0x170a5714112daEfF20E798B6e92e25B86Ea603C1",
                    "market_manager": "0x5ed98Ebb66A929758C7Fe5Ac60c979aDF0F4040a"
                },
                "arbitrum": {
                    "sports_amm": "0xd9aB397Fb7B3849A010f0e9a516Ab333feC52891",
                    "market_manager": "0x3E10355b57e7eFEbB6F405beFf6C2dDb078d8769"
                }
            },
            "rpc": {
                "use_free_endpoints": True,
                "max_retries": 3,
                "timeout": 30
            }
        },
        "signals": {
            "registry": [
                {
                    "name": "implied_raw",
                    "weight": 0.4,
                    "enabled": True
                },
                {
                    "name": "blockchain_oracle",
                    "weight": 0.3,
                    "enabled": True
                },
                {
                    "name": "external_grpc",
                    "weight": 0.3,
                    "enabled": False  # Enable if you have external signals
                }
            ],
            "min_edge": 0.02,  # 2% minimum edge
            "max_bet_fraction": 0.05  # Max 5% of bankroll per bet
        },
        "trading": {
            "kelly_fraction": 0.25,  # Conservative Kelly
            "max_positions": 20,
            "min_odds": 1.5,
            "max_odds": 10.0,
            "update_frequency": 300  # Check every 5 minutes
        },
        "monitoring": {
            "metrics_enabled": True,
            "log_level": "INFO",
            "dashboard_port": 8888
        }
    }
    
    # Save config
    with open('paper_trading_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created paper_trading_config.json")
    return config


def setup_paper_trading_session(config: Dict[str, Any]):
    """Set up a new paper trading session."""
    print("\n=== Setting Up Paper Trading Session ===\n")
    
    try:
        from paper_trading_sessions import PaperTradingSessionManager
        from paper_trading_db import init_db
        
        # Initialize database
        init_db()
        
        # Create session
        manager = PaperTradingSessionManager()
        session = manager.create_session(
            name=config['paper_trading']['session_name'],
            initial_capital=config['paper_trading']['initial_capital'],
            description="Realistic paper trading with blockchain data"
        )
        
        print(f"✅ Created session: {session['id']}")
        print(f"   Name: {session['name']}")
        print(f"   Capital: ${session['initial_capital']:,.2f}")
        
        return session['id']
        
    except Exception as e:
        logger.error(f"Failed to create session: {e}")
        return None


def create_paper_trading_script(session_id: str, config: Dict[str, Any]):
    """Create a script to run paper trading with blockchain data."""
    script = f'''#!/usr/bin/env python3
"""
Realistic Paper Trading with Blockchain Data
Session: {session_id}
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime, timezone
import time

from paper_trading_engine import PaperTradingEngine
from blockchain_reader import BlockchainReader
from signal_registry import SignalRegistry
from database_v2 import db_manager
from models import Market, Odd
from kelly_multimarket import calculate_kelly_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
SESSION_ID = "{session_id}"
CONFIG = {json.dumps(config, indent=2)}

class RealisticPaperTrader:
    """Paper trader using real blockchain data."""
    
    def __init__(self):
        self.running = True
        self.engine = PaperTradingEngine(SESSION_ID)
        self.signal_registry = SignalRegistry()
        self.blockchain_readers = {{}}
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info("Received shutdown signal, stopping...")
        self.running = False
    
    async def setup(self):
        """Set up blockchain readers and signals."""
        # Initialize blockchain readers
        for network in CONFIG['blockchain']['networks']:
            try:
                reader = BlockchainReader(
                    network=network,
                    contracts=CONFIG['blockchain']['contracts'][network]
                )
                self.blockchain_readers[network] = reader
                logger.info(f"✅ Connected to {{network}}")
            except Exception as e:
                logger.error(f"Failed to connect to {{network}}: {{e}}")
        
        # Set up signals
        for signal_config in CONFIG['signals']['registry']:
            if signal_config['enabled']:
                self.signal_registry.add_signal(
                    signal_config['name'],
                    weight=signal_config['weight']
                )
                logger.info(f"Added signal: {{signal_config['name']}} (weight={{signal_config['weight']}})")
    
    async def fetch_blockchain_markets(self):
        """Fetch current markets from blockchain."""
        all_markets = []
        
        for network, reader in self.blockchain_readers.items():
            try:
                markets = await reader.get_active_markets()
                logger.info(f"Found {{len(markets)}} active markets on {{network}}")
                all_markets.extend(markets)
            except Exception as e:
                logger.error(f"Error fetching markets from {{network}}: {{e}}")
        
        return all_markets
    
    async def evaluate_opportunities(self, markets):
        """Evaluate betting opportunities using signals."""
        opportunities = []
        
        with db_manager.get_db_session() as db:
            for market in markets:
                try:
                    # Get latest odds
                    odds = db.query(Odd).filter(
                        Odd.market_id == market['id']
                    ).order_by(Odd.updated_at.desc()).limit(3).all()
                    
                    if not odds:
                        continue
                    
                    # Calculate probabilities from signals
                    market_data = {{
                        'home_odds': [o.home_odds for o in odds],
                        'away_odds': [o.away_odds for o in odds],
                        'draw_odds': [o.draw_odds for o in odds] if odds[0].draw_odds else None
                    }}
                    
                    probabilities = self.signal_registry.get_probabilities(market_data)
                    
                    # Calculate edges
                    for outcome, prob in probabilities.items():
                        implied_prob = 1 / market_data[f'{{outcome}}_odds'][-1]
                        edge = prob - implied_prob
                        
                        if edge > CONFIG['signals']['min_edge']:
                            opportunities.append({{
                                'market_id': market['id'],
                                'market_name': f"{{market['home_team']}} vs {{market['away_team']}}",
                                'outcome': outcome,
                                'probability': prob,
                                'odds': market_data[f'{{outcome}}_odds'][-1],
                                'edge': edge,
                                'kelly': edge / (market_data[f'{{outcome}}_odds'][-1] - 1)
                            }})
                
                except Exception as e:
                    logger.error(f"Error evaluating market {{market['id']}}: {{e}}")
        
        return opportunities
    
    def execute_trades(self, opportunities):
        """Execute paper trades based on opportunities."""
        if not opportunities:
            return
        
        # Apply Kelly criterion with constraints
        current_capital = self.engine.get_current_capital()
        
        # Sort by edge
        opportunities.sort(key=lambda x: x['edge'], reverse=True)
        
        # Apply position limits
        max_positions = CONFIG['trading']['max_positions']
        current_positions = len(self.engine.get_open_positions())
        available_slots = max_positions - current_positions
        
        for opp in opportunities[:available_slots]:
            # Calculate bet size
            kelly_fraction = CONFIG['trading']['kelly_fraction']
            max_bet = current_capital * CONFIG['signals']['max_bet_fraction']
            
            bet_fraction = min(
                kelly_fraction * opp['kelly'],
                CONFIG['signals']['max_bet_fraction']
            )
            bet_amount = min(
                current_capital * bet_fraction,
                max_bet
            )
            
            if bet_amount < 10:  # Minimum bet size
                continue
            
            # Execute paper trade
            try:
                result = self.engine.place_bet(
                    market_id=opp['market_id'],
                    market_name=opp['market_name'],
                    outcome=opp['outcome'],
                    odds=opp['odds'],
                    stake=bet_amount,
                    probability=opp['probability'],
                    edge=opp['edge']
                )
                
                logger.info(f"✅ Placed bet: {{opp['market_name']}} - {{opp['outcome']}} "
                          f"@ {{opp['odds']:.2f}} for ${{bet_amount:.2f}}")
                
            except Exception as e:
                logger.error(f"Failed to place bet: {{e}}")
    
    async def run(self):
        """Main trading loop."""
        await self.setup()
        
        logger.info("Starting realistic paper trading...")
        logger.info(f"Session: {{SESSION_ID}}")
        logger.info(f"Initial capital: ${{CONFIG['paper_trading']['initial_capital']:,.2f}}")
        
        last_update = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to update
                if current_time - last_update > CONFIG['trading']['update_frequency']:
                    logger.info("Checking for opportunities...")
                    
                    # Fetch markets from blockchain
                    markets = await self.fetch_blockchain_markets()
                    
                    # Evaluate opportunities
                    opportunities = await self.evaluate_opportunities(markets)
                    
                    if opportunities:
                        logger.info(f"Found {{len(opportunities)}} opportunities")
                        
                        # Execute trades
                        self.execute_trades(opportunities)
                    else:
                        logger.info("No opportunities found")
                    
                    # Update metrics
                    stats = self.engine.get_session_stats()
                    logger.info(f"Portfolio: ${{stats['current_value']:,.2f}} "
                              f"({{stats['total_return_pct']:+.2f}}%)")
                    
                    last_update = current_time
                
                # Sleep for a bit
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Error in main loop: {{e}}")
                await asyncio.sleep(60)
        
        logger.info("Paper trading stopped")
        
        # Final stats
        final_stats = self.engine.get_session_stats()
        logger.info(f"\\nFinal Results:")
        logger.info(f"Total return: ${{final_stats['total_return']:,.2f}} "
                  f"({{final_stats['total_return_pct']:+.2f}}%)")
        logger.info(f"Win rate: {{final_stats['win_rate']:.2%}}")
        logger.info(f"Total bets: {{final_stats['total_bets']}}")


async def main():
    """Run the paper trader."""
    trader = RealisticPaperTrader()
    await trader.run()


if __name__ == "__main__":
    asyncio.run(main())
'''
    
    filename = 'run_realistic_paper_trading.py'
    with open(filename, 'w') as f:
        f.write(script)
    
    os.chmod(filename, 0o755)
    print(f"✅ Created {filename}")
    return filename


def create_monitoring_script():
    """Create a script to monitor paper trading performance."""
    script = '''#!/usr/bin/env python3
"""
Monitor realistic paper trading performance
"""

import time
import sys
from datetime import datetime
from paper_trading_sessions import PaperTradingSessionManager
from paper_trading_db import get_db_session
from paper_trading_models_v2 import PaperTradingPosition

def monitor_session(session_id: str):
    """Monitor a paper trading session."""
    manager = PaperTradingSessionManager()
    
    print(f"\\n=== Monitoring Paper Trading Session ===")
    print(f"Session ID: {session_id}")
    print(f"{'='*50}\\n")
    
    try:
        while True:
            # Get session stats
            session = manager.get_session(session_id)
            if not session:
                print("Session not found!")
                break
            
            # Get positions
            with get_db_session() as db:
                open_positions = db.query(PaperTradingPosition).filter(
                    PaperTradingPosition.session_id == session_id,
                    PaperTradingPosition.status == 'open'
                ).all()
                
                total_positions = db.query(PaperTradingPosition).filter(
                    PaperTradingPosition.session_id == session_id
                ).count()
            
            # Calculate metrics
            capital = session['current_capital']
            pnl = capital - session['initial_capital']
            pnl_pct = (pnl / session['initial_capital']) * 100
            
            # Clear screen and display
            print("\\033[2J\\033[H")  # Clear screen
            print(f"=== Paper Trading Monitor ===")
            print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"\\n--- Portfolio ---")
            print(f"Current Value: ${capital:,.2f}")
            print(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print(f"\\n--- Positions ---")
            print(f"Open: {len(open_positions)}")
            print(f"Total: {total_positions}")
            
            if open_positions:
                print(f"\\n--- Open Positions ---")
                for pos in open_positions[:5]:  # Show first 5
                    print(f"  {pos.market_name[:30]:<30} {pos.outcome:<10} "
                          f"${pos.stake:.2f} @ {pos.odds:.2f}")
                if len(open_positions) > 5:
                    print(f"  ... and {len(open_positions) - 5} more")
            
            print(f"\\n--- Performance ---")
            if total_positions > 0:
                wins = sum(1 for p in open_positions if p.pnl and p.pnl > 0)
                win_rate = (wins / total_positions) * 100 if total_positions > 0 else 0
                print(f"Win Rate: {win_rate:.1f}%")
            
            print("\\nPress Ctrl+C to exit")
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\\n\\nMonitoring stopped")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # Get latest session
        manager = PaperTradingSessionManager()
        sessions = manager.list_sessions()
        if sessions:
            session_id = sessions[-1]['id']
            print(f"Using latest session: {session_id}")
        else:
            print("No sessions found!")
            sys.exit(1)
    
    monitor_session(session_id)
'''
    
    with open('monitor_paper_trading.py', 'w') as f:
        f.write(script)
    
    os.chmod('monitor_paper_trading.py', 0o755)
    print("✅ Created monitor_paper_trading.py")


def create_analysis_script():
    """Create a script to analyze paper trading results."""
    script = '''#!/usr/bin/env python3
"""
Analyze paper trading results
"""

import pandas as pd
from datetime import datetime
from paper_trading_sessions import PaperTradingSessionManager
from paper_trading_db import get_db_session
from paper_trading_models_v2 import PaperTradingPosition

def analyze_session(session_id: str):
    """Analyze a paper trading session."""
    manager = PaperTradingSessionManager()
    session = manager.get_session(session_id)
    
    if not session:
        print(f"Session {session_id} not found!")
        return
    
    print(f"\\n=== Paper Trading Analysis ===")
    print(f"Session: {session['name']}")
    print(f"Started: {session['created_at']}")
    print(f"\\n{'='*50}\\n")
    
    # Get all positions
    with get_db_session() as db:
        positions = db.query(PaperTradingPosition).filter(
            PaperTradingPosition.session_id == session_id
        ).all()
    
    if not positions:
        print("No positions found!")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame([{
        'market': p.market_name,
        'outcome': p.outcome,
        'odds': p.odds,
        'stake': p.stake,
        'status': p.status,
        'pnl': p.pnl or 0,
        'created': p.created_at
    } for p in positions])
    
    # Overall metrics
    print("--- Overall Performance ---")
    print(f"Total Positions: {len(df)}")
    print(f"Total Staked: ${df['stake'].sum():,.2f}")
    
    closed = df[df['status'] != 'open']
    if len(closed) > 0:
        print(f"\\n--- Closed Positions ---")
        print(f"Total: {len(closed)}")
        print(f"Wins: {len(closed[closed['pnl'] > 0])}")
        print(f"Losses: {len(closed[closed['pnl'] < 0])}")
        print(f"Win Rate: {len(closed[closed['pnl'] > 0]) / len(closed) * 100:.1f}%")
        print(f"Total P&L: ${closed['pnl'].sum():,.2f}")
        print(f"Average P&L: ${closed['pnl'].mean():,.2f}")
    
    # By outcome
    print(f"\\n--- By Outcome ---")
    outcome_stats = df.groupby('outcome').agg({
        'stake': ['count', 'sum'],
        'pnl': 'sum'
    }).round(2)
    print(outcome_stats)
    
    # By odds range
    print(f"\\n--- By Odds Range ---")
    df['odds_range'] = pd.cut(df['odds'], bins=[0, 1.5, 2, 3, 5, 100], 
                              labels=['1.0-1.5', '1.5-2.0', '2.0-3.0', '3.0-5.0', '5.0+'])
    odds_stats = df.groupby('odds_range').agg({
        'stake': ['count', 'sum'],
        'pnl': 'sum'
    }).round(2)
    print(odds_stats)
    
    # Time analysis
    if len(df) > 1:
        print(f"\\n--- Time Analysis ---")
        df['hour'] = pd.to_datetime(df['created']).dt.hour
        hourly = df.groupby('hour')['stake'].count()
        print(f"Most active hours: {hourly.nlargest(3).index.tolist()}")
    
    # Current portfolio
    open_positions = df[df['status'] == 'open']
    if len(open_positions) > 0:
        print(f"\\n--- Current Portfolio ---")
        print(f"Open Positions: {len(open_positions)}")
        print(f"Total Exposure: ${open_positions['stake'].sum():,.2f}")
        print(f"\\nTop 5 Open Positions:")
        for _, pos in open_positions.nlargest(5, 'stake').iterrows():
            print(f"  {pos['market'][:40]:<40} ${pos['stake']:.2f} @ {pos['odds']:.2f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        session_id = sys.argv[1]
    else:
        # Get latest session
        manager = PaperTradingSessionManager()
        sessions = manager.list_sessions()
        if sessions:
            session_id = sessions[-1]['id']
            print(f"Using latest session: {session_id}")
        else:
            print("No sessions found!")
            sys.exit(1)
    
    analyze_session(session_id)
'''
    
    with open('analyze_paper_trading.py', 'w') as f:
        f.write(script)
    
    os.chmod('analyze_paper_trading.py', 0o755)
    print("✅ Created analyze_paper_trading.py")


def main():
    """Set up realistic paper trading environment."""
    print("=== Realistic Paper Trading Setup ===\n")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\nPlease fix the prerequisites and try again.")
        sys.exit(1)
    
    # Create configuration
    print("\n=== Creating Configuration ===\n")
    config = create_realistic_config()
    
    # Set up paper trading session
    session_id = setup_paper_trading_session(config)
    if not session_id:
        print("\n❌ Failed to create paper trading session")
        sys.exit(1)
    
    # Create scripts
    print("\n=== Creating Scripts ===\n")
    trading_script = create_paper_trading_script(session_id, config)
    create_monitoring_script()
    create_analysis_script()
    
    # Display instructions
    print("\n=== Setup Complete! ===\n")
    print("Your realistic paper trading environment is ready.\n")
    
    print("📊 Configuration saved to: paper_trading_config.json")
    print(f"🎮 Session ID: {session_id}\n")
    
    print("To start paper trading with blockchain data:")
    print(f"  python {trading_script}\n")
    
    print("To monitor performance in real-time:")
    print(f"  python monitor_paper_trading.py {session_id}\n")
    
    print("To analyze results:")
    print(f"  python analyze_paper_trading.py {session_id}\n")
    
    print("To view in the web dashboard:")
    print("  1. Start the web monitor: python web_monitor_with_metrics.py")
    print("  2. Open: http://localhost:8888/dashboard")
    print(f"  3. Select session: {session_id}\n")
    
    print("💡 Tips:")
    print("  - The trader will use real blockchain odds data")
    print("  - It will check for opportunities every 5 minutes")
    print("  - All trades are simulated (no real money)")
    print("  - You can run multiple sessions in parallel")
    print("  - Stop with Ctrl+C anytime\n")
    
    print("🚀 Ready to start realistic paper trading!")


if __name__ == "__main__":
    main()