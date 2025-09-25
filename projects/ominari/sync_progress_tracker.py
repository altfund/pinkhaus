#!/usr/bin/env python3
"""
Track sync progress: API vs Blockchain data
Shows how many games exist vs how many we've synced
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import requests
from datetime import datetime, timezone
from database_v2 import db_manager
from models import Market
from web3 import Web3
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyncProgressTracker:
    def __init__(self):
        self.api_stats = {
            'total_games': 0,
            'real_games': 0,
            'futures': 0,
            'last_check': None
        }
        
        self.db_stats = {
            'total_markets': 0,
            'active_markets': 0,
            'by_source': {},
            'last_update': None
        }
        
        self.blockchain_stats = {
            'arbitrum': {'checked': False, 'markets': 0},
            'optimism': {'checked': False, 'markets': 0}
        }
        
    def check_api_totals(self):
        """Check total games available in API."""
        logger.info("📡 Checking API totals...")
        
        try:
            response = requests.get("https://api.overtime.io/overtime-v2/games-info", timeout=30)
            if response.status_code == 200:
                games = response.json()
                
                self.api_stats['total_games'] = len(games)
                self.api_stats['real_games'] = 0
                self.api_stats['futures'] = 0
                
                # Analyze game types
                for game_id, info in games.items():
                    teams = info.get('teams', [])
                    if len(teams) == 2:
                        home = teams[0].get('name', '')
                        away = teams[1].get('name', '') if len(teams) > 1 else ''
                        
                        # Check if it's a futures market
                        if any(term in f"{home} {away}" for term in 
                              ['Winner', 'Championship', 'MVP', 'To Win', 'To Make']):
                            self.api_stats['futures'] += 1
                        else:
                            self.api_stats['real_games'] += 1
                            
                self.api_stats['last_check'] = datetime.now(timezone.utc)
                
                logger.info(f"✅ API Total: {self.api_stats['total_games']} games")
                logger.info(f"   Real games: {self.api_stats['real_games']}")
                logger.info(f"   Futures: {self.api_stats['futures']}")
                
        except Exception as e:
            logger.error(f"Error checking API: {e}")
            
    def check_database_stats(self):
        """Check what we have in database."""
        logger.info("\n💾 Checking database...")
        
        with db_manager.get_db_session() as db:
            self.db_stats['total_markets'] = db.query(Market).count()
            self.db_stats['active_markets'] = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).count()
            
            # Count by source
            sources = db.query(Market.source).distinct().all()
            for source, in sources:
                count = db.query(Market).filter(Market.source == source).count()
                self.db_stats['by_source'][source] = count
                
            self.db_stats['last_update'] = datetime.now(timezone.utc)
            
            logger.info(f"✅ DB Total: {self.db_stats['total_markets']} markets")
            logger.info(f"   Active: {self.db_stats['active_markets']}")
            logger.info("   By source:")
            for source, count in self.db_stats['by_source'].items():
                logger.info(f"     • {source}: {count}")
                
    def check_blockchain_activity(self):
        """Check blockchain for market activity."""
        logger.info("\n🔗 Checking blockchain activity...")
        
        chains = {
            'arbitrum': {
                'rpc': 'https://arb1.arbitrum.io/rpc',
                'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395'
            },
            'optimism': {
                'rpc': 'https://mainnet.optimism.io',
                'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431'
            }
        }
        
        for chain_name, config in chains.items():
            try:
                w3 = Web3(Web3.HTTPProvider(config['rpc']))
                if w3.is_connected():
                    # Check recent transactions
                    current_block = w3.eth.block_number
                    tx_count = w3.eth.get_transaction_count(config['amm'])
                    
                    self.blockchain_stats[chain_name]['checked'] = True
                    self.blockchain_stats[chain_name]['tx_count'] = tx_count
                    self.blockchain_stats[chain_name]['latest_block'] = current_block
                    
                    logger.info(f"✅ {chain_name}: Connected at block {current_block:,}")
                    logger.info(f"   AMM tx count: {tx_count}")
                    
            except Exception as e:
                logger.error(f"Error checking {chain_name}: {e}")
                
    def calculate_sync_progress(self):
        """Calculate sync progress percentage."""
        if self.api_stats['real_games'] > 0:
            progress = (self.db_stats['total_markets'] / self.api_stats['real_games']) * 100
            return min(progress, 100)  # Cap at 100%
        return 0
        
    def print_summary(self):
        """Print comprehensive summary."""
        logger.info("\n" + "=" * 60)
        logger.info("📊 SYNC PROGRESS SUMMARY")
        logger.info("=" * 60)
        
        # API Stats
        logger.info(f"\n🌐 API STATISTICS:")
        logger.info(f"Total games in API: {self.api_stats['total_games']:,}")
        logger.info(f"Real games (not futures): {self.api_stats['real_games']:,}")
        logger.info(f"Futures/special markets: {self.api_stats['futures']:,}")
        
        # Database Stats
        logger.info(f"\n💾 DATABASE STATISTICS:")
        logger.info(f"Total synced markets: {self.db_stats['total_markets']}")
        logger.info(f"Active markets: {self.db_stats['active_markets']}")
        
        # Progress
        progress = self.calculate_sync_progress()
        logger.info(f"\n📈 SYNC PROGRESS:")
        logger.info(f"Synced {self.db_stats['total_markets']} of {self.api_stats['real_games']} real games")
        logger.info(f"Progress: {progress:.1f}%")
        
        # Progress bar
        bar_length = 40
        filled = int(bar_length * progress / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        logger.info(f"[{bar}] {progress:.1f}%")
        
        # Blockchain Stats
        logger.info(f"\n🔗 BLOCKCHAIN STATUS:")
        for chain, stats in self.blockchain_stats.items():
            if stats['checked']:
                logger.info(f"{chain}: ✅ Connected")
            else:
                logger.info(f"{chain}: ❌ Not checked")
                
        logger.info("\n" + "=" * 60)
        
    def continuous_monitor(self):
        """Run continuous monitoring."""
        logger.info("🚀 Starting Sync Progress Monitor")
        logger.info("Will check every 30 seconds...")
        
        while True:
            try:
                self.check_api_totals()
                self.check_database_stats()
                self.check_blockchain_activity()
                self.print_summary()
                
                logger.info("\n⏰ Next check in 30 seconds... (Ctrl+C to stop)")
                time.sleep(30)
                
            except KeyboardInterrupt:
                logger.info("\n👋 Stopping monitor...")
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                time.sleep(30)

def main():
    tracker = SyncProgressTracker()
    tracker.continuous_monitor()

if __name__ == "__main__":
    main()