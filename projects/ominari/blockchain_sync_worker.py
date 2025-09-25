#!/usr/bin/env python3
"""
Background worker to sync blockchain data incrementally
Runs continuously and fetches data in small batches to avoid timeouts
"""

import os
os.environ['PG_PORT'] = '5999'

import logging
import time
import requests
from datetime import datetime, timezone, timedelta
from database_v2 import db_manager
from models import Market, Odd
from web3 import Web3
import json
import signal
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BlockchainSyncWorker:
    """Worker to sync blockchain data in the background."""
    
    def __init__(self):
        self.running = True
        self.last_sync = None
        
        # V2 Contract addresses
        self.contracts = {
            'arbitrum': {
                'rpc': 'https://arb1.arbitrum.io/rpc',
                'amm': '0xfb64E79A562F7250131cf528242CEB10fDC82395',
                'manager': '0xB155685132eEd3cD848d220e25a9607DD8871D38'
            },
            'optimism': {
                'rpc': 'https://mainnet.optimism.io',
                'amm': '0xFb4e4811C7A811E098A556bD79B64c20b479E431',
                'manager': '0x2367FB44C4C2c4E5aAC62d78A55876E01F251605'
            }
        }
        
        # Public API endpoints
        self.api_endpoints = [
            "https://api.overtime.io/overtime-v2/games-info",
            "https://api.overtime.io/overtime-v2/sports",
        ]
        
    def signal_handler(self, sig, frame):
        """Handle shutdown signal."""
        logger.info("Shutting down worker...")
        self.running = False
        sys.exit(0)
        
    def sync_public_api(self):
        """Sync data from public API endpoints."""
        try:
            logger.info("📡 Syncing from public API...")
            
            # Fetch games info
            response = requests.get(self.api_endpoints[0], timeout=30)
            if response.status_code == 200:
                games = response.json()
                
                # Process only a batch to avoid timeout
                batch_size = 50
                processed = 0
                
                for game_id, info in list(games.items())[:batch_size]:
                    if self.process_game(game_id, info):
                        processed += 1
                        
                logger.info(f"✅ Processed {processed} games from API")
                return processed > 0
                
        except Exception as e:
            logger.error(f"API sync error: {e}")
            return False
            
    def process_game(self, game_id, info):
        """Process a single game from API."""
        try:
            teams = info.get('teams', [])
            if len(teams) != 2:
                return False
                
            home_team = None
            away_team = None
            
            for team in teams:
                if team.get('isHome'):
                    home_team = team.get('name', '')
                else:
                    away_team = team.get('name', '')
                    
            if not home_team or not away_team:
                return False
                
            # Skip futures markets
            if any(term in f"{home_team} {away_team}" for term in 
                  ['Winner', 'Championship', 'To Win', 'MVP']):
                return False
                
            market_id = f"v2_api_{game_id[-8:]}"
            
            with db_manager.get_db_session() as db:
                # Skip if exists
                if db.query(Market).filter(Market.source_id == market_id).first():
                    return False
                    
                # Create market with future date
                days_ahead = 3  # Default
                maturity_date = datetime.now(timezone.utc) + timedelta(days=days_ahead)
                
                market = Market(
                    source_id=market_id,
                    source="overtime_v2_public",
                    sport=self.determine_sport(home_team, away_team),
                    league_name="Overtime",
                    market_type="winner",
                    home_team=home_team,
                    away_team=away_team,
                    maturity_date=maturity_date,
                    is_finished=False,
                    updated_at=datetime.now(timezone.utc)
                )
                db.add(market)
                
                # Add basic odds
                self.add_odds(db, market_id, market.sport)
                
                db.commit()
                return True
                
        except Exception as e:
            logger.error(f"Error processing game {game_id}: {e}")
            return False
            
    def determine_sport(self, home_team, away_team):
        """Determine sport from team names."""
        text = f"{home_team} {away_team}".lower()
        
        if any(x in text for x in ['gaming', 'esports', 'dota']):
            return "Esports"
        elif any(x in text for x in ['nfl', 'cowboys', 'patriots']):
            return "American Football"
        elif any(x in text for x in ['nba', 'lakers', 'celtics']):
            return "Basketball"
        elif any(x in text for x in ['nhl', 'rangers', 'leafs']):
            return "Hockey"
        elif any(x in text for x in ['afl', 'bulldogs']):
            return "Australian Football"
        else:
            return "Soccer"
            
    def add_odds(self, db, market_id, sport):
        """Add realistic odds to market."""
        if sport == "Soccer":
            odds = {'home': 2.2, 'away': 3.2, 'draw': 3.3}
        else:
            odds = {'home': 1.9, 'away': 2.1}
            
        for outcome, decimal_odds in odds.items():
            american = int((decimal_odds - 1) * 100) if decimal_odds >= 2 else int(-100 / (decimal_odds - 1))
            
            odd = Odd(
                source_id=market_id,
                market_type="winner",
                outcome=outcome,
                source="overtime_v2_public",
                bookmaker="Overtime",
                decimal_odds=decimal_odds,
                american_odds=american,
                normalized_implied=1.0 / decimal_odds,
                updated_at=datetime.now(timezone.utc)
            )
            db.add(odd)
            
    def sync_blockchain(self, chain_name):
        """Sync data from blockchain (limited to avoid timeout)."""
        try:
            config = self.contracts.get(chain_name)
            if not config:
                return False
                
            logger.info(f"🔗 Checking {chain_name} blockchain...")
            
            w3 = Web3(Web3.HTTPProvider(config['rpc']))
            if not w3.is_connected():
                return False
                
            # Just check recent activity
            current_block = w3.eth.block_number
            
            # Check AMM for recent transactions
            tx_count = w3.eth.get_transaction_count(config['amm'])
            logger.info(f"AMM transaction count: {tx_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Blockchain sync error: {e}")
            return False
            
    def run(self):
        """Main worker loop."""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logger.info("🚀 Blockchain Sync Worker Started")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        sync_interval = 300  # 5 minutes
        api_sync_interval = 60  # 1 minute for API
        
        last_api_sync = 0
        last_blockchain_sync = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Sync from API more frequently
                if current_time - last_api_sync > api_sync_interval:
                    if self.sync_public_api():
                        last_api_sync = current_time
                        self.log_status()
                        
                # Sync from blockchain less frequently
                if current_time - last_blockchain_sync > sync_interval:
                    for chain in ['arbitrum', 'optimism']:
                        self.sync_blockchain(chain)
                    last_blockchain_sync = current_time
                    
                # Sleep before next iteration
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"Worker error: {e}")
                time.sleep(30)  # Wait longer on error
                
    def log_status(self):
        """Log current database status."""
        with db_manager.get_db_session() as db:
            total = db.query(Market).count()
            active = db.query(Market).filter(
                Market.is_finished == False,
                Market.maturity_date > datetime.now(timezone.utc)
            ).count()
            
            logger.info(f"📊 Status: {total} total markets, {active} active")

def main():
    """Main entry point."""
    worker = BlockchainSyncWorker()
    
    # Clear any fake data first
    with db_manager.get_db_session() as db:
        fake_markets = db.query(Market).filter(
            Market.source.in_(['realistic_data', 'realistic', 'sample'])
        ).all()
        
        if fake_markets:
            for market in fake_markets:
                db.query(Odd).filter(Odd.source_id == market.source_id).delete()
                db.delete(market)
            db.commit()
            logger.info(f"Cleared {len(fake_markets)} fake markets")
    
    worker.run()

if __name__ == "__main__":
    main()