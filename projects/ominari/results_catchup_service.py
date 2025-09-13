#!/usr/bin/env python3
"""
Automated Results Catch-up Service

This service:
1. Monitors for recently finished matches
2. Updates results in the database
3. Updates paper trading positions
4. Can run continuously or as a cron job
"""

import time
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from database_v2 import db_manager
from models import Market, Odd
from paper_trading_sessions import PaperTradingSessionManager
from sqlalchemy import and_, or_, func
import argparse
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResultsCatchupService:
    """Service to catch up on match results and update positions."""
    
    def __init__(self, lookback_hours: int = 24):
        self.lookback_hours = lookback_hours
        self.session_manager = PaperTradingSessionManager()
        self.stats = {
            'matches_processed': 0,
            'positions_updated': 0,
            'errors': 0,
            'last_run': None
        }
        self.state_file = "catchup_state.json"
        self.load_state()
    
    def load_state(self):
        """Load the last run state."""
        try:
            with open(self.state_file, 'r') as f:
                saved_state = json.load(f)
                self.stats.update(saved_state)
        except FileNotFoundError:
            logger.info("No previous state found, starting fresh")
    
    def save_state(self):
        """Save the current run state."""
        self.stats['last_run'] = datetime.now(timezone.utc).isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def find_recently_finished_matches(self) -> List[Market]:
        """Find matches that finished in the lookback period."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.lookback_hours)
        
        # If we have a last run time, use that as cutoff instead
        if self.stats.get('last_run'):
            try:
                last_run = datetime.fromisoformat(self.stats['last_run'])
                if last_run > cutoff_time:
                    cutoff_time = last_run
                    logger.info(f"Using last run time as cutoff: {cutoff_time}")
            except:
                pass
        
        with db_manager.get_db_session() as db:
            # Find matches that:
            # 1. Are marked as finished
            # 2. Were updated after our cutoff
            # 3. Have valid scores
            finished_matches = db.query(Market).filter(
                and_(
                    Market.is_finished == True,
                    Market.last_update >= cutoff_time,
                    Market.home_score.isnot(None),
                    Market.away_score.isnot(None)
                )
            ).order_by(Market.last_update.desc()).all()
            
            logger.info(f"Found {len(finished_matches)} recently finished matches")
            return finished_matches
    
    def update_position_for_finished_match(self, market: Market) -> int:
        """Update paper trading positions for a finished match."""
        updates_made = 0
        
        # Get current session
        current_session = self.session_manager.get_current_session()
        if not current_session:
            return 0
        
        market_id = market.source_id
        resolved_outcome = market.resolved_outcome
        
        # Check open positions
        positions_to_close = []
        for pos_key, position in current_session.get('positions', {}).items():
            if position.get('market_id') == market_id:
                positions_to_close.append((pos_key, position))
        
        # Close positions for finished markets
        for pos_key, position in positions_to_close:
            outcome = position.get('outcome')
            
            # Determine if won or lost
            if resolved_outcome and outcome == resolved_outcome:
                result = 'won'
            else:
                result = 'lost'
            
            logger.info(f"Closing position {pos_key} as {result} "
                       f"(market: {market.home_team} vs {market.away_team}, "
                       f"result: {resolved_outcome})")
            
            # Close the position
            try:
                self.session_manager.close_position(
                    current_session['session_id'],
                    pos_key,
                    market_result=result
                )
                updates_made += 1
            except Exception as e:
                logger.error(f"Error closing position {pos_key}: {e}")
                self.stats['errors'] += 1
        
        # Also check if any closed positions need result updates
        for position in current_session.get('closed_positions', []):
            if position.get('market_id') == market_id and position.get('result') == 'pending':
                outcome = position.get('outcome')
                if resolved_outcome and outcome == resolved_outcome:
                    position['result'] = 'won'
                else:
                    position['result'] = 'lost'
                
                # Add market result info
                position['market_result'] = {
                    'resolved_outcome': resolved_outcome,
                    'home_score': market.home_score,
                    'away_score': market.away_score,
                    'finish_time': market.last_update.isoformat()
                }
                
                updates_made += 1
                logger.info(f"Updated closed position result for {market_id}")
        
        if updates_made > 0:
            self.session_manager._save_sessions()
        
        return updates_made
    
    def process_finished_matches(self, matches: List[Market]):
        """Process a list of finished matches."""
        for market in matches:
            try:
                logger.info(f"Processing: {market.home_team} vs {market.away_team} "
                           f"(Result: {market.home_score}-{market.away_score})")
                
                updates = self.update_position_for_finished_match(market)
                self.stats['matches_processed'] += 1
                self.stats['positions_updated'] += updates
                
            except Exception as e:
                logger.error(f"Error processing market {market.source_id}: {e}")
                self.stats['errors'] += 1
    
    def run_catchup(self):
        """Run a single catch-up cycle."""
        logger.info(f"Starting catch-up (lookback: {self.lookback_hours} hours)")
        
        # Find recently finished matches
        finished_matches = self.find_recently_finished_matches()
        
        if finished_matches:
            self.process_finished_matches(finished_matches)
        else:
            logger.info("No new finished matches found")
        
        # Save state
        self.save_state()
        
        # Print summary
        logger.info(f"Catch-up complete: "
                   f"Processed {self.stats['matches_processed']} matches, "
                   f"Updated {self.stats['positions_updated']} positions, "
                   f"Errors: {self.stats['errors']}")
    
    def run_continuous(self, interval_minutes: int = 30):
        """Run continuously with specified interval."""
        logger.info(f"Starting continuous catch-up service (interval: {interval_minutes} minutes)")
        
        while True:
            try:
                self.run_catchup()
                logger.info(f"Sleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
                
            except KeyboardInterrupt:
                logger.info("Stopping catch-up service...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in continuous mode: {e}")
                time.sleep(60)  # Wait a minute before retrying

def setup_cron_job():
    """Print instructions for setting up as a cron job."""
    print("\nTo set up automatic catch-up as a cron job:")
    print("\n1. Open crontab:")
    print("   crontab -e")
    print("\n2. Add one of these entries:")
    print("   # Run every 30 minutes:")
    print("   */30 * * * * cd /path/to/ominari && python results_catchup_service.py --mode once >> logs/catchup.log 2>&1")
    print("\n   # Run every hour:")
    print("   0 * * * * cd /path/to/ominari && python results_catchup_service.py --mode once >> logs/catchup.log 2>&1")
    print("\n   # Run every 4 hours:")
    print("   0 */4 * * * cd /path/to/ominari && python results_catchup_service.py --mode once >> logs/catchup.log 2>&1")

def main():
    parser = argparse.ArgumentParser(description='Results catch-up service')
    parser.add_argument('--mode', choices=['once', 'continuous', 'setup'], 
                       default='once', help='Run mode')
    parser.add_argument('--lookback', type=int, default=24, 
                       help='Hours to look back for finished matches')
    parser.add_argument('--interval', type=int, default=30, 
                       help='Minutes between runs in continuous mode')
    
    args = parser.parse_args()
    
    if args.mode == 'setup':
        setup_cron_job()
        return
    
    # Create service
    service = ResultsCatchupService(lookback_hours=args.lookback)
    
    if args.mode == 'once':
        service.run_catchup()
    elif args.mode == 'continuous':
        service.run_continuous(interval_minutes=args.interval)

if __name__ == "__main__":
    main()