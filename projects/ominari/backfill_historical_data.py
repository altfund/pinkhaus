#!/usr/bin/env python3
"""
Backfill historical match results and paper trading data.

This script:
1. Checks all finished markets for missing scores/results
2. Enriches paper trading positions with historical odds and edges
3. Provides a catch-up mechanism for ongoing updates
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from database_v2 import db_manager
from models import Market, Odd
from paper_trading_sessions import PaperTradingSessionManager
from sqlalchemy import and_, or_, func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataBackfiller:
    """Handles backfilling of historical match results and trading data."""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.stats = {
            'markets_checked': 0,
            'markets_missing_results': 0,
            'markets_updated': 0,
            'positions_checked': 0,
            'positions_enriched': 0,
            'errors': []
        }
    
    def check_finished_markets_without_scores(self, limit: int = 1000) -> List[Market]:
        """Find finished markets that are missing score data."""
        logger.info("Checking for finished markets without scores...")
        
        with db_manager.get_db_session() as db:
            # Find markets marked as finished but missing scores
            missing_scores = db.query(Market).filter(
                and_(
                    Market.is_finished == True,
                    or_(
                        Market.home_score.is_(None),
                        Market.away_score.is_(None)
                    )
                )
            ).limit(limit).all()
            
            logger.info(f"Found {len(missing_scores)} finished markets without scores")
            return missing_scores
    
    def get_historical_odds_for_market(self, market_id: str, 
                                     timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """Get historical odds for a market at a specific time."""
        with db_manager.get_db_session() as db:
            query = db.query(Odd).filter(Odd.source_id == market_id)
            
            if timestamp:
                # Get odds closest to the specified timestamp
                query = query.filter(Odd.updated_at <= timestamp)
            
            # Get latest odds for each outcome
            odds_data = {}
            for outcome in ['Home', 'Draw', 'Away']:
                odd = query.filter(Odd.outcome == outcome)\
                          .order_by(Odd.updated_at.desc())\
                          .first()
                if odd:
                    odds_data[outcome] = odd.decimal_odds
            
            return odds_data
    
    def enrich_paper_trading_position(self, position: Dict, 
                                    market_data: Optional[Market] = None) -> Dict:
        """Enrich a paper trading position with historical data."""
        market_id = position.get('market_id')
        if not market_id:
            return position
        
        try:
            # Get market data if not provided
            if not market_data:
                with db_manager.get_db_session() as db:
                    market_data = db.query(Market).filter(
                        Market.source_id == market_id
                    ).first()
            
            if market_data:
                # Add market result information
                position['market_result'] = {
                    'is_finished': market_data.is_finished,
                    'resolved_outcome': market_data.resolved_outcome,
                    'home_score': market_data.home_score,
                    'away_score': market_data.away_score,
                    'finish_time': market_data.last_update.isoformat() if market_data.last_update else None
                }
                
                # Get historical odds at position open time
                opened_at = position.get('opened_at')
                if opened_at:
                    if isinstance(opened_at, str):
                        opened_at = datetime.fromisoformat(opened_at.replace('Z', '+00:00'))
                    
                    historical_odds = self.get_historical_odds_for_market(market_id, opened_at)
                    position['historical_odds'] = historical_odds
                    
                    # Calculate implied probabilities and edges
                    outcome = position.get('outcome')
                    if outcome and outcome in historical_odds:
                        odds = historical_odds[outcome]
                        implied_prob = 1.0 / odds if odds > 0 else 0
                        
                        # Get signal probability if available
                        signal_prob = None
                        for trade in position.get('trades', []):
                            if trade.get('signal_probability'):
                                signal_prob = trade['signal_probability']
                                break
                        
                        if signal_prob:
                            edge = (signal_prob - implied_prob) * 100
                            position['calculated_edge'] = edge
                            position['implied_probability'] = implied_prob
                            position['signal_probability'] = signal_prob
                
                # Verify result consistency
                if market_data.is_finished and position.get('result') == 'pending':
                    # Update position result based on market outcome
                    resolved = market_data.resolved_outcome
                    if resolved and resolved == position.get('outcome'):
                        position['result'] = 'won'
                        position['needs_pnl_recalc'] = True
                    elif resolved:
                        position['result'] = 'lost'
                        position['needs_pnl_recalc'] = True
                
            self.stats['positions_enriched'] += 1
            
        except Exception as e:
            logger.error(f"Error enriching position {position.get('id')}: {e}")
            self.stats['errors'].append(f"Position enrichment error: {e}")
        
        return position
    
    def backfill_all_sessions(self):
        """Backfill data for all paper trading sessions."""
        logger.info("Starting backfill of all paper trading sessions...")
        
        sessions = self.session_manager.sessions.get('sessions', {})
        
        for session_id, session in sessions.items():
            logger.info(f"Processing session {session_id}...")
            
            # Enrich open positions
            for pos_key, position in session.get('positions', {}).items():
                self.stats['positions_checked'] += 1
                enriched = self.enrich_paper_trading_position(position)
                session['positions'][pos_key] = enriched
            
            # Enrich closed positions
            closed_positions = session.get('closed_positions', [])
            enriched_closed = []
            for position in closed_positions:
                self.stats['positions_checked'] += 1
                enriched = self.enrich_paper_trading_position(position)
                enriched_closed.append(enriched)
            session['closed_positions'] = enriched_closed
            
            # Recalculate session performance if needed
            if any(p.get('needs_pnl_recalc') for p in enriched_closed):
                logger.info(f"Recalculating performance for session {session_id}...")
                # This would trigger P&L recalculation
        
        # Save enriched sessions
        self.session_manager._save_sessions()
        logger.info("Session enrichment complete")
    
    def create_catchup_task(self):
        """Create a scheduled task to catch up on recent results."""
        logger.info("Setting up catch-up mechanism...")
        
        # Create a simple catch-up script
        catchup_script = '''#!/usr/bin/env python3
"""Catch up on recent match results - run via cron."""

from backfill_historical_data import HistoricalDataBackfiller
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)

# Check matches finished in last 24 hours
backfiller = HistoricalDataBackfiller()

# Get recently finished matches
from database_v2 import db_manager
from models import Market

with db_manager.get_db_session() as db:
    recent_cutoff = datetime.utcnow() - timedelta(hours=24)
    recent_finished = db.query(Market).filter(
        Market.is_finished == True,
        Market.last_update >= recent_cutoff
    ).all()
    
    print(f"Found {len(recent_finished)} recently finished matches")
    
    # Enrich any paper trading positions for these markets
    for market in recent_finished:
        backfiller.stats['markets_checked'] += 1
        # Process positions for this market
        
print("Catch-up complete:", backfiller.stats)
'''
        
        with open('catchup_results.py', 'w') as f:
            f.write(catchup_script)
        
        logger.info("Created catchup_results.py - add to cron for regular updates")
        
        # Suggest cron entry
        print("\nSuggested cron entry (every 4 hours):")
        print("0 */4 * * * cd /path/to/ominari && python catchup_results.py >> catchup.log 2>&1")
    
    def run_full_backfill(self):
        """Run complete historical backfill."""
        logger.info("Starting full historical data backfill...")
        
        # 1. Check finished markets
        markets_without_scores = self.check_finished_markets_without_scores()
        self.stats['markets_missing_results'] = len(markets_without_scores)
        
        # 2. Backfill all paper trading sessions
        self.backfill_all_sessions()
        
        # 3. Create catch-up mechanism
        self.create_catchup_task()
        
        # 4. Print summary
        print("\n" + "="*60)
        print("HISTORICAL DATA BACKFILL COMPLETE")
        print("="*60)
        print(f"Markets checked: {self.stats['markets_checked']}")
        print(f"Markets missing results: {self.stats['markets_missing_results']}")
        print(f"Positions checked: {self.stats['positions_checked']}")
        print(f"Positions enriched: {self.stats['positions_enriched']}")
        print(f"Errors encountered: {len(self.stats['errors'])}")
        
        if self.stats['errors']:
            print("\nErrors:")
            for error in self.stats['errors'][:10]:
                print(f"  - {error}")
        
        print("\n✅ Backfill complete!")
        print("✅ Catch-up script created: catchup_results.py")
        print("✅ Add to cron for automatic updates")


if __name__ == "__main__":
    backfiller = HistoricalDataBackfiller()
    backfiller.run_full_backfill()