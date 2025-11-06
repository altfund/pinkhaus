#!/usr/bin/env python3
"""Settlement process for PostgreSQL paper trading positions"""

import os
import sys
from datetime import datetime, timezone
import requests
import logging

# Set PostgreSQL environment
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

from paper_trading_postgres_integrated import PaperTradingSessionManager
import psycopg2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PositionSettler:
    """Handles settling finished positions"""
    
    def __init__(self):
        self.session_manager = PaperTradingSessionManager()
        self.api_url = "https://api.overtime.io/overtime-v2/games-info"
    
    def fetch_match_results(self, market_ids):
        """Fetch results from Overtime API for given market IDs"""
        try:
            # Query API for all games
            response = requests.get(
                self.api_url,
                headers={'accept': 'application/json'}
            )
            
            if response.status_code == 200:
                data = response.json()
                games = data.get('games', [])
                
                results = {}
                for game in games:
                    market_id = game.get('gameId')
                    if market_id in market_ids:
                        # Check if game is finished
                        is_cancelled = game.get('isCanceled', False)
                        is_resolved = game.get('isResolved', False)
                        
                        if is_resolved and not is_cancelled:
                            home_score = game.get('homeScore', 0)
                            away_score = game.get('awayScore', 0)
                            
                            # Determine outcome
                            if home_score > away_score:
                                outcome = 'home'
                            elif away_score > home_score:
                                outcome = 'away'
                            else:
                                outcome = 'draw'
                            
                            results[market_id] = {
                                'is_finished': True,
                                'home_score': home_score,
                                'away_score': away_score,
                                'outcome': outcome,
                                'is_cancelled': is_cancelled
                            }
                            
                            logger.info(f"Found result for {market_id}: {home_score}-{away_score} ({outcome})")
                
                return results
            else:
                logger.error(f"API request failed: {response.status_code}")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching match results: {e}")
            return {}
    
    def settle_positions(self, session_id):
        """Settle all finished positions for a session"""
        
        # Get positions from database
        positions = self.session_manager.get_positions(session_id)
        pending_positions = [p for p in positions if p['status'] == 'pending']
        
        if not pending_positions:
            logger.info("No pending positions to settle")
            return 0
        
        logger.info(f"Found {len(pending_positions)} pending positions")
        
        # Collect market IDs
        market_ids = set()
        position_map = {}  # map market_id to positions
        
        for pos in pending_positions:
            market_id = pos.get('match_id')
            if market_id:
                market_ids.add(market_id)
                if market_id not in position_map:
                    position_map[market_id] = []
                position_map[market_id].append(pos)
        
        # Fetch results from API
        logger.info(f"Fetching results for {len(market_ids)} markets...")
        match_results = self.fetch_match_results(market_ids)
        
        if not match_results:
            logger.info("No finished matches found")
            return 0
        
        # Settle each position
        settled_count = 0
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
            database=os.environ['PG_DB']
        )
        
        try:
            with conn.cursor() as cur:
                for market_id, result in match_results.items():
                    positions_for_market = position_map.get(market_id, [])
                    
                    for pos in positions_for_market:
                        bet_on = pos['bet_on'].lower()
                        outcome = result['outcome']
                        
                        # Determine if won
                        won = (bet_on == outcome)
                        
                        # Calculate P&L
                        stake = float(pos['stake'])
                        odds = float(pos['odds'])
                        
                        if won:
                            pnl = stake * (odds - 1)
                            status = 'won'
                        else:
                            pnl = -stake
                            status = 'lost'
                        
                        # Update position
                        cur.execute("""
                            UPDATE betting_positions
                            SET status = %s,
                                pnl = %s,
                                settled_at = %s,
                                home_score = %s,
                                away_score = %s
                            WHERE bet_id = %s
                        """, (
                            status,
                            pnl,
                            datetime.now(timezone.utc),
                            result['home_score'],
                            result['away_score'],
                            pos['bet_id']
                        ))
                        
                        logger.info(
                            f"Settled: {pos['home_team']} vs {pos['away_team']} - "
                            f"Bet: {bet_on.upper()} @ {odds:.2f} - "
                            f"Result: {outcome.upper()} ({result['home_score']}-{result['away_score']}) - "
                            f"{status.upper()} - P&L: ${pnl:.2f}"
                        )
                        
                        settled_count += 1
                
                # Update session bankroll based on settled positions
                if settled_count > 0:
                    # Calculate total P&L from all settled positions
                    cur.execute("""
                        SELECT SUM(pnl) as total_pnl
                        FROM betting_positions
                        WHERE session_id = %s AND status IN ('won', 'lost')
                    """, (session_id,))
                    
                    result = cur.fetchone()
                    total_pnl = result[0] if result[0] else 0
                    
                    # Update session
                    cur.execute("""
                        UPDATE paper_trading_sessions
                        SET current_bankroll = starting_bankroll + %s,
                            updated_at = %s
                        WHERE session_id = %s
                    """, (total_pnl, datetime.now(timezone.utc), session_id))
                    
                    conn.commit()
                    
                    logger.info(f"Session updated with total P&L: ${total_pnl:.2f}")
        
        except Exception as e:
            logger.error(f"Error settling positions: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
        
        return settled_count
    
    def update_market_results(self):
        """Update markets table with results (optional)"""
        conn = psycopg2.connect(
            host=os.environ['PG_HOST'],
            port=os.environ['PG_PORT'],
            user=os.environ['PG_USER'],
            password=os.environ['PG_PASSWORD'],
            database=os.environ['PG_DB']
        )
        
        try:
            with conn.cursor() as cur:
                # Get all unfinished markets
                cur.execute("""
                    SELECT DISTINCT match_id 
                    FROM markets 
                    WHERE is_finished = false
                """)
                
                market_ids = set()
                for row in cur.fetchall():
                    if row[0]:
                        market_ids.add(row[0])
                
                if market_ids:
                    logger.info(f"Updating results for {len(market_ids)} markets...")
                    match_results = self.fetch_match_results(market_ids)
                    
                    for market_id, result in match_results.items():
                        cur.execute("""
                            UPDATE markets
                            SET is_finished = true,
                                home_score = %s,
                                away_score = %s,
                                updated_at = %s
                            WHERE match_id = %s
                        """, (
                            result['home_score'],
                            result['away_score'],
                            datetime.now(timezone.utc),
                            market_id
                        ))
                    
                    conn.commit()
                    logger.info(f"Updated {len(match_results)} market results")
                    
        except Exception as e:
            logger.error(f"Error updating market results: {e}")
            conn.rollback()
        finally:
            conn.close()

def main():
    """Main settlement process"""
    settler = PositionSettler()
    
    # Get current session
    session_id = settler.session_manager.get_current_session()
    
    if not session_id:
        logger.info("No active session found")
        return
    
    logger.info(f"Processing settlements for session: {session_id}")
    
    # Update market results first (optional)
    settler.update_market_results()
    
    # Settle positions
    settled_count = settler.settle_positions(session_id)
    
    if settled_count > 0:
        logger.info(f"\n✅ Successfully settled {settled_count} positions")
        
        # Show updated session status
        session = settler.session_manager.get_session(session_id)
        positions = settler.session_manager.get_positions(session_id)
        
        won = len([p for p in positions if p['status'] == 'won'])
        lost = len([p for p in positions if p['status'] == 'lost'])
        pending = len([p for p in positions if p['status'] == 'pending'])
        
        total_pnl = sum(float(p.get('pnl', 0)) for p in positions if p['status'] in ['won', 'lost'])
        
        print(f"\n📊 Session Summary:")
        print(f"   Current Bankroll: ${session['current_bankroll']:,.2f}")
        print(f"   Total P&L: ${total_pnl:,.2f}")
        print(f"   Won: {won}")
        print(f"   Lost: {lost}")
        print(f"   Pending: {pending}")
        
        if won + lost > 0:
            win_rate = won / (won + lost)
            print(f"   Win Rate: {win_rate:.1%}")
    else:
        logger.info("No positions were settled - all matches may still be in progress")

if __name__ == "__main__":
    main()