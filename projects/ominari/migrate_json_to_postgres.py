#!/usr/bin/env python3
"""
Migrate existing JSON paper trading sessions to PostgreSQL
"""
import os
import json
import logging
from datetime import datetime
from paper_trading_postgres_integrated import PaperTradingSessionManager

# Set environment variables
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_sessions():
    """Migrate JSON sessions to PostgreSQL."""
    print("🔄 Starting migration from JSON to PostgreSQL...")
    
    # Check if JSON file exists
    json_file = "paper_trading_sessions.json"
    if not os.path.exists(json_file):
        print("❌ No JSON file found. Nothing to migrate.")
        return
    
    try:
        # Load JSON data
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        
        sessions = json_data.get("sessions", {})
        if not sessions:
            print("📭 No sessions found in JSON file.")
            return
        
        print(f"📋 Found {len(sessions)} sessions to migrate")
        
        # Initialize PostgreSQL manager
        pg_manager = PaperTradingSessionManager()
        
        # Migrate each session
        migrated = 0
        for session_id, session_data in sessions.items():
            try:
                # Check if session already exists
                existing = pg_manager.get_session(session_id)
                if existing:
                    print(f"⏭️  Session {session_id} already exists in PostgreSQL, skipping")
                    continue
                
                # Create session with custom ID
                with pg_manager.get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO paper_trading_sessions 
                            (session_id, session_name, initial_bankroll, status, created_at, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            session_id,
                            session_data.get('session_name', f'Migrated Session {session_id}'),
                            session_data.get('initial_bankroll', 10000),
                            session_data.get('status', 'active'),
                            session_data.get('created_at', datetime.now()),
                            json.dumps({
                                'migrated_from_json': True,
                                'original_data': session_data
                            })
                        ))
                        
                        # Create snapshot
                        cur.execute("""
                            INSERT INTO paper_trading_snapshots
                            (session_id, cash_balance, portfolio_value, total_pnl,
                             win_count, loss_count, pending_count)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """, (
                            session_id,
                            session_data.get('current_bankroll', session_data.get('initial_bankroll', 10000)),
                            session_data.get('portfolio_value', session_data.get('initial_bankroll', 10000)),
                            session_data.get('total_pnl', 0),
                            session_data.get('wins', 0),
                            session_data.get('losses', 0),
                            session_data.get('pending', 0)
                        ))
                        
                        # Migrate positions if any
                        positions = session_data.get('positions', {})
                        open_positions = positions.get('open', [])
                        
                        for pos in open_positions:
                            bet_id = f"{session_id}_migrated_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
                            cur.execute("""
                                INSERT INTO paper_trading_positions
                                (session_id, bet_id, match_id, home_team, away_team,
                                 bet_on, odds, stake, potential_return, status)
                                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                            """, (
                                session_id,
                                bet_id,
                                pos.get('match_id', ''),
                                pos.get('home_team', ''),
                                pos.get('away_team', ''),
                                pos.get('bet_on', ''),
                                pos.get('odds', 0),
                                pos.get('stake', 0),
                                pos.get('potential_return', pos.get('stake', 0) * pos.get('odds', 0)),
                                'pending'
                            ))
                        
                        conn.commit()
                        migrated += 1
                        print(f"✅ Migrated session: {session_id}")
                        
            except Exception as e:
                logger.error(f"Error migrating session {session_id}: {str(e)}")
                print(f"❌ Failed to migrate session {session_id}: {str(e)}")
        
        print(f"\n📊 Migration complete: {migrated}/{len(sessions)} sessions migrated")
        
        # Backup JSON file
        if migrated > 0:
            backup_file = f"paper_trading_sessions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.rename(json_file, backup_file)
            print(f"💾 Original JSON file backed up to: {backup_file}")
        
    except Exception as e:
        logger.error(f"Migration error: {str(e)}")
        print(f"❌ Migration failed: {str(e)}")

if __name__ == "__main__":
    migrate_sessions()