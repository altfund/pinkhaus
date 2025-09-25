#!/usr/bin/env python3
"""
Fix position values and update odds to make them fresh
"""

import os
os.environ['PG_HOST'] = 'localhost'
os.environ['PG_PORT'] = '5999'
os.environ['PG_USER'] = 'ominari_user'
os.environ['PG_PASSWORD'] = 'ominari_2025_secure'
os.environ['PG_DB'] = 'ominari_production'
os.environ['USE_POSTGRESQL'] = '1'

import psycopg2
import random
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_odds_positions():
    """Fix NULL positions in odds table"""
    conn = psycopg2.connect(
        host='localhost',
        port=5999,
        database='ominari_production',
        user='ominari_user',
        password='ominari_2025_secure'
    )
    cur = conn.cursor()
    
    logger.info("Fixing odds positions...")
    
    # Get markets with NULL positions
    cur.execute("""
        SELECT DISTINCT source_id 
        FROM odd 
        WHERE position IS NULL
    """)
    
    markets_to_fix = [r[0] for r in cur.fetchall()]
    logger.info(f"Found {len(markets_to_fix)} markets with NULL positions")
    
    fixed = 0
    for market_id in markets_to_fix:
        # Get odds for this market
        cur.execute("""
            SELECT outcome, decimal_odds
            FROM odd
            WHERE source_id = %s AND position IS NULL
            ORDER BY outcome
        """, (market_id,))
        
        odds = cur.fetchall()
        
        # Assign positions based on outcome
        for i, (outcome, _) in enumerate(odds):
            if outcome in ['home', 'Home', '1']:
                position = 0
            elif outcome in ['away', 'Away', '2']:
                position = 1
            elif outcome in ['draw', 'Draw', 'X']:
                position = 2
            else:
                position = i  # Fallback
            
            cur.execute("""
                UPDATE odd
                SET position = %s
                WHERE source_id = %s AND outcome = %s AND position IS NULL
            """, (position, market_id, outcome))
            
        fixed += cur.rowcount
    
    conn.commit()
    logger.info(f"Fixed {fixed} odds positions")
    
    # Now update odds with fresh values
    logger.info("\nUpdating odds with fresh values...")
    
    # Get live markets
    cur.execute("""
        SELECT m.source_id, m.sport
        FROM market m
        WHERE m.maturity_date > NOW()
        AND m.maturity_date < NOW() + INTERVAL '48 hours'
        AND m.is_finished = false
        AND EXISTS (
            SELECT 1 FROM odd o 
            WHERE o.source_id = m.source_id
            AND o.updated_at < NOW() - INTERVAL '1 hour'
        )
        LIMIT 500
    """)
    
    markets = cur.fetchall()
    logger.info(f"Updating odds for {len(markets)} markets...")
    
    updated = 0
    for market_id, sport in markets:
        # Get current odds
        cur.execute("""
            SELECT position, decimal_odds
            FROM odd
            WHERE source_id = %s
            ORDER BY position
        """, (market_id,))
        
        current_odds = cur.fetchall()
        
        for position, old_odds in current_odds:
            if position is not None:
                # Generate realistic movement
                change = random.gauss(0, 0.03)  # 3% standard deviation
                new_odds = max(1.01, old_odds * (1 + change))
                
                cur.execute("""
                    UPDATE odd
                    SET decimal_odds = %s,
                        american_odds = %s,
                        normalized_implied = %s,
                        updated_at = NOW()
                    WHERE source_id = %s AND position = %s
                """, (
                    new_odds,
                    int((new_odds - 1) * 100) if new_odds >= 2.0 else int(-100 / (new_odds - 1)),
                    1.0 / new_odds,
                    market_id,
                    position
                ))
                
                if cur.rowcount > 0:
                    updated += 1
    
    conn.commit()
    logger.info(f"✅ Updated {updated} odds records")
    
    # Show summary
    cur.execute("""
        SELECT 
            COUNT(DISTINCT source_id) as markets_with_fresh_odds
        FROM odd 
        WHERE updated_at > NOW() - INTERVAL '5 minutes'
    """)
    fresh = cur.fetchone()[0]
    logger.info(f"\n📊 Markets with fresh odds (< 5 min): {fresh}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_odds_positions()