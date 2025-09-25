#!/usr/bin/env python3
"""
Investigate why Draw/Away odds are missing
"""
import os
import psycopg2
from psycopg2.extras import RealDictCursor

# Set environment
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

def investigate_odds():
    """Deep dive into odds data structure."""
    print("🔍 Investigating Odds Data Issue")
    print("=" * 60)
    
    conn = psycopg2.connect(
        host=os.environ['PG_HOST'],
        port=os.environ['PG_PORT'],
        database=os.environ['PG_DB'],
        user=os.environ['PG_USER'],
        password=os.environ['PG_PASSWORD'],
        cursor_factory=RealDictCursor
    )
    cur = conn.cursor()
    
    try:
        # 1. Check odds table structure
        print("\n1️⃣ Checking odds table structure...")
        cur.execute("""
            SELECT column_name, data_type, is_nullable
            FROM information_schema.columns
            WHERE table_name = 'odd'
            ORDER BY ordinal_position
        """)
        print("Columns in 'odd' table:")
        for col in cur.fetchall():
            print(f"   - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")
        
        # 2. Sample raw odds data
        print("\n2️⃣ Sampling raw odds data...")
        cur.execute("""
            SELECT * FROM odd 
            WHERE market_id IN (
                SELECT id FROM market 
                WHERE is_finished = false 
                AND kickoff_time > NOW()
                LIMIT 5
            )
            ORDER BY market_id, outcome
        """)
        
        odds = cur.fetchall()
        if odds:
            print(f"Found {len(odds)} odds records")
            current_market = None
            for odd in odds:
                if odd['market_id'] != current_market:
                    current_market = odd['market_id']
                    print(f"\n📊 Market ID: {current_market}")
                print(f"   - {odd['outcome']}: {odd.get('decimal_odds', 'NULL')} (provider: {odd.get('provider', 'N/A')})")
        
        # 3. Check odds distribution
        print("\n3️⃣ Checking odds distribution by outcome...")
        cur.execute("""
            SELECT 
                outcome,
                COUNT(*) as count,
                COUNT(CASE WHEN decimal_odds > 1 THEN 1 END) as valid_odds,
                AVG(CASE WHEN decimal_odds > 1 THEN decimal_odds END) as avg_odds,
                MIN(CASE WHEN decimal_odds > 1 THEN decimal_odds END) as min_odds,
                MAX(CASE WHEN decimal_odds > 1 THEN decimal_odds END) as max_odds
            FROM odd
            WHERE market_id IN (
                SELECT id FROM market WHERE is_finished = false
            )
            GROUP BY outcome
            ORDER BY outcome
        """)
        
        print("\nOdds distribution:")
        for row in cur.fetchall():
            print(f"   {row['outcome'].upper() if row['outcome'] else 'NULL'}:")
            print(f"      Total: {row['count']}")
            print(f"      Valid (>1): {row['valid_odds']} ({row['valid_odds']/row['count']*100:.1f}%)")
            if row['avg_odds']:
                print(f"      Range: {row['min_odds']:.2f} - {row['max_odds']:.2f} (avg: {row['avg_odds']:.2f})")
        
        # 4. Check if it's a join issue
        print("\n4️⃣ Checking market-odds relationship...")
        cur.execute("""
            SELECT 
                m.id,
                m.home_team,
                m.away_team,
                COUNT(o.id) as odds_count,
                STRING_AGG(o.outcome || ':' || COALESCE(o.decimal_odds::text, 'NULL'), ', ') as all_odds
            FROM market m
            LEFT JOIN odd o ON o.market_id = m.id
            WHERE m.is_finished = false
            AND m.kickoff_time > NOW()
            GROUP BY m.id, m.home_team, m.away_team
            LIMIT 10
        """)
        
        print("\nSample market-odds relationships:")
        for row in cur.fetchall():
            print(f"\n{row['home_team']} vs {row['away_team']} (ID: {row['id']})")
            print(f"   Odds found: {row['odds_count']}")
            print(f"   Details: {row['all_odds']}")
        
        # 5. Check if source_id vs market_id is the issue
        print("\n5️⃣ Checking source_id usage...")
        cur.execute("""
            SELECT 
                COUNT(DISTINCT market_id) as by_market_id,
                COUNT(DISTINCT source_id) as by_source_id
            FROM odd
        """)
        counts = cur.fetchone()
        print(f"Unique market_ids in odds: {counts['by_market_id']}")
        print(f"Unique source_ids in odds: {counts['by_source_id']}")
        
        # Check if we're using the wrong join key
        cur.execute("""
            SELECT 
                m.source_id,
                m.home_team,
                m.away_team,
                COUNT(o.id) as odds_by_source,
                STRING_AGG(o.outcome || ':' || COALESCE(o.decimal_odds::text, 'NULL'), ', ') as odds_details
            FROM market m
            LEFT JOIN odd o ON o.source_id = m.source_id
            WHERE m.is_finished = false
            AND m.kickoff_time > NOW()
            GROUP BY m.source_id, m.home_team, m.away_team
            HAVING COUNT(o.id) > 1
            LIMIT 5
        """)
        
        print("\n🔑 Joining by source_id instead:")
        for row in cur.fetchall():
            print(f"\n{row['home_team']} vs {row['away_team']} (source: {row['source_id']})")
            print(f"   Odds found: {row['odds_by_source']}")
            print(f"   Details: {row['odds_details']}")
        
    finally:
        conn.close()
    
    print("\n" + "=" * 60)
    print("💡 Investigation complete. Check findings above.")

if __name__ == "__main__":
    investigate_odds()