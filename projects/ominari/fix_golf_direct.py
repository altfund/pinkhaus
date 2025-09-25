#!/usr/bin/env python3
"""
Fix sport misclassification for golf tournaments - direct database connection
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# Golf tournament patterns
GOLF_TOURNAMENTS = [
    'scottish open',
    'masters',
    'us open',
    'british open',
    'pga championship',
    'ryder cup',
    'golf',
    'tournament leader',
    'end of round leader',
    'arnold palmer',
    'memorial tournament',
    'wells fargo',
    'players championship'
]

def fix_golf_misclassification():
    """Fix markets misclassified as other sports when they're actually golf."""
    
    # Direct connection
    conn = psycopg2.connect(
        host='localhost',
        port=5999,
        database='ominari_production',
        user='ominari_user',
        password='ominari_2025_secure'
    )
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Build the WHERE clause
    conditions = []
    for pattern in GOLF_TOURNAMENTS:
        conditions.extend([
            f"home_team ILIKE '%{pattern}%'",
            f"away_team ILIKE '%{pattern}%'",
            f"league_name ILIKE '%{pattern}%'"
        ])
    
    where_clause = " OR ".join(conditions)
    
    # Find misclassified golf markets
    query = f"""
    SELECT source_id, sport, home_team, away_team, league_name
    FROM market
    WHERE ({where_clause})
    AND sport != 'Golf'
    """
    
    cur.execute(query)
    misclassified = cur.fetchall()
    
    print(f"Found {len(misclassified)} misclassified golf markets")
    
    # Group by current sport classification
    sport_counts = {}
    for market in misclassified:
        sport = market['sport'] or 'Unknown'
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    
    print("\nCurrent misclassifications:")
    for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sport}: {count}")
    
    # Show some examples
    print("\nExamples of misclassified markets:")
    for market in misclassified[:10]:
        print(f"  {market['sport']} -> Golf: {market['home_team']} vs {market['away_team']}")
    
    # Update to Golf
    if misclassified:
        response = input("\nUpdate these markets to Golf? (y/n): ")
        if response.lower() == 'y':
            # Update in batches
            source_ids = [m['source_id'] for m in misclassified]
            
            # Create a list of %s placeholders
            placeholders = ','.join(['%s'] * len(source_ids))
            update_query = f"""
            UPDATE market
            SET sport = 'Golf'
            WHERE source_id IN ({placeholders})
            """
            
            cur.execute(update_query, source_ids)
            conn.commit()
            print(f"\n✅ Updated {len(misclassified)} markets to Golf")
        else:
            print("❌ Cancelled - no changes made")
    else:
        print("\n✅ No misclassified golf markets found!")
    
    # Verify the update
    print("\n=== Current Sport Distribution ===")
    cur.execute("""
    SELECT sport, COUNT(source_id) as count
    FROM market
    GROUP BY sport
    ORDER BY count DESC
    LIMIT 15
    """)
    
    sports = cur.fetchall()
    for row in sports:
        print(f"{row['sport']}: {row['count']}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_golf_misclassification()