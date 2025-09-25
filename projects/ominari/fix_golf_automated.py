#!/usr/bin/env python3
"""
Fix sport misclassification for golf tournaments - automated
"""

import psycopg2
from psycopg2.extras import RealDictCursor

# More specific golf patterns
GOLF_PATTERNS = [
    # Major tournaments
    'genesis scottish open',
    'scottish open 2025',
    'the masters',
    'us open golf',
    'british open',
    'the open championship',
    'pga championship',
    'ryder cup',
    
    # Tournament formats
    'end of round % leader',
    'tournament leader',
    'top 5 finish',
    'top 10 finish',
    'make the cut',
    'hole in one',
    
    # Golf-specific terms
    'arnold palmer invitational',
    'memorial tournament',
    'wells fargo championship',
    'players championship',
    'fedex cup',
    'tour championship'
]

# Exclude patterns (to avoid false positives)
EXCLUDE_PATTERNS = [
    'vs',  # eSports often have "vs" in team names
    'academy',
    'gaming',
    'esports',
    'state',
    'wave'
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
    
    # Find true golf markets
    query = """
    SELECT source_id, sport, home_team, away_team, league_name
    FROM market
    WHERE sport != 'Golf'
    AND (
        -- Genesis Scottish Open specific
        (home_team ILIKE '%genesis scottish open%' OR away_team ILIKE '%genesis scottish open%')
        OR (home_team ILIKE '%scottish open 2025%' OR away_team ILIKE '%scottish open 2025%')
        
        -- Tournament leader markets
        OR (home_team ILIKE '%end of round%leader%' OR away_team ILIKE '%end of round%leader%')
        OR (home_team ILIKE '%tournament leader%' OR away_team ILIKE '%tournament leader%')
        
        -- Golf majors
        OR (home_team ILIKE '%the masters%' OR away_team ILIKE '%the masters%')
        OR (home_team ILIKE '%us open golf%' OR away_team ILIKE '%us open golf%')
        OR (home_team ILIKE '%british open%' OR away_team ILIKE '%british open%')
        OR (home_team ILIKE '%pga championship%' OR away_team ILIKE '%pga championship%')
        OR (home_team ILIKE '%ryder cup%' OR away_team ILIKE '%ryder cup%')
        
        -- Other golf tournaments
        OR (home_team ILIKE '%arnold palmer%' OR away_team ILIKE '%arnold palmer%')
        OR (home_team ILIKE '%memorial tournament%' OR away_team ILIKE '%memorial tournament%')
        OR (home_team ILIKE '%wells fargo%' OR away_team ILIKE '%wells fargo%')
        OR (home_team ILIKE '%players championship%' OR away_team ILIKE '%players championship%')
    )
    -- Exclude obvious non-golf
    AND home_team NOT ILIKE '%academy%'
    AND away_team NOT ILIKE '%academy%'
    AND home_team NOT ILIKE '%gaming%'
    AND away_team NOT ILIKE '%gaming%'
    AND home_team NOT ILIKE '%esports%'
    AND away_team NOT ILIKE '%esports%'
    """
    
    cur.execute(query)
    misclassified = cur.fetchall()
    
    print(f"Found {len(misclassified)} misclassified golf markets")
    
    if not misclassified:
        print("\n✅ No misclassified golf markets found!")
        cur.close()
        conn.close()
        return
    
    # Group by current sport classification
    sport_counts = {}
    for market in misclassified:
        sport = market['sport'] or 'Unknown'
        sport_counts[sport] = sport_counts.get(sport, 0) + 1
    
    print("\nCurrent misclassifications:")
    for sport, count in sorted(sport_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {sport}: {count}")
    
    # Show all examples (should be small list now)
    print("\nMarkets to fix:")
    for market in misclassified:
        print(f"  {market['sport']:15} -> Golf: {market['home_team']} vs {market['away_team']}")
    
    # Update to Golf automatically
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
    
    # Verify the update
    print("\n=== Current Sport Distribution (Top 15) ===")
    cur.execute("""
    SELECT sport, COUNT(source_id) as count
    FROM market
    GROUP BY sport
    ORDER BY count DESC
    LIMIT 15
    """)
    
    sports = cur.fetchall()
    for row in sports:
        print(f"{row['sport'] or 'Unknown':20}: {row['count']:,}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    fix_golf_misclassification()