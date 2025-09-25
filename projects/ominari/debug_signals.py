#!/usr/bin/env python3
"""Debug signal structure to fix 'outcome' error"""
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999',
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

import psycopg2
from psycopg2.extras import RealDictCursor
from edge_calculator import EdgeCalculator
import json

# Initialize edge calculator
edge_calculator = EdgeCalculator()

# Get a sample market
conn = psycopg2.connect(
    host=os.environ['PG_HOST'],
    port=os.environ['PG_PORT'],
    user=os.environ['PG_USER'],
    password=os.environ['PG_PASSWORD'],
    database=os.environ['PG_DB']
)
cur = conn.cursor(cursor_factory=RealDictCursor)

cur.execute("""
    SELECT 
        m.source_id as market_id,
        m.home_team,
        m.away_team,
        MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) as home_odds,
        MAX(CASE WHEN o.outcome = 'draw' THEN o.decimal_odds END) as draw_odds,
        MAX(CASE WHEN o.outcome = 'away' THEN o.decimal_odds END) as away_odds
    FROM market m
    LEFT JOIN odd o ON o.source_id = m.source_id
    WHERE m.is_finished = FALSE
    AND m.maturity_date > NOW()
    GROUP BY m.source_id, m.home_team, m.away_team
    HAVING MAX(CASE WHEN o.outcome = 'home' THEN o.decimal_odds END) IS NOT NULL
    LIMIT 1
""")

market = cur.fetchone()
print("Sample market:", market)

if market:
    # Calculate edge - note: edge_calculator.calculate_edges only takes markets
    edges = edge_calculator.calculate_edges([market])
    
    print("\nEdge structure:")
    print(json.dumps(edges[0], indent=2, default=str))
    
    # Check keys
    print("\nEdge keys:", edges[0].keys())
    
    # Check if edge contains outcome-based structure
    edge_data = edges[0].get('edge', {})
    print("\nEdge data type:", type(edge_data))
    print("Edge data:", edge_data)

conn.close()