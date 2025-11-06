#\!/usr/bin/env python3
import os
os.environ.update({
    'PG_HOST': 'localhost',
    'PG_PORT': '5999', 
    'PG_USER': 'ominari_user',
    'PG_PASSWORD': 'ominari_2025_secure',
    'PG_DB': 'ominari_production'
})

import psycopg2

conn = psycopg2.connect(
    host=os.environ['PG_HOST'],
    port=os.environ['PG_PORT'],
    user=os.environ['PG_USER'],
    password=os.environ['PG_PASSWORD'],
    database=os.environ['PG_DB']
)

cur = conn.cursor()

# Check paper_trading_sessions columns
cur.execute("""
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_name = 'paper_trading_sessions'
    ORDER BY ordinal_position
""")

print("paper_trading_sessions columns:")
for row in cur.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()
