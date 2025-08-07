#!/usr/bin/env python3
import sqlite3
import json
import re
import pandas as pd
import argparse

import os

# directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(BASE_DIR, "sport_odds.db")  # <— your actual filename here

CHUNK_SIZE = 1000

BACKFILL_SQL = """
                SELECT o.id, o.source_id, o.outcome,
                       o.position AS current_position, m.position_names
                  FROM odd o
                  JOIN market m USING(source_id)
                 WHERE o.position=0 --IS NULL
                 AND o.outcome<>'option_1'
                 AND o.outcome<>''
                 AND o.outcome IS NOT NULL
                 AND o.market_type  = 'winner'
                 LIMIT ?
                """


def backfill_positions(db_path, preview=False, limit=10, chunk_size=CHUNK_SIZE):
    """
    If preview=True, prints a sample of rows with their current vs inferred positions.
    Otherwise, fully backfills all odds.position NULLs in chunks.
    """
    print("Opening database at:", db_path)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Build market → position_names map once
    cur.execute("SELECT source_id, position_names FROM market")
    mapping = {
        row["source_id"]: json.loads(row["position_names"] or "[]")
        for row in cur.fetchall()
    }

    def infer_position(names, outcome):
        # First try exact lookup in names[]
        if names:
            try:
                return names.index(outcome)
            except ValueError:
                pass
        # Fallback: parse "option_2" → 1
        m = re.match(r"^option_(\d+)$", outcome)
        if m:
            return int(m.group(1)) - 1
        return None

    if preview:
        # 1) Pull a small sample
        sample_sql = BACKFILL_SQL
        df = pd.read_sql(sample_sql, conn, params=(limit,))

        # 2) Deserialize only the few JSON blobs
        df["position_names"] = df["position_names"].apply(
            lambda s: json.loads(s or "[]")
        )

        # 3) Compute inferred_position
        # build a simple Python list of inferred positions
        inferred = []
        for _, row in df.iterrows():
            inferred.append(infer_position(row["position_names"], row["outcome"]))

        # assign that list back as a single column
        df["inferred_position"] = inferred

        # 4) Show what would be updated
        print(
            df[
                [
                    "id",
                    "source_id",
                    "position_names",
                    "outcome",
                    "current_position",
                    "inferred_position",
                ]
            ]
        )
        conn.close()
        return

    # FULL backfill in chunks
    # Ensure we have an index for fast WHERE position IS NULL
    cur.execute("CREATE INDEX IF NOT EXISTS idx_odd_position_null ON odd(position)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_odd_source ON odd(source_id)")
    conn.commit()

    while True:
        # 1) fetch next batch of NULL positions
        cur.execute(BACKFILL_SQL, (chunk_size,))
        batch = cur.fetchall()
        if not batch:
            break

        updates = []
        for row in batch:
            names = mapping.get(row["source_id"], [])
            outcome = row["outcome"]
            pos = infer_position(names, outcome)
            if pos is not None:
                updates.append((pos, row["id"]))

        # 2) apply updates in one go
        if updates:
            cur.executemany("UPDATE odd SET position = ? WHERE id = ?", updates)
            conn.commit()

    conn.close()
    print("✅ Full backfill complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Backfill or preview positions in your odds table"
    )
    p.add_argument(
        "--preview",
        "-p",
        action="store_true",
        help="Show a sample of inferred positions without writing",
    )
    p.add_argument(
        "--limit",
        "-n",
        type=int,
        default=10,
        help="Number of rows to preview when using --preview",
    )
    p.add_argument("--db", "-d", default=DB_PATH, help="Path to your SQLite DB file")
    p.add_argument(
        "--chunk", "-c", type=int, default=1000, help="Chunk size for full backfill"
    )
    args = p.parse_args()

    backfill_positions(
        db_path=args.db, preview=args.preview, limit=args.limit, chunk_size=args.chunk
    )
