# Working with the 201GB Database

The `sport_odds.db` database is 201GB, which creates challenges for indexing and querying. Here are several approaches to work with it effectively.

## Quick Start (No Indexes Required)

If you need to run backtests immediately without waiting for indexes:

```bash
# Run backtest using recent data only (fast, no indexes needed)
uv run python no_index_backtest.py
```

This uses smart queries that leverage SQLite's built-in rowid index.

## Background Indexing Solutions

### Option 1: Background Indexer (Recommended)

Create indexes incrementally in the background without blocking:

```bash
# Start background indexing
./start_background_indexing.sh

# Check status
uv run python background_indexer.py --status

# View progress
tail -f background_indexing.log
```

The indexer will:
- Create indexes one by one with low priority
- Save progress between indexes
- Resume if interrupted
- Complete in several hours without blocking database use

### Option 2: Quick Essential Indexes Only

Create only the most critical indexes:

```bash
# Create only 3 most important indexes
uv run python background_indexer.py --quick
```

This creates:
1. `idx_odd_updated_at` - For time-based queries (fastest to create)
2. `idx_market_source` - For market lookups (small table, quick)
3. `idx_odd_source_updated` - For latest odds (may take hours)

### Option 3: Manual SQL Script

Run indexing directly with SQLite:

```bash
# Run in background with low priority
nice -n 19 sqlite3 sport_odds.db < create_indexes_nonblocking.sql &
```

### Option 4: Overnight Batch

Create a full set of indexes overnight:

```bash
# Run with nohup to continue after logout
nohup nice -n 19 uv run python background_indexer.py &
```

## Performance Tips

### 1. Use Recent Data Only

The `no_index_backtest.py` script fetches only recent records using rowid:
- Always fast regardless of database size
- Good for testing and development
- Limits data to most recent N records

### 2. Enable WAL Mode

WAL (Write-Ahead Logging) improves concurrency:

```sql
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
```

### 3. Increase Cache Size

For 201GB database, use more cache:

```sql
PRAGMA cache_size=100000;  -- ~400MB cache
PRAGMA mmap_size=30000000000;  -- 30GB memory map
```

### 4. Use Time Chunks

Instead of querying entire periods, use hourly chunks:
- Reduces memory usage
- Avoids timeouts
- Can be parallelized

## Monitoring Index Creation

Check index creation progress:

```bash
# See which indexes exist
sqlite3 sport_odds.db "SELECT name FROM sqlite_master WHERE type='index';"

# Monitor database file activity
iostat -x 1 | grep sport_odds

# Check SQLite process
ps aux | grep sqlite3
```

## Cached Backtest (After Indexing)

Once indexes are created, use the optimized cached backtest:

```bash
# Run with all optimizations
uv run python cached_vectorized_backtest.py
```

This provides:
- 50-100x performance improvement
- Sliding window data caching
- Signal computation memoization
- Optional parallel processing

## Emergency Options

If nothing else works:

1. **Sample the database**: Create a smaller test database
   ```bash
   sqlite3 sport_odds.db "ATTACH 'test_odds.db' AS test; 
   CREATE TABLE test.odd AS SELECT * FROM odd WHERE rowid % 1000 = 0;"
   ```

2. **Use PostgreSQL**: Migrate to a database better suited for 200GB+ data

3. **Partition by date**: Split into monthly databases

## Expected Timelines

- Creating `idx_odd_updated_at`: 30-60 minutes
- Creating `idx_odd_source_updated`: 2-4 hours  
- Creating all indexes: 4-8 hours
- First query after index creation: May be slow (building cache)
- Subsequent queries: Very fast

## Best Practices

1. Always use background indexing for large databases
2. Start with essential indexes only
3. Run indexing during low-usage periods
4. Monitor progress with logs
5. Save state to resume if interrupted
6. Use recent data for development/testing
7. Consider database migration for production use