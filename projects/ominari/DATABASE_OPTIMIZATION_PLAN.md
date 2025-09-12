# Database Optimization Plan

## Current Situation
- Database size: **202.69 GB** (was 216GB, some optimization already done)
- Main culprit: **Odds table with 515,919,464 records**
- Markets: 47,046 records (reasonable)

## Optimization Strategy

### 1. Odds Table Optimization (Highest Priority)

The odds table is storing every single odds update, leading to massive duplication.

#### Current Structure Issues:
- Storing full timestamps for each update (8 bytes)
- Storing market references repeatedly
- Storing odds as FLOAT (8 bytes) instead of optimized format
- No compression or archival

#### Proposed Changes:

1. **Create Odds Archive Table** - Move old odds (>30 days) to compressed archive
2. **Create Odds Summary Table** - Store only key statistics per market/outcome
3. **Optimize Data Types**:
   - Store odds as INTEGER (multiply by 1000) - saves 4 bytes per record
   - Use SMALLINT for common values (0-65535 range)
   - Use basis points for percentages
   
4. **Add Partitioning** - Partition by date for easier archival
5. **Add Compression** - Use SQLite page-level compression

### 2. Market Table Optimization

1. **Normalize Team Names** - Create teams table, use IDs
2. **Use ENUMs** for:
   - Sport types
   - Market types  
   - Status values
   - Outcomes

3. **Optimize String Storage**:
   - Fixed-length fields where possible
   - Remove redundant data

### 3. Create Summary Tables

1. **market_daily_summary** - Pre-aggregated daily stats
2. **market_final_odds** - Only final odds before market close
3. **market_stats** - Min/max/avg odds per outcome

### 4. Archive Strategy

1. Keep only last 30 days in main odds table
2. Archive older data to compressed tables
3. Create views for transparent access

## Implementation Steps

### Phase 1: Create New Schema
```sql
-- Teams normalization
CREATE TABLE teams (
    team_id INTEGER PRIMARY KEY,
    team_name VARCHAR(100) UNIQUE NOT NULL
);

-- Optimized odds table (recent data only)
CREATE TABLE odds_recent (
    market_id VARCHAR(68) NOT NULL,
    outcome TINYINT NOT NULL,  -- 0=Home, 1=Draw, 2=Away
    odds_x1000 INTEGER NOT NULL,  -- Odds * 1000
    updated_at INTEGER NOT NULL,  -- Unix timestamp
    PRIMARY KEY (market_id, outcome, updated_at)
) WITHOUT ROWID;

-- Final odds before close
CREATE TABLE odds_final (
    market_id VARCHAR(68) NOT NULL,
    outcome TINYINT NOT NULL,
    final_odds_x1000 INTEGER NOT NULL,
    min_odds_x1000 INTEGER NOT NULL,
    max_odds_x1000 INTEGER NOT NULL,
    avg_odds_x1000 INTEGER NOT NULL,
    update_count INTEGER NOT NULL,
    PRIMARY KEY (market_id, outcome)
) WITHOUT ROWID;
```

### Phase 2: Migration Script
1. Create new tables
2. Migrate recent data (last 30 days)
3. Calculate and store aggregates
4. Archive old data
5. Drop old odds table
6. Create views for compatibility

### Phase 3: Update Application Code
1. Update models to use new schema
2. Add odds conversion helpers (x1000)
3. Update queries to use summary tables

## Expected Results

### Storage Savings:
- Odds table: ~80% reduction (normalize + archive + optimize types)
- Market table: ~30% reduction (normalize teams + ENUMs)
- **Total expected: 150-170GB savings**

### Performance Gains:
- Faster queries (smaller tables)
- Better cache utilization
- Reduced I/O

### Maintenance Benefits:
- Easier backups (smaller DB)
- Faster archival process
- Better data lifecycle management

## Risk Mitigation
1. Full backup before migration
2. Test on subset first
3. Keep old schema for rollback
4. Monitor performance post-migration