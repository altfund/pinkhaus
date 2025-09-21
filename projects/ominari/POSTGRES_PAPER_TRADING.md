# PostgreSQL Paper Trading Integration

## Overview

The paper trading system has been migrated from JSON file storage to PostgreSQL for better reliability, concurrent access, and data integrity. The new system provides:

- **Atomic transactions** - No more corrupted JSON files
- **Concurrent access** - Multiple processes can read/write safely
- **Better performance** - Indexed queries and aggregations
- **Data persistence** - Automatic backups with PostgreSQL
- **Rich querying** - SQL analytics on trading history

## Architecture

### Database Tables

1. **paper_trading_sessions** - Trading session metadata
   - `session_id` - Unique identifier
   - `session_name` - Descriptive name
   - `initial_bankroll` - Starting capital
   - `status` - active/closed
   - `created_at` - Timestamp

2. **paper_trading_positions** - Individual trades/bets
   - `position_id` - Auto-incrementing ID
   - `session_id` - Link to session
   - `bet_id` - Unique bet identifier
   - `match_id`, `sport`, `home_team`, `away_team` - Game details
   - `bet_on` - home/draw/away
   - `odds`, `stake`, `potential_return` - Betting details
   - `status` - pending/won/lost/cancelled
   - `placed_at`, `settled_at` - Timestamps
   - `pnl` - Profit/loss when settled

3. **paper_trading_snapshots** - Point-in-time session state
   - `snapshot_id` - Auto-incrementing ID
   - `session_id` - Link to session
   - `cash_balance` - Available cash
   - `positions_value` - Value of open positions
   - `portfolio_value` - Total value
   - `win_count`, `loss_count`, `pending_count` - Statistics

## Integration with web_monitor.py

The system uses `paper_trading_postgres_integrated.py` which provides a drop-in replacement for the JSON-based `PaperTradingSessionManager`. No code changes required in web_monitor.py except the import statement.

## Usage

### Testing the Integration
```bash
python test_postgres_paper_trading.py
```

### Migrating from JSON
```bash
# Migrate existing JSON sessions to PostgreSQL
python migrate_json_to_postgres.py
```

### Fixing Over-leveraging
```bash
# Emergency fix for over-leveraged positions
python fix_overleveraging_postgres.py
```

### Monitoring
```bash
# Watch paper trading activity
./simple_monitor.sh

# Or use the Python monitor
python monitor_live_system.py
```

## Configuration

The system uses environment variables set in web_monitor.py:
```python
PG_HOST = 'localhost'
PG_PORT = '5999'
PG_USER = 'ominari_user'
PG_PASSWORD = 'ominari_2025_secure'
PG_DB = 'ominari_production'
```

## Risk Management

The PostgreSQL implementation enforces strict risk limits:
- **2% max per game** (was 25%)
- **1% max per bet** (was 25%)
- **0.5% max per market**
- **100% max total exposure**

## Known Issues

### Draw/Away Odds Missing
Currently, the database only contains HOME odds. DRAW and AWAY odds are null/zero, which causes:
- Only HOME bets being placed
- Poor diversification
- Higher risk concentration

This needs to be fixed in the odds data synchronization.

## SQL Queries for Analysis

### Check current session status:
```sql
SELECT s.*, snap.*
FROM paper_trading_sessions s
LEFT JOIN LATERAL (
    SELECT * FROM paper_trading_snapshots 
    WHERE session_id = s.session_id 
    ORDER BY snapshot_time DESC LIMIT 1
) snap ON true
WHERE s.status = 'active';
```

### Get exposure by outcome:
```sql
SELECT 
    bet_on, 
    COUNT(*) as count,
    SUM(stake) as total_stake,
    AVG(odds) as avg_odds
FROM paper_trading_positions
WHERE status = 'pending'
GROUP BY bet_on;
```

### Performance over time:
```sql
SELECT 
    DATE(placed_at) as date,
    COUNT(*) as trades,
    SUM(CASE WHEN is_winner THEN 1 ELSE 0 END) as wins,
    SUM(pnl) as daily_pnl
FROM paper_trading_positions
WHERE status = 'settled'
GROUP BY DATE(placed_at)
ORDER BY date DESC;
```

## Troubleshooting

### Connection Issues
- Verify PostgreSQL is running: `ps aux | grep postgres`
- Check port 5999 is open: `netstat -an | grep 5999`
- Test connection: `psql -h localhost -p 5999 -U ominari_user -d ominari_production`

### Migration Failed
- Check logs in `web_monitor_fixed.log`
- Verify tables exist: `python check_db_schema.py`
- Manual table creation: `python create_paper_trading_tables.py`

### Over-leveraging Returns
- Run: `python fix_overleveraging_postgres.py`
- Restart web_monitor.py
- Check exposure is < 100%