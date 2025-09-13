# Results Catch-up Integration

## Overview

The historical results catch-up service has been integrated into the Ominari infrastructure to automatically update match results and enrich paper trading positions with historical data.

## Integration Points

### 1. **run_everything.py** (Main Scheduler)
- Added `results_catchup_service.py` to run every 30 minutes
- Configured with 6-hour lookback window for safety
- Runs as: `python results_catchup_service.py --mode once --lookback 6`

### 2. **ominari_unified.py** (Unified Trading System)
- Added `results_catchup` task with 30-minute interval
- Integrated into the async scheduler loop
- Runs in executor to handle synchronous code

## How It Works

### Scheduled Execution
Every 30 minutes, the system will:
1. Check for matches that finished in the last 6 hours
2. Update paper trading positions with final results
3. Enrich positions with historical odds and edges
4. Save state to avoid reprocessing

### What Gets Updated
- **Match Results**: Scores and resolved outcomes
- **Position Status**: Changes from 'pending' to 'won'/'lost'
- **Historical Data**: Odds at trade time, edge calculations
- **P&L**: Verified and corrected if needed

## Manual Operations

### Initial Backfill
For first-time setup or after extended downtime:
```bash
python run_historical_backfill.py
```

### Check Status
```bash
python check_historical_data.py
```

### Force Manual Update
```bash
python results_catchup_service.py --mode once --lookback 24
```

## Monitoring

### Log Files
- Main scheduler: `scheduler_output.log`
- Unified system: `ominari_unified.log`
- Catch-up state: `catchup_state.json`

### Check Last Run
```bash
cat catchup_state.json | jq .last_run
```

### View Recent Activity
```bash
tail -f scheduler_output.log | grep "results_catchup"
```

## Troubleshooting

### Results Not Updating
1. Check if service is running:
   ```bash
   grep "Results catch-up" scheduler_output.log | tail -10
   ```

2. Check for errors:
   ```bash
   grep -i error catchup_state.json
   ```

3. Run manually with verbose output:
   ```bash
   python results_catchup_service.py --mode once --lookback 12
   ```

### Performance Impact
- Minimal: Queries only finished matches
- Database operations are limited and indexed
- Runs in background without blocking trading

## Configuration

### Adjust Frequency
In `run_everything.py`:
```python
"results_catchup_service.py": {
    "interval": 30,  # Change to desired minutes
    "depends_on": [],
    "args": ["--mode", "once", "--lookback", "6"]
},
```

In `ominari_unified.py`:
```python
'results_catchup': 1800,  # Change to desired seconds
```

### Adjust Lookback Window
Modify the `--lookback` parameter to check further back in time (in hours).

## Benefits

1. **Automated**: No manual intervention needed
2. **Reliable**: Catches up automatically after downtime
3. **Complete**: Ensures all historical data is filled
4. **Efficient**: Only processes what's needed

Your historical results will now stay up-to-date automatically!