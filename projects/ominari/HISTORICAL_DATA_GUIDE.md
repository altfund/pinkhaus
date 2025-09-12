# Historical Data Management Guide

## Overview

This guide explains how to ensure all historical match results, odds, and edges are properly filled and kept up-to-date in your Ominari trading system.

## Components

### 1. **Historical Data Backfiller** (`backfill_historical_data.py`)
- Checks all finished markets for missing scores/results
- Enriches paper trading positions with historical market data
- Adds resolved outcomes, scores, and finish times to positions
- Creates catch-up scripts for ongoing updates

### 2. **Trading History Enricher** (`enrich_trading_history.py`)
- Adds detailed odds history to all positions
- Calculates edges at trade time based on signals vs implied probability
- Tracks odds movements during position lifetime
- Verifies P&L calculations
- Adds comprehensive statistics (min/max odds, edge stats, etc.)

### 3. **Results Catch-up Service** (`results_catchup_service.py`)
- Monitors for recently finished matches
- Automatically updates paper trading positions with results
- Can run as a one-time job or continuous service
- Maintains state to avoid reprocessing
- Handles position closing for finished markets

### 4. **Status Checker** (`check_historical_data.py`)
- Shows current state of historical data
- Reports on match results coverage
- Shows paper trading enrichment status
- Displays catch-up service status

## Usage

### Initial Setup (One-Time)

1. **Run the complete backfill**:
   ```bash
   python run_historical_backfill.py
   ```
   This will:
   - Backfill all historical match results
   - Enrich all paper trading positions with odds/edges
   - Set up the catch-up service

2. **Check the status**:
   ```bash
   python check_historical_data.py
   ```

### Ongoing Maintenance

#### Option 1: Cron Job (Recommended)
```bash
# Set up cron job
python results_catchup_service.py --mode setup

# Add to crontab (runs every 30 minutes)
*/30 * * * * cd /path/to/ominari && python results_catchup_service.py --mode once >> logs/catchup.log 2>&1
```

#### Option 2: Systemd Service
```bash
# Install the service
sudo cp ominari-catchup.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ominari-catchup.service
sudo systemctl start ominari-catchup.service

# Check status
sudo systemctl status ominari-catchup.service
```

#### Option 3: Manual Run
```bash
# Run once
python results_catchup_service.py --mode once

# Run continuously
python results_catchup_service.py --mode continuous --interval 30
```

## What Gets Added

### To Markets in Database:
- `resolved_outcome`: Computed from scores (Home/Draw/Away)
- `home_score`, `away_score`: Final match scores
- `is_finished`: Match completion status

### To Paper Trading Positions:
```python
{
    'market_result': {
        'is_finished': True,
        'resolved_outcome': 'Home',
        'home_score': 2,
        'away_score': 1,
        'finish_time': '2025-09-08T16:30:00Z'
    },
    'historical_odds': {
        'Home': 2.50,
        'Draw': 3.20,
        'Away': 2.80
    },
    'odds_movement': {
        'opening': 2.45,
        'closing': 2.55,
        'min': 2.40,
        'max': 2.60,
        'changes': 15
    },
    'calculated_edge': 2.5,
    'edge_stats': {
        'avg_edge': 2.3,
        'min_edge': 1.8,
        'max_edge': 3.1,
        'trades_with_edge': 3
    }
}
```

## Dashboard Integration

The unified dashboard will automatically show:
- Complete results for all closed matches
- Final scores in format "Home (2-1)"
- Proper win/loss status for positions
- Historical odds and edges in position details

## Troubleshooting

1. **Missing Results**: Run `python backfill_historical_data.py`
2. **Positions Not Updating**: Check catch-up service logs
3. **Dashboard Not Showing Results**: Ensure API is fetching finished markets
4. **P&L Mismatches**: Run enricher with verification: `python enrich_trading_history.py`

## Monitoring

Check logs for ongoing updates:
```bash
# Catch-up service logs
tail -f logs/catchup.log

# Check state
cat catchup_state.json
```

## Best Practices

1. Run initial backfill during low activity periods
2. Set catch-up frequency based on your trading volume (30-60 minutes typical)
3. Monitor logs for any persistent errors
4. Periodically run status check to ensure data completeness

Your historical data is now complete and will automatically stay up-to-date!