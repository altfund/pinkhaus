# Position Tracker Guide

## Overview

The Position Tracker is a real-time monitoring tool that displays open trading positions, portfolio performance, and sends alerts for important events.

## Quick Start

### Single Snapshot
```bash
uv run python position_tracker.py --once
```

### Continuous Monitoring (updates every 60 seconds)
```bash
uv run python position_tracker.py
```

### Custom Update Interval
```bash
# Update every 30 seconds
uv run python position_tracker.py --interval 30

# Update every 5 minutes
uv run python position_tracker.py --interval 300
```

### Monitor Specific Session
```bash
uv run python position_tracker.py --session paper_trading_sessions_production.json
```

## Features

### Dashboard Display

**Portfolio Summary:**
- Initial bankroll
- Current cash balance
- Total portfolio value (cash + open positions)
- Overall P&L ($ and %)
- Trade statistics (total, winning, losing)

**Open Positions:**
- Market name and outcome (HOME/AWAY/DRAW)
- Entry odds
- Stake amount
- Current position value
- Unrealized P&L ($ and %)
- Time to maturity (minutes/hours/days)

### Automated Alerts

The tracker automatically monitors and alerts on:

1. **Portfolio Changes** (⚠️/🚨)
   - Alerts when portfolio value changes by >2%
   - High severity alert when change >5%

2. **Expiring Positions** (🚨)
   - Alerts when positions expire in <30 minutes
   - High severity for positions already expired but not settled

3. **Large P&L Moves** (⚠️)
   - Alerts on unrealized gains/losses >20%

### Position Sorting

Positions are automatically sorted by maturity date (soonest first) so you can easily see which positions need attention.

## Example Output

```
========================================================================================================================
POSITION TRACKER - 2025-12-11 01:18:32 UTC
========================================================================================================================

PORTFOLIO SUMMARY:
  Initial Bankroll: $10,000.00
  Current Cash:     $8,099.90
  Portfolio Value:  $9,816.37
  Total P&L:        $-183.63 (-1.84%)

  Total Trades:     31
  Open Positions:   20
  Winning Trades:   3
  Losing Trades:    8

OPEN POSITIONS (20):
------------------------------------------------------------------------------------------------------------------------
Market                                   | Side   | Odds   | Stake          | Value           | P&L                    | TTM
------------------------------------------------------------------------------------------------------------------------
Beitar Jerusalem FC vs Ihoud Bnei Sakhni | HOME   @ 3.99 | Stake: $70.25 | Value: $70.25 | P&L:     +$0.00 (+0.0%) | TTM: EXPIRED
Real Madrid CF vs VfL Wolfsburg          | HOME   @ 3.19 | Stake: $113.48 | Value: $113.48 | P&L:     +$0.00 (+0.0%) | TTM: EXPIRED
Spirit Academy vs Zero Tenacity          | HOME   @ 2.94 | Stake: $84.97 | Value: $84.97 | P&L:     +$0.00 (+0.0%) | TTM:    1m
South Africa vs England                  | HOME   @ 3.20 | Stake: $85.12 | Value: $85.12 | P&L:     +$0.00 (+0.0%) | TTM:   14m
Angers Sporting Club de l'Ouest vs FC Na | HOME   @ 2.94 | Stake: $107.90 | Value: $107.90 | P&L:     +$0.00 (+0.0%) | TTM:   34m
Nashville Predators vs Calgary Flames    | HOME   @ 2.50 | Stake: $90.29 | Value: $90.29 | P&L:     +$0.00 (+0.0%) | TTM:   10h
========================================================================================================================

RECENT ALERTS:
  [01:18:32] expired_unsettled: Expired position not settled: Beitar Jerusalem FC vs Ihoud Bnei Sakhnin FC
  [01:18:32] expired_unsettled: Expired position not settled: Real Madrid CF vs VfL Wolfsburg
```

## Integration with Trading System

The position tracker reads from the same session files used by:
- `carver_heartbeat_with_backtest.py` - Signal generation and portfolio valuation
- `execute_trading_solution.py` - Live trade execution
- `web_dashboard_real_odds.py` - Web dashboard

All components share the same session data, ensuring consistency across the system.

## Running in Background

To run continuously in the background:

```bash
# Start tracker
uv run python position_tracker.py > logs/position_tracker.log 2>&1 &

# View live output
tail -f logs/position_tracker.log

# Stop tracker
pkill -f position_tracker.py
```

## Notifications

Currently, notifications are logged to console and the notification log within the tracker. Future enhancements could include:
- Discord/Slack notifications
- Email alerts
- SMS for critical events
- Desktop notifications

## Troubleshooting

**No session file found:**
- Ensure environment variable `OMINARI_ENV` is set (dev/staging/production)
- Or specify session file with `--session` parameter

**Positions showing "?":**
- Check that maturity_date field exists in session data
- Verify date format is ISO 8601 compatible

**No alerts appearing:**
- Alerts only trigger on significant events (>2% changes, positions expiring <30min)
- Run in continuous mode to detect changes over time
