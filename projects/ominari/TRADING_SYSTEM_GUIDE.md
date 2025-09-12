# Ominari Trading System Guide

## Overview
The Ominari trading system now includes a comprehensive paper trading dashboard, real-time monitoring, and portfolio analysis tools.

## Components

### 1. Trading Dashboard (`trading_dashboard.py`)
Displays upcoming matches with odds and betting signals, manages paper trading portfolio.

**Usage:**
```bash
# Display upcoming matches with signals
python trading_dashboard.py

# Filter by sport
python trading_dashboard.py --sport Soccer

# Auto-execute trades based on signals
python trading_dashboard.py --auto-trade

# Update results for completed matches
python trading_dashboard.py --update-results
```

**Features:**
- Shows upcoming matches with current odds
- Calculates betting signals and edges
- Displays best betting opportunities
- Manages paper trading portfolio ($10,000 starting capital)
- Executes trades using Kelly criterion sizing
- Tracks active bets

### 2. Live Monitor (`live_monitor.py`)
Real-time monitoring interface with auto-refresh.

**Usage:**
```bash
# Run with default 60-second refresh
python live_monitor.py

# Custom refresh interval (e.g., 30 seconds)
python live_monitor.py --interval 30
```

**Features:**
- Portfolio status summary
- Matches starting soon
- Recent odds movements
- Active bets tracking
- Auto-refresh display
- Signal performance metrics

### 3. Portfolio Analysis (`portfolio_analysis.py`)
Analyzes paper trading performance and generates reports.

**Usage:**
```bash
# Generate full report with plots
python portfolio_analysis.py

# Generate report without plots
python portfolio_analysis.py --no-plots
```

**Features:**
- Portfolio performance metrics (ROI, Sharpe ratio, drawdown)
- Win/loss analysis
- Trade distribution by time and confidence
- Performance visualization plots
- Trading recommendations

## Data Files

- `paper_portfolio.json` - Current portfolio state
- `paper_trades.csv` - Historical trade records
- `portfolio_plots/` - Performance visualization charts

## Demo Mode

To see the system in action with sample data:
```bash
python demo_paper_trading.py
```

This creates demo trades and portfolio data for testing.

## Signal Providers

Currently using:
- `ImpliedRawSignal` - Basic implied probability from odds

External signal providers (require gRPC servers):
- `ExternalGrpcSignal` - External predictions via gRPC
- `GrantSignal` - Grant's prediction model

## Notes

- Paper trading starts with $10,000 capital
- Uses fractional Kelly criterion (25%) for bet sizing
- Maximum 5% of bankroll per bet
- Minimum edge requirement: 2%
- Confidence threshold: 60%