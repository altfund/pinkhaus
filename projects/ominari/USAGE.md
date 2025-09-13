# Ominari Trading System - Usage Guide

## Overview

The Ominari Trading System is a sophisticated sports betting analysis platform that uses Kelly Criterion optimization with risk management to identify and execute paper trades on soccer markets.

## Quick Start

1. **Start the System**
   ```bash
   ./start_system.sh
   ```

2. **Access the Dashboard**
   Open your browser to: http://localhost:8888

3. **Stop the System**
   ```bash
   ./stop_system.sh
   ```

## Dashboard Guide

### Main Tab (Markets View)
- **Real-time market data**: Shows upcoming soccer matches with odds
- **Signal analysis**: Each market shows implied probability and edge calculations
- **Game chunks**: Markets are grouped by start time for easier navigation
- **Auto-refresh**: Data updates automatically every 30 seconds

### Trading Tab
- **Portfolio Overview**: Shows current portfolio value, cash available, and positions
- **Execute Trades**: Click the green "Execute Trades Now" button to run paper trading
- **Trading Log**: Real-time log of all trading activity
- **Open Positions**: Current holdings with P&L tracking
- **Recent Trades**: History of executed trades

### Strategy Tab
- **Performance Metrics**: Sharpe ratio, volatility, expected returns
- **Signal Weights**: Current signal configuration (80% implied probability, 20% coin flip)
- **Cash Flow Analysis**: Realistic return expectations
- **Recent Sessions**: Historical performance data

## How Paper Trading Works

1. **Market Evaluation** (Every 15 minutes)
   - System fetches current odds from Overtime Markets API
   - Calculates implied probabilities with overround adjustment
   - Identifies markets with positive or negative edge

2. **Kelly Optimization**
   - Uses fractional Kelly (25%) for conservative sizing
   - Allows negative EV bets for portfolio hedging
   - Applies multiple risk limits:
     - Max 25% per game
     - Max 25% per single bet
     - Max 10% per market type
     - Min $10 bet size

3. **Portfolio Rebalancing**
   - Checks existing positions before placing new trades
   - Only trades the delta (difference) to avoid double betting
   - Closes positions when Kelly no longer recommends them

4. **Session Management**
   - All trades are recorded with timestamps
   - Positions are tracked with average entry prices
   - P&L is calculated when markets resolve

## Risk Management

The system implements strict risk controls:

- **Position Limits**: Never uses more than ~25% of bankroll
- **Diversification**: Spreads bets across multiple markets
- **Rebalancing**: Adjusts positions as probabilities change
- **Stop Loss**: Can close positions that exceed loss limits

## Data Flow

```
Overtime API → SQLite Database → Paper Trading Engine → Web Dashboard
     ↓                                    ↓
   Odds Data                    Kelly Optimization
                                         ↓
                                 Risk Management
                                         ↓
                                 Trade Execution
```

## Troubleshooting

### No Markets Showing
- Check if data collection is running: `ps aux | grep free_data_pull`
- Verify database has recent data: `python safe_query.py recent --hours 1`
- Ensure you're looking at the correct time window (next 24 hours)

### No Trades Executing
- Markets need sufficient edge (usually >0.1%)
- Check trading log for "No tradeable markets found"
- Verify risk limits aren't preventing trades

### Dashboard Not Loading
- Check web monitor is running: `curl http://localhost:8888/api/status`
- Look for errors in: `tail -f logs/web_monitor.log`
- Ensure port 8888 is not in use by another process

## Configuration

Key parameters in the system:

- **Kelly Fraction**: 25% (conservative)
- **Evaluation Frequency**: 15 minutes
- **Market Window**: Next 24 hours
- **Initial Bankroll**: $10,000

To modify these, edit the relevant files:
- `simple_paper_trading.py`: Kelly parameters
- `web_monitor.py`: Dashboard configuration
- `free_data_pull.py`: Data collection settings

## Best Practices

1. **Start Small**: The default $10,000 bankroll is for paper trading
2. **Monitor Regularly**: Check the trading log for system health
3. **Review Positions**: Ensure positions align with your risk tolerance
4. **Study Results**: Use the performance metrics to understand strategy effectiveness

## Advanced Usage

### Manual Backtesting
```bash
python run_backtest.py
```

### Direct Database Queries (Safe)
```bash
python safe_query.py summary
python safe_query.py count Market --filter "sport = 'Soccer'"
python safe_query.py sample Odd --limit 10
```

### Custom Signals
To add custom signals, inherit from `SignalProvider` in `signals.py` and implement the `get_probs()` method.

## Support

For issues or questions:
1. Check logs in the `logs/` directory
2. Review error messages in the trading log
3. Consult the technical documentation in other .md files

Remember: This is a paper trading system for research and analysis. Always understand the risks before real trading.