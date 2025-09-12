# Complete Ominari Trading System Guide

## 🌐 Web Access
**Dashboard: http://localhost:8888**
- Real-time trading metrics
- Live market tickers
- Recent trades display
- System status monitoring
- Auto-refreshes every 5 seconds

## 📝 CLI Log Monitoring

### Quick Commands:
```bash
# Follow main logs with color coding
./ominari_logs.sh

# Show only trades
./ominari_logs.sh --trades

# Show only errors
./ominari_logs.sh --errors

# Show last 100 lines (no follow)
./ominari_logs.sh -n 100

# Follow all log files
./ominari_logs.sh --all

# Search for specific pattern
./ominari_logs.sh -g "paper trading"

# Help
./ominari_logs.sh -h
```

### Direct Log Tailing:
```bash
# Tail main system log
tail -f ominari_unified.log

# Tail with grep for trades
tail -f ominari_unified.log | grep -E "trade|order|fill"

# Tail multiple logs
tail -f ominari_unified.log web_monitor.log paper_trades.log
```

## 🔄 Mirror Exchange (Copy Trading)

The system now includes `mirror_exchange.py` which:
- Syncs real-time data from Overtime Markets
- Maintains local order books
- Stores historical data for backtesting
- Enables paper trading against real market conditions

Usage:
```python
from mirror_exchange import MirrorExchange

mirror = MirrorExchange(network='optimism')
await mirror.start()

# Get live market data
markets = mirror.get_market_data()

# Place paper order against real market
result = await mirror.place_paper_order(
    'account', 'market_id', 'home', 100, 'market'
)

# Get historical data
hist_df = mirror.get_historical_data(
    start_date=datetime.now() - timedelta(days=30)
)
```

## 📊 Historical Data & Backtesting

### Fetching Historical Data:
```python
from historical_data_fetcher import HistoricalDataFetcher

fetcher = HistoricalDataFetcher()

# Fetch 30 days of data
stats = await fetcher.fetch_historical_range(
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now(),
    sports=['NFL', 'NBA', 'EPL']
)

# Export for backtesting
df = fetcher.export_for_backtesting(start_date, end_date)
```

### Running Backtests:
```bash
# Run standard backtest
python run_backtest.py

# Run vectorized backtest
python vectorized_backtest.py

# Run with specific date range
python backtest.py --start 2024-01-01 --end 2024-12-31
```

Backtest results are:
- Saved in `backtests/` directory as CSV files
- Visible in the web dashboard (backtest chart section)
- Include equity curves, Sharpe ratios, and drawdowns

## ⚙️ Current Settings

### System Configuration (`ominari_unified.py`):
```python
# Scheduling intervals
'data_collection': 300,      # 5 minutes
'signal_generation': 600,    # 10 minutes
'paper_trading': 900,        # 15 minutes
'position_rebalancing': 1800,  # 30 minutes
'alpha_research': 3600,      # 1 hour
'reporting': 7200,           # 2 hours

# Capital and Risk Limits
initial_capital: 10000.0
max_position_size: 0.1      # 10% per position
max_cluster_exposure: 0.25  # 25% correlated positions
max_total_exposure: 0.6     # 60% total
min_rebalance_threshold: 0.05  # 5% change threshold
```

### Active Signal Providers:
1. **implied_probability** - Uses market odds
2. **coin_flip** - External gRPC signal
3. **grant** - LLM-based predictions

### Data Source Priority:
1. Local GraphQL (if available)
2. Public GraphQL
3. Direct blockchain
4. API (backup only)

## 🚀 System Status

Check current status:
```bash
# System status
ps aux | grep ominari_unified

# Check PID
cat ominari_daemon.pid

# Database sizes
du -h *.db

# Recent trades
sqlite3 paper_trades.db "SELECT * FROM paper_orders ORDER BY timestamp DESC LIMIT 10;"
```

## 📈 Live Features in Dashboard

1. **Market Tickers**: Scrolling ticker showing live odds changes
2. **Order Book**: Real-time bid/ask from mirror exchange
3. **Trade Feed**: Latest paper trades with P&L
4. **Signal Status**: Current signal strengths
5. **Performance Chart**: Equity curve from backtests
6. **Log Viewer**: Filtered, color-coded logs

## 🔧 Troubleshooting

### If web monitor won't start:
```bash
# Kill any existing processes
pkill -f "python.*monitor"

# Start unified monitor
python monitor_unified.py
```

### If logs aren't updating:
```bash
# Check if system is running
ps aux | grep ominari_unified

# Restart if needed
./start_ominari.sh
```

### Database issues:
```bash
# Backup databases
cp *.db backups/

# Check integrity
sqlite3 sport_odds.db "PRAGMA integrity_check;"
```

## 🎯 Next Steps

1. **Increase Historical Data**:
   ```bash
   python historical_data_fetcher.py --days 90 --sports NFL NBA
   ```

2. **Optimize Backtesting**:
   - Adjust signal weights in `signals.py`
   - Modify strategy parameters in `carver_framework.py`

3. **Enhanced Monitoring**:
   - Set up alerts for large trades
   - Add performance benchmarks
   - Create daily reports

The system is now fully operational with comprehensive monitoring, copy trading from real exchanges, and extensive historical data capabilities!