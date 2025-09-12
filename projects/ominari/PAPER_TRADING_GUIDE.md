# Ominari Paper Trading Guide

## Overview

The Ominari system now includes a fully integrated paper trading engine that runs automatically with the main application. Paper trading simulates real trades without using real money, allowing you to test and refine strategies safely.

## Architecture

### Unified System (`ominari_unified.py`)
- **Single Application**: Everything runs from one Python process
- **Integrated Scheduler**: No need for separate cron jobs or schedulers
- **Automatic Tasks**:
  - Data collection (every 5 minutes)
  - Signal generation (every 10 minutes)
  - Paper trading execution (every 15 minutes)
  - Alpha research (every hour)
  - Performance reporting (every 2 hours)

### Key Components
1. **Data Collection**: Uses GraphQL/Blockchain first, API as backup
2. **Signal Generation**: Multiple signal providers run in parallel
3. **Paper Trading**: Simulates order execution with realistic slippage
4. **Alpha Research**: Tracks signal performance over time
5. **Reporting**: Generates detailed performance reports

## Quick Start

### Option 1: Manual Start (Testing)
```bash
# Start paper trading immediately
python start_paper_trading.py
```

### Option 2: Run Full System
```bash
# Start the unified system
python ominari_unified.py
```

### Option 3: Auto-Start on Boot (Production)
```bash
# Make setup script executable
chmod +x setup_autostart.sh

# Run setup (requires sudo)
./setup_autostart.sh

# Start the service
sudo systemctl start ominari-trading

# Check status
sudo systemctl status ominari-trading
```

## Paper Trading Configuration

Default configuration in `ominari_unified.py`:
- Initial capital: $10,000
- Bet size: $100 per trade
- Signal threshold: 0.65 (65% confidence to trade)

### Modifying Configuration

Edit the initialization in `ominari_unified.py`:
```python
self.paper_trading_engine = PaperTradingEngine(
    initial_capital=10000.0,  # Change starting capital
    db_path="paper_trades.db"
)
```

## Monitoring

### Live Logs
```bash
# Watch unified system logs
tail -f ominari_unified.log

# Watch service logs (if using systemd)
sudo journalctl -u ominari-trading -f
```

### Performance Reports
Reports are generated automatically in:
- `performance_reports/` - System performance metrics
- `alpha_reports/` - Signal performance analysis
- `paper_trades.db` - All paper trading data

### Database Queries
```sql
-- View recent paper trades
sqlite3 paper_trades.db "SELECT * FROM fills ORDER BY timestamp DESC LIMIT 10;"

-- Check portfolio performance
sqlite3 paper_trades.db "SELECT * FROM portfolio_history ORDER BY timestamp DESC LIMIT 1;"
```

## Trading Logic

The system executes trades based on:
1. **Signal Consensus**: Averages signals from all providers
2. **Threshold**: Trades when average signal > 0.65 or < 0.35
3. **Position Sizing**: Fixed $100 bets (can be modified)
4. **Risk Management**: Built-in position limits

### Signal Providers
- `ImpliedRawSignal`: Uses market implied probabilities
- `ExternalGrpcSignal`: External model integration (if configured)
- Custom signals can be added in `signals.py`

## System Status

### Check What's Running
```bash
# Check all Ominari processes
ps aux | grep -E "ominari|paper_trading" | grep -v grep

# Check Docker containers (Graph Node)
docker ps | grep ominari

# Check systemd service
sudo systemctl status ominari-trading
```

### Key Files
- `ominari_unified.py` - Main application
- `paper_trading_engine.py` - Trading simulation logic
- `signals/latest.json` - Current trading signals
- `paper_trades.db` - Trading history database

## Troubleshooting

### System Won't Start
1. Check logs: `tail -f ominari_unified.log`
2. Verify Graph Node is running: `docker ps`
3. Check database connections: `ls *.db`

### No Trades Executing
1. Verify data collection: Check `scheduler_output.log`
2. Check signals: `cat signals/latest.json`
3. Review thresholds in `_evaluate_trade()` method

### Performance Issues
1. Adjust intervals in `OminariUnifiedSystem.__init__()`
2. Reduce logging verbosity
3. Check disk space for databases

## Advanced Configuration

### Custom Trading Strategy
Modify `_evaluate_trade()` in `ominari_unified.py`:
```python
def _evaluate_trade(self, market: Market, signals: Dict) -> Optional[Dict]:
    # Add your custom logic here
    # Example: Kelly Criterion, multi-factor models, etc.
```

### Add New Signal Provider
1. Create provider in `signals.py`
2. Add to `get_signal_providers()`
3. Restart system

### Adjust Scheduling
Edit intervals in `ominari_unified.py`:
```python
self.intervals = {
    'data_collection': 180,      # 3 minutes
    'signal_generation': 300,    # 5 minutes
    'paper_trading': 300,        # 5 minutes
}
```

## Production Checklist

- [ ] Graph Node running with local subgraphs
- [ ] Environment variables configured (.env file)
- [ ] Adequate disk space for databases
- [ ] Systemd service enabled for auto-start
- [ ] Monitoring/alerting configured
- [ ] Backup strategy for databases
- [ ] Performance baselines established

## Next Steps

1. **Monitor Initial Performance**: Let system run for 24-48 hours
2. **Review Reports**: Check `performance_reports/` directory
3. **Tune Signals**: Adjust thresholds based on results
4. **Scale Up**: Increase bet sizes gradually
5. **Add Features**: Implement more sophisticated strategies

The system is designed to be autonomous - once started, it will continuously collect data, generate signals, and execute paper trades without manual intervention.