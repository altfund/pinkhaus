# Paper Trading Setup for Ominari

This document explains how to set up and run paper trading with the Ominari Trading System.

## Overview

The Ominari system now includes fully integrated paper trading capabilities that:
- Run automatically on system startup
- Collect market data every 5 minutes
- Generate trading signals every 10 minutes
- Execute paper trades every 15 minutes
- Generate performance reports every 2 hours
- Track alpha research metrics hourly

## Quick Start

### 1. Manual Start
```bash
# Start the unified system immediately
./start_ominari.sh
```

### 2. Automatic Startup Setup
```bash
# Set up automatic startup on system boot
./setup_autostart.sh

# Or manually add to crontab
crontab -e
# Add this line:
@reboot /path/to/ominari/start_ominari.sh > /tmp/ominari_startup.log 2>&1
```

### 3. Using Systemd (Recommended for Production)
```bash
# Install the service
sudo cp ominari-trading.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable ominari-trading.service

# Start the service
sudo systemctl start ominari-trading

# Check status
sudo systemctl status ominari-trading

# View logs
sudo journalctl -u ominari-trading -f
```

## Architecture

### Unified System (`ominari_unified.py`)
The core application that integrates all components:
- **Data Collection**: Uses `DataSourceManager` to fetch from blockchain/GraphQL/API
- **Signal Generation**: Runs multiple signal providers (implied probability, gRPC signals)
- **Paper Trading**: Executes trades based on signal consensus
- **Alpha Research**: Tracks signal performance over time
- **Performance Reporting**: Generates comprehensive system metrics

### Paper Trading Engine (`paper_trading_engine.py`)
Realistic trading simulation with:
- Quote collection and storage
- Market impact modeling
- Slippage calculation
- Commission tracking
- Performance metrics

### Signal Providers (`signals.py`)
- `ImpliedRawSignal`: Uses implied probabilities from odds
- `ExternalGrpcSignal`: Integrates with external gRPC services
- `GrantSignal`: Uses LLM-based predictions

## Configuration

### Scheduling Intervals
Edit in `ominari_unified.py`:
```python
self.intervals = {
    'data_collection': 300,      # 5 minutes
    'signal_generation': 600,    # 10 minutes
    'paper_trading': 900,        # 15 minutes
    'alpha_research': 3600,      # 1 hour
    'reporting': 7200,           # 2 hours
}
```

### Paper Trading Parameters
```python
# Initial capital (in ominari_unified.py)
self.paper_trading_engine = PaperTradingEngine(
    initial_capital=10000.0,  # Start with $10k
    db_path="paper_trades.db"
)

# Bet sizing (in _evaluate_trade method)
'size': 100.0,  # $100 per bet
```

### Signal Thresholds
```python
# Trade when average signal > threshold
if avg_signal > 0.65:  # Strong positive signal
    # Place bet on home team
elif avg_signal < 0.35:  # Strong negative signal
    # Place bet on away team
```

## Directory Structure
```
ominari/
├── logs/                    # System logs
├── paper_trades/           # Paper trading data
├── signals/                # Signal snapshots
├── alpha_reports/          # Alpha research reports
├── performance_reports/    # System performance metrics
├── betting_reports/        # Detailed betting analysis
├── paper_trades.db         # Paper trading database
├── alpha_research.db       # Alpha research database
└── sport_odds.db          # Main market database
```

## Monitoring

### Check System Status
```bash
# Check if running
ps aux | grep ominari_unified

# View PID
cat ominari_daemon.pid

# Tail logs
tail -f logs/ominari_*.log
```

### View Performance
```bash
# Latest performance report
ls -la performance_reports/

# Paper trading results
cat paper_trading_results.json

# Alpha research reports
ls -la alpha_reports/
```

### Database Queries
```bash
# Check paper trades
sqlite3 paper_trades.db "SELECT * FROM paper_orders ORDER BY timestamp DESC LIMIT 10;"

# View performance
sqlite3 paper_trades.db "SELECT * FROM paper_performance ORDER BY timestamp DESC LIMIT 1;"
```

## Troubleshooting

### System Won't Start
1. Check dependencies: `uv sync`
2. Check environment: `source .venv/bin/activate`
3. Check logs: `tail -f logs/ominari_*.log`

### No Trades Executing
1. Check data collection: Look for "Data collection completed" in logs
2. Check signals: Verify signals are being generated
3. Check thresholds: May need to adjust signal thresholds

### Performance Issues
1. Check database size: `du -h *.db`
2. Check memory usage: `ps aux | grep ominari`
3. Adjust intervals if needed

## Testing

### Run Paper Trading Demo
```bash
# Simple demo with simulated data
python paper_trading_simple.py
```

### Test Individual Components
```bash
# Test data collection
python -c "from free_data_pull_v2 import DataCollector; import asyncio; asyncio.run(DataCollector().run_collection_cycle())"

# Test signals
python -c "from signals import get_signal_providers; print([s.name for s in get_signal_providers()])"
```

## Advanced Configuration

### Adding New Signal Providers
1. Create signal class in `signals.py`:
```python
class MySignal(SignalProvider):
    name = "my_signal"
    
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # Your logic here
        return probabilities
```

2. Add to `SIGNAL_PROVIDERS` list in `signals.py`

### Modifying Trading Logic
Edit `_evaluate_trade` method in `ominari_unified.py` to change:
- Signal combination logic
- Threshold values
- Bet sizing
- Risk management

### Custom Reporting
Add new reporting methods to `run_reporting` in `ominari_unified.py`

## Security Notes

- Never commit `.env` files with real API keys
- Paper trading uses simulated money only
- Keep database backups: `cp *.db backups/`
- Monitor disk space for growing databases

## Next Steps

1. Monitor paper trading performance
2. Adjust signal weights based on results
3. Implement additional signal providers
4. Fine-tune betting strategies
5. Prepare for live trading transition