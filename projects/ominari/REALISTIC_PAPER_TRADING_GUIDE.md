# Realistic Paper Trading Guide - Ominari Trading System

## Overview

This guide explains how to run realistic paper trading using live blockchain data without risking real money. The system pulls real-time odds from blockchain oracles and executes simulated trades based on your signals and strategies.

## Quick Start (5 Minutes)

```bash
# 1. Set up API authentication (if not done already)
python setup_api_auth.py
export OMINARI_API_KEY="omin_your_key_here"

# 2. Set up realistic paper trading
python setup_realistic_paper_trading.py

# 3. Start paper trading with blockchain data
python run_realistic_paper_trading.py

# 4. Monitor in another terminal
python monitor_paper_trading.py
```

That's it! Your bot is now paper trading with real blockchain data.

## What Makes It Realistic?

### 1. **Real-Time Blockchain Data**
- Pulls odds directly from Optimism and Arbitrum contracts
- Uses the same oracles that real money markets use
- Updates every minute (configurable)

### 2. **Actual Market Conditions**
- Real spreads and overround
- Actual liquidity constraints
- Live market movements

### 3. **Production Trading Logic**
- Same signal providers as live trading
- Identical Kelly criterion calculations
- Real position management rules

### 4. **No Simulation Artifacts**
- No perfect fills
- No looking ahead in time
- Real latency considerations

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Blockchain    │────▶│    Signal    │────▶│   Trading   │
│    Readers      │     │   Registry   │     │   Engine    │
└─────────────────┘     └──────────────┘     └─────────────┘
         │                       │                     │
         ▼                       ▼                     ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│  Optimism RPC   │     │   Implied    │     │Paper Trading│
│  Arbitrum RPC   │     │   Oracle     │     │  Database   │
│  Fallback RPCs  │     │   Hybrid     │     │   (SQLite)  │
└─────────────────┘     └──────────────┘     └─────────────┘
```

## Configuration Options

### Paper Trading Config (`paper_trading_config.json`)

```json
{
  "paper_trading": {
    "initial_capital": 10000.0,      // Starting bankroll
    "data_sources": {
      "primary": "blockchain",       // Use blockchain first
      "fallback": "database",        // Fall back to historical
      "update_interval": 60          // Check every 60 seconds
    }
  },
  "signals": {
    "registry": [
      {
        "name": "blockchain_oracle",  // Real oracle data
        "weight": 0.5,
        "enabled": true
      },
      {
        "name": "implied_raw",       // Market implied odds
        "weight": 0.5,
        "enabled": true
      }
    ],
    "min_edge": 0.02,               // 2% minimum edge
    "max_bet_fraction": 0.05        // Max 5% per bet
  },
  "trading": {
    "kelly_fraction": 0.25,         // Conservative Kelly
    "max_positions": 20,            // Position limit
    "update_frequency": 300         // Trade every 5 min
  }
}
```

## Available Signals

### 1. Blockchain Oracle Signal
```python
# Uses real oracle prices from blockchain
from blockchain_signal import BlockchainOracleSignal

signal = BlockchainOracleSignal(networks=['optimism', 'arbitrum'])
```

### 2. Hybrid Blockchain Signal
```python
# Combines oracle data with historical patterns
from blockchain_signal import HybridBlockchainSignal

signal = HybridBlockchainSignal()
```

### 3. Implied Raw Signal
```python
# Uses market implied probabilities
from signals import ImpliedRawSignal

signal = ImpliedRawSignal()
```

## Monitoring Your Performance

### Real-Time Dashboard
```bash
# Terminal-based monitoring
python monitor_paper_trading.py [session_id]

# Web dashboard (more features)
python web_monitor_with_metrics.py
# Then open http://localhost:8888/dashboard
```

### Performance Analysis
```bash
# Get detailed analysis
python analyze_paper_trading.py [session_id]

# Export to CSV
python export_paper_trades.py [session_id] --format csv
```

### Metrics Available
- Total P&L and percentage return
- Win rate and profit factor
- Sharpe ratio and max drawdown
- Position distribution by sport/league
- Hourly activity patterns
- Edge vs actual results

## Advanced Usage

### Running Multiple Strategies
```bash
# Strategy 1: Conservative
python setup_realistic_paper_trading.py --name "Conservative" \
  --kelly 0.1 --min-edge 0.03

# Strategy 2: Aggressive  
python setup_realistic_paper_trading.py --name "Aggressive" \
  --kelly 0.5 --min-edge 0.01

# Compare results
python compare_strategies.py Conservative Aggressive
```

### Custom Signal Combinations
```python
# In your paper_trading_config.json
"signals": {
  "registry": [
    {"name": "blockchain_oracle", "weight": 0.4},
    {"name": "implied_raw", "weight": 0.3},
    {"name": "your_custom_signal", "weight": 0.3}
  ]
}
```

### Backtesting vs Paper Trading
```python
# Backtest with historical data
python backtest.py --start 2024-01-01 --end 2024-12-31

# Paper trade with live data
python run_realistic_paper_trading.py

# Compare results
python compare_backtest_vs_paper.py
```

## Common Scenarios

### 1. Testing a New Signal
```python
# Add your signal to the registry
from signal_registry import SignalRegistry
from your_module import YourNewSignal

registry = SignalRegistry()
registry.add_signal("your_new_signal", weight=0.5)

# Run paper trading
python run_realistic_paper_trading.py
```

### 2. Optimizing Kelly Fraction
```bash
# Run multiple sessions with different Kelly
for kelly in 0.1 0.25 0.5; do
  python setup_realistic_paper_trading.py --kelly $kelly
done

# Analyze results after a week
python analyze_kelly_performance.py
```

### 3. League-Specific Testing
```python
# Configure to only trade specific leagues
"filters": {
  "leagues": ["Premier League", "La Liga"],
  "min_liquidity": 1000,
  "sports": ["Soccer"]
}
```

## Transitioning to Live Trading

When your paper trading shows consistent profits:

### 1. Performance Checklistlist
- [ ] Minimum 1000 paper trades
- [ ] Consistent profit over 30+ days  
- [ ] Max drawdown < 20%
- [ ] Win rate matches backtest
- [ ] No manual interventions

### 2. Start Small
```python
# Begin with minimal capital
config['trading']['initial_capital'] = 100  # Start with $100
config['trading']['kelly_fraction'] = 0.1   # Very conservative
```

### 3. Gradual Scaling
```python
# Scale up based on performance
if win_rate > 0.55 and profit_factor > 1.2:
    config['trading']['kelly_fraction'] *= 1.1  # 10% increase
```

## Troubleshooting

### RPC Connection Issues
```bash
# Test blockchain connectivity
python rpc_config.py test

# Use free endpoints if needed
export USE_FREE_RPC=true
```

### No Trading Opportunities
```python
# Lower minimum edge temporarily
config['signals']['min_edge'] = 0.01  # 1% edge

# Check signal outputs
python debug_signals.py
```

### Database Issues
```bash
# Reset paper trading database
rm paper_trading.db
python paper_trading_db.py init

# Check database integrity
python check_database.py
```

## Best Practices

1. **Run Continuously**: Let it run 24/7 for realistic results
2. **Don't Interfere**: Avoid manual position changes
3. **Monitor Daily**: Check performance but don't overtrade
4. **Document Changes**: Log any config modifications
5. **Compare Strategies**: Run multiple sessions in parallel

## Integration with Monitoring

The paper trading system integrates with Prometheus/Grafana:

```yaml
# Metrics exported
ominari_paper_trading_pnl{session="realistic_test"}
ominari_paper_trading_positions{session="realistic_test"}
ominari_paper_trading_win_rate{session="realistic_test"}
```

View in Grafana dashboards for professional monitoring.

## Next Steps

1. **Start Paper Trading**: Run for at least 2 weeks
2. **Analyze Results**: Daily performance reviews
3. **Optimize Signals**: Based on real results
4. **Scale Testing**: Increase capital in paper trading
5. **Go Live**: When consistently profitable

## Support

- Check logs: `tail -f logs/paper_trading.log`
- Debug mode: `python run_realistic_paper_trading.py --debug`
- Join Discord: [Community link]

Remember: Paper trading profits don't guarantee live trading success, but realistic paper trading gets you as close as possible to the real experience!