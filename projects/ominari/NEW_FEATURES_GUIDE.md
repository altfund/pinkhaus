# New Features Quick Start Guide

## 1. Signal Registry System

### Basic Usage
```python
from signal_registry import SignalRegistry, DynamicWeightManager
from integrate_signal_registry import create_integrated_registry

# Create and populate registry
registry = create_integrated_registry()

# List available signals
signals = registry.list_signals()
print(f"Active signals: {signals}")

# Get aggregated predictions
markets_df = get_current_markets()  # Your market data
predictions = registry.aggregate_predictions(markets_df)
```

### Adding New Signals
```python
from signal_registry import BaseSignalProvider

class MyCustomSignal(BaseSignalProvider):
    def __init__(self):
        super().__init__(name="my_signal", version="1.0.0")
    
    def predict(self, markets_df):
        # Your prediction logic
        return probabilities

# Register
registry.register(MyCustomSignal())
```

### Changing Weight Strategy
```python
# Use different weighting methods
manager = DynamicWeightManager(registry)

# Options: 'equal', 'inverse_variance', 'bayesian', 'online_learning', 'regime_based'
manager.update_weights(method='bayesian')
```

## 2. Production Risk Management

### Start with Conservative Settings
```bash
# Run with conservative risk limits
python production_deploy.py \
    --risk-preset conservative \
    --bankroll 1000 \
    --session-name "test_conservative"
```

### Custom Risk Configuration
```python
from production_risk_config import ProductionRiskConfig, PositionLimits, PortfolioLimits

config = ProductionRiskConfig(
    position_limits=PositionLimits(
        max_single_bet_pct=0.01,    # 1% max per bet
        max_parlay_size=3,
        min_edge=0.02,              # 2% minimum edge
        max_odds=5.0                # No extreme longshots
    ),
    portfolio_limits=PortfolioLimits(
        max_total_exposure_pct=0.10,  # 10% max exposure
        max_positions=20
    ),
    kill_switch_enabled=True
)

# Save and use
config.save("my_risk_config.json")
```

### Emergency Stop
```python
# If things go wrong, kill switch stops all trading
risk_manager = RiskManager(config)
risk_manager.trigger_kill_switch("Manual intervention required")
```

## 3. API Client Usage

### Python Client
```python
from ominari_api_client import OminariAPIClient

client = OminariAPIClient("http://localhost:8888")

# Get portfolio status
portfolio = client.get_portfolio("session_id")
print(f"Balance: ${portfolio.cash_balance}")
print(f"Open positions: {portfolio.open_positions}")

# Execute trades (dry run first!)
result = client.execute_trades("session_id", dry_run=True)
for rec in result['recommendations']:
    print(f"{rec['market_name']}: {rec['edge']:.2%} edge, ${rec['stake']} stake")

# If happy, execute for real
if input("Execute? (y/n): ").lower() == 'y':
    result = client.execute_trades("session_id", dry_run=False)
```

### REST API
```bash
# Get system status
curl http://localhost:8888/api/status

# Get portfolio
curl "http://localhost:8888/api/trading/portfolio?session_id=main"

# Get recent activity  
curl "http://localhost:8888/api/dashboard/activity?session_id=main&hours=24"
```

## 4. Blockchain Integration

### Setup RPC Connection
```bash
# Set environment variables (get free tier from Alchemy/Infura)
export OPTIMISM_RPC_URL="https://opt-mainnet.g.alchemy.com/v2/YOUR-API-KEY"
export TRADING_PRIVATE_KEY="0x..."  # Only for real trading!
```

### Read Market Data
```python
from blockchain_reader import BlockchainReader

reader = BlockchainReader(network='optimism')

# Fetch recent markets
markets = reader.fetch_recent_markets(hours_back=24)
for market in markets:
    print(f"{market['home_team']} vs {market['away_team']}")
    
    # Get current odds
    odds = reader.get_current_odds(market['address'])
    print(f"Odds: {odds}")
```

### Paper Trading Mode (Recommended!)
```python
from blockchain_trading import BlockchainIntegration

# Always start with paper trading
integration = BlockchainIntegration(
    network='optimism',
    paper_trading_mode=True  # No real money!
)

# Find opportunities
opportunities = integration.evaluate_blockchain_opportunities(min_edge=0.03)

for opp in opportunities[:5]:
    print(f"{opp['market'].home_team}: {opp['edge']:.1%} edge @ {opp['odd'].decimal_odds}")
```

## 5. Monitoring Dashboard

### Start Monitoring Stack
```bash
# Generate configuration
python monitoring_stack.py

# Start services
cd monitoring
docker-compose up -d

# Access dashboards
# - Grafana: http://localhost:3000 (admin/admin)
# - Prometheus: http://localhost:9091
```

### View Real-time Metrics
```python
from telemetry import get_telemetry

telemetry = get_telemetry()

# Record custom metrics
telemetry.record_bet_placed(amount=100, market="EPL", strategy="value")
telemetry.record_edge_found(edge=0.05, odds=2.1)
```

## 6. Running Full System

### Development Mode
```bash
# Terminal 1: Start web monitor
python web_monitor_v2.py

# Terminal 2: Run trading system
python run_ominari_unified.py

# Terminal 3: Monitor logs
tail -f logs/ominari_trading.log
```

### Production Mode
```bash
# Use systemd services
sudo systemctl start ominari-trading
sudo systemctl start ominari-monitor
sudo systemctl status ominari-trading
```

## Common Workflows

### 1. Morning Checkin
```python
# Check system health
python safe_query.py summary

# Review overnight performance
from ominari_api_client import OminariAPIClient
client = OminariAPIClient()
perf = client.get_performance("main", days=1)
print(f"Last 24h P&L: ${perf['total_pnl']}")
```

### 2. Add New Data Source
```python
from data_source_manager import DataSourceManager

manager = DataSourceManager()
manager.register_source(
    name="my_api",
    fetcher=my_api_fetcher,
    normalizer=my_api_normalizer
)
```

### 3. Backtest New Strategy
```python
# Use the cached vectorized backtest
from cached_vectorized_backtest import CachedVectorizedBacktest

backtest = CachedVectorizedBacktest(
    start_date="2025-01-01",
    end_date="2025-09-01",
    signal_provider="my_new_signal"
)

results = backtest.run()
print(f"Sharpe: {results['sharpe']:.2f}")
print(f"Max Drawdown: {results['max_drawdown']:.1%}")
```

## Troubleshooting

### Database Issues
```bash
# Check connection
python safe_query.py test

# If timeout, use smaller queries
python safe_query.py count Market --limit 1000
```

### Blockchain Connection
```bash
# Test connection
python test_blockchain_connection.py

# If failing, check RPC URL
echo $OPTIMISM_RPC_URL
```

### Performance Issues
```bash
# Check what's running
ps aux | grep python

# Monitor resource usage
htop

# Check disk space
df -h
```

## Next Steps

1. Complete database migration
2. Set up proper RPC endpoints
3. Test all systems in paper mode
4. Deploy monitoring stack
5. Run small live tests
6. Scale up gradually

Remember: Always test with paper trading first!