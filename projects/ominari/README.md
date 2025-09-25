# Ominari Blockchain Trading System

A sophisticated sports betting trading system that integrates blockchain data from Optimism and Arbitrum networks with advanced signal generation, paper trading, and real-time monitoring capabilities.

## Overview

Ominari is a comprehensive trading platform that:
- Collects and analyzes sports betting market data from blockchain sources
- Generates trading signals using multiple strategies
- Executes paper trading with realistic market simulation
- Provides real-time monitoring and analytics
- Supports PostgreSQL for data storage and Redis for caching

## Features

- **Multi-chain Support**: Optimism and Arbitrum blockchain integration
- **Signal Generation**: Multiple signal strategies including implied probability, volume-weighted, and blockchain-enhanced signals
- **Paper Trading**: Realistic simulation with commission, slippage, and risk management
- **Web Dashboard**: Real-time monitoring at http://localhost:8888
- **Docker Support**: Full containerization with docker-compose
- **Monitoring Stack**: Prometheus and Grafana integration
- **PostgreSQL Database**: Scalable data storage with connection pooling
- **Redis Caching**: High-performance caching layer

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- PostgreSQL 15+
- Redis (optional, for caching)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ominari
```

2. Install dependencies using uv:
```bash
uv sync
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Start PostgreSQL (if not using Docker):
```bash
./run_with_postgres.py
```

### Running with Docker

The easiest way to run the entire system:

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

This will start:
- PostgreSQL database (port 5432)
- Redis cache (port 6379)
- Ominari trading system (port 8000)
- Blockchain sync daemon
- Paper trading engine
- Web monitor dashboard (port 8888)
- Prometheus monitoring (port 9091)
- Grafana dashboards (port 3000)

### Running Locally

1. Start the web dashboard:
```bash
python web_monitor_unified.py
```

2. Run blockchain sync:
```bash
python blockchain_reader.py --daemon
```

3. Start paper trading:
```bash
python paper_trading_engine.py
```

## Testing

Run all tests:
```bash
python -m pytest tests/ -v
```

Run integration tests:
```bash
python blockchain_integration_tests.py
```

Test results:
- Unit tests: 100% passing (24/24 tests)
- Integration tests: 90% passing (9/10 tests)
- Performance: 7,948 markets/second, 435,184 signals/second

## Architecture

### Core Components

1. **Blockchain Reader** (`blockchain_reader.py`)
   - Connects to Optimism/Arbitrum RPC endpoints
   - Fetches market creation events and trades
   - Stores data in PostgreSQL

2. **Signal System** (`signal_registry.py`, `signals.py`)
   - Modular signal provider architecture
   - Dynamic weight management
   - Signal aggregation and performance tracking

3. **Paper Trading Engine** (`paper_trading_engine.py`)
   - Realistic order execution simulation
   - Commission and slippage modeling
   - Performance metrics and reporting

4. **Web Monitor** (`web_monitor_unified.py`)
   - Real-time dashboard on port 8888
   - Market overview and analytics
   - Paper trading performance tracking

5. **Database** (`database_v2.py`)
   - PostgreSQL with SQLAlchemy ORM
   - Automatic retries and connection pooling
   - Migration support with Alembic

### Data Flow

```
Blockchain ’ Reader ’ PostgreSQL ’ Signal Generation ’ Paper Trading ’ Dashboard
                           “
                         Redis (Cache)
```

## Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/ominari_db

# Blockchain RPC
OPTIMISM_RPC_URL=https://mainnet.optimism.io
ARBITRUM_RPC_URL=https://arb1.arbitrum.io/rpc

# Redis
REDIS_URL=redis://localhost:6379

# Trading
PAPER_TRADING_BANKROLL=10000
MAX_POSITION_SIZE=0.25
MIN_BET_SIZE=1.0
```

### RPC Endpoints

Multiple RPC endpoints are configured for reliability:
- Optimism: Alchemy, Infura, Public endpoints
- Arbitrum: Alchemy, Infura, Public endpoints

See `rpc_config.py` for full configuration.

## API Endpoints

- `GET /health` - Health check
- `GET /api/v1/markets` - List active markets
- `GET /api/v1/signals` - Get current signals
- `GET /api/v1/paper-trades` - Paper trading history
- `GET /metrics` - Prometheus metrics

## Monitoring

### Prometheus Metrics

Available at http://localhost:9091:
- Market counts by sport/status
- Signal performance metrics
- Paper trading P&L
- System performance counters

### Grafana Dashboards

Access at http://localhost:3000 (admin/ominari_admin_2025):
- Trading Overview
- Signal Performance
- System Metrics
- Blockchain Sync Status

## Development

### Code Style

Format code:
```bash
just format
```

Lint code:
```bash
just lint
```

### Adding New Signals

1. Create signal class inheriting from `BaseSignalProvider`
2. Implement required methods: `get_probs()`, `get_parameters()`, `get_required_columns()`
3. Register in signal registry
4. Test with backtesting framework

Example:
```python
class MySignal(BaseSignalProvider):
    def __init__(self):
        super().__init__(name="my_signal", version="1.0.0")
        
    def get_probs(self, df: pd.DataFrame) -> pd.Series:
        # Your signal logic here
        return probabilities
```

### Database Migrations

Create migration:
```bash
alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
alembic upgrade head
```

## Deployment

### Production Deployment

1. Set environment to production:
```bash
export ENVIRONMENT=production
```

2. Use deployment script:
```bash
./deploy.sh production deploy
```

3. Check deployment status:
```bash
./deploy.sh production status
```

### Docker Deployment

Build and deploy with Docker:
```bash
docker-compose -f docker-compose.yml up -d
```

For production:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## Troubleshooting

### Common Issues

1. **Database connection errors**
   - Check PostgreSQL is running
   - Verify DATABASE_URL is correct
   - Check user permissions

2. **RPC connection failures**
   - System automatically retries with different endpoints
   - Check RPC URLs in configuration
   - Monitor logs for specific errors

3. **Redis not available**
   - System works without Redis but performance is reduced
   - Install and start Redis for caching

4. **Port conflicts**
   - Web dashboard: 8888
   - API: 8000
   - PostgreSQL: 5432
   - Redis: 6379
   - Prometheus: 9091
   - Grafana: 3000

### Logs

View logs:
```bash
# Docker logs
docker-compose logs -f

# Local logs
tail -f logs/ominari.log
```

## Security

- Never commit `.env` files
- Use environment variables for sensitive data
- Database connections use SSL in production
- API endpoints have rate limiting
- Blockchain RPC calls are authenticated when using private endpoints

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Run tests: `pytest tests/`
4. Ensure code quality: `just check`
5. Commit changes with clear messages
6. Push and create pull request

## License

[License details]

## Support

For issues and questions:
- GitHub Issues: [repository issues]
- Documentation: See `/docs` directory
- Logs: Check application logs for detailed error messages