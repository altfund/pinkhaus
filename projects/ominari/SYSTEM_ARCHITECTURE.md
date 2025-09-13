# Ominari Trading System Architecture

## Overview

Ominari is a comprehensive sports betting trading system that implements systematic trading strategies adapted from Robert Carver's futures trading methodology. The system features multi-stage alpha research, real-time data collection, paper trading, and production deployment capabilities.

## Core Components

### 1. Data Layer

#### Database (`database.py`, `models.py`)
- SQLite/PostgreSQL with SQLAlchemy ORM
- Optimized for 200GB+ datasets
- WAL mode for concurrency
- Indexed for performance

#### Real-time Data Collection
- **API Integration** (`free_data_pull.py`): REST API data collection
- **GraphQL Client** (`graphql_client.py`): Multi-network GraphQL queries
- **Blockchain Reader** (`blockchain_reader.py`): Direct on-chain data access
- **Quote Collector** (`paper_trading_engine.py`): Real-time quote streaming

### 2. Signal System

#### Signal Framework (`signals.py`)
- Base `SignalProvider` abstract class
- Hot-pluggable signal architecture
- Signal implementations:
  - `ImpliedRawSignal`: Market implied probabilities
  - `ExternalGrpcSignal`: External model integration
  - Custom signals via `signal_registry.py`

#### Dynamic Signal Management (`signal_registry.py`)
- Runtime signal registration
- Multiple weighting strategies:
  - Equal weights
  - Inverse variance
  - Bayesian updating
  - Online learning
  - Regime-based adaptation
- Performance tracking and persistence

### 3. Alpha Research Pipeline (`alpha_research_pipeline.py`)

Six-stage research process:

1. **Raw R&D**: Initial hypothesis testing
   - Min 1,000 samples, 30 days
   - Sharpe > 0.5, p-value < 0.05

2. **In-Sample Testing**: Parameter optimization
   - Min 5,000 samples, 90 days
   - Sharpe > 1.0, p-value < 0.01

3. **Out-of-Sample Testing**: Validation
   - Max 20% performance degradation
   - Min 2,000 samples, 60 days

4. **Walk-Forward Analysis**: Stability testing
   - Rolling window validation
   - 70% consistency requirement

5. **Paper Trading**: Live simulation
   - Real quotes, simulated execution
   - Tracking error < 10%

6. **Live Trading**: Production deployment
   - Risk limits enforced
   - Continuous monitoring

### 4. Systematic Trading Framework (`carver_framework.py`)

Implements Carver's modular approach:

- **Forecast Generation**: Multiple trading rules
- **Forecast Scaling**: Standardized [-20, +20] range
- **Forecast Combination**: Correlation-adjusted weights
- **Position Sizing**: Volatility-targeted positions
- **Risk Management**: Portfolio-level constraints

Key adaptations for sports betting:
- Kelly criterion integration
- Mutual exclusivity constraints
- No short positions (bets only)
- Event-driven vs continuous markets

### 5. Backtesting Engine

#### Vectorized Backtest (`vectorized_backtest.py`)
- Pandas/NumPy based for performance
- Parallel strategy evaluation
- Comprehensive metrics calculation

#### Cached Backtest (`cached_vectorized_backtest.py`)
- Sliding window data cache
- 90% reduction in database queries
- Signal memoization

### 6. Execution Layer

#### Paper Trading (`paper_trading_engine.py`)
- Realistic market simulation
- Slippage and impact modeling
- Quote collection and storage
- Performance tracking

#### Live Trading (`live_trading_overtime.py`)
- Production betting execution
- Real-time position management
- Risk limits enforcement

### 7. Infrastructure

#### Multi-Environment Support (`deployment_config.py`)
- Development: Local SQLite, debug mode
- Testing: Isolated test database
- Staging: Production-like with mainnet
- Production: Full monitoring, HA setup

#### Deployment
- Docker containerization
- Kubernetes manifests
- GitHub Actions CI/CD
- Multi-network blockchain support

## Data Flow

```
1. Data Collection
   ├── REST APIs → Database
   ├── GraphQL → Real-time updates
   └── Blockchain → On-chain events

2. Signal Generation
   ├── Raw signals → Signal Registry
   ├── Weight calculation → Dynamic Weights
   └── Forecast combination → Final predictions

3. Position Sizing
   ├── Volatility calculation
   ├── Kelly optimization
   └── Risk constraints

4. Order Management
   ├── Paper trading simulation
   └── Live execution

5. Performance Analysis
   ├── Real-time monitoring
   ├── Attribution analysis
   └── Research feedback loop
```

## Key Design Patterns

1. **Abstract Factory**: Signal providers
2. **Strategy Pattern**: Multiple forecast methods
3. **Observer Pattern**: Real-time data updates
4. **Repository Pattern**: Data access layer
5. **Pipeline Pattern**: Alpha research stages

## Performance Optimizations

1. **Database**:
   - Composite indexes on (match_id, market_type, timestamp)
   - Partitioning by date
   - Read replicas for analytics

2. **Computation**:
   - Vectorized operations
   - Sliding window caching
   - Signal memoization
   - Parallel processing

3. **Network**:
   - Connection pooling
   - Request batching
   - WebSocket subscriptions
   - CDN for static data

## Monitoring & Observability

1. **Metrics** (Prometheus/Grafana):
   - Trading performance
   - System latency
   - Hit rates
   - Error rates

2. **Logging** (ELK Stack):
   - Structured JSON logs
   - Correlation IDs
   - Audit trail

3. **Tracing** (OpenTelemetry):
   - Distributed tracing
   - Performance profiling
   - Bottleneck identification

## Security Considerations

1. **Secrets Management**:
   - Environment-specific configs
   - AWS Secrets Manager / Vault
   - Encrypted at rest

2. **Access Control**:
   - API key rotation
   - Network segmentation
   - Least privilege principle

3. **Audit**:
   - All trades logged
   - Immutable audit trail
   - Compliance reporting

## Scaling Considerations

1. **Horizontal Scaling**:
   - Stateless services
   - Load balancing
   - Auto-scaling groups

2. **Data Scaling**:
   - Database sharding
   - Time-series optimization
   - Cold storage tiering

3. **Computation Scaling**:
   - Distributed backtesting
   - GPU acceleration
   - Caching layers

## Future Enhancements

1. **Machine Learning Pipeline**:
   - Feature engineering
   - Model training infrastructure
   - A/B testing framework

2. **Advanced Risk Management**:
   - Correlation breakdowns
   - Tail risk hedging
   - Dynamic leverage

3. **Multi-Asset Support**:
   - Cross-sport arbitrage
   - Correlation trading
   - Portfolio optimization