# Ominari Trading System API

Version: 1.0.0

The Ominari Trading System is a comprehensive sports betting platform with paper trading,
risk management, and multi-signal strategies. This API provides access to trading operations,
market data, performance metrics, and system monitoring.


## Table of Contents

- [System](#system)
- [Health & Monitoring](#health--monitoring)
- [Trading](#trading)
- [Markets](#markets)
- [Performance](#performance)
- [Dashboard](#dashboard)
- [WebSocket API](#websocket-api)
- [Rate Limiting](#rate-limiting)
- [Caching](#caching)

## Servers

- **Development server**: `http://localhost:8000`
- **Web monitor server**: `http://localhost:8888`
- **Production server**: `https://api.ominari.com`

## System

System status and health endpoints

### GET /api/status

**Get system status**

Returns current system status including health checks and component states

#### Responses

**200**: System status information

Returns: [`SystemStatus`](#systemstatus)

---

### GET /api/database-stats

**Get database statistics**

Returns statistics about the database including table sizes and record counts

#### Responses

**200**: Database statistics

Returns: [`DatabaseStats`](#databasestats)

---


## Health & Monitoring

Health checks and monitoring endpoints with rate limiting

### GET /health

**Get system health status**

Returns comprehensive health status of all system components including database, session manager, and WebSocket connections.

**Rate Limit**: 30 requests/minute (API limit)

#### Responses

**200**: System is healthy

```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "services": {
    "database": {
      "status": "up",
      "markets": 1500
    },
    "session_manager": {
      "status": "up"
    },
    "websocket": {
      "status": "up",
      "clients": 5
    }
  }
}
```

**503**: System is unhealthy

```json
{
  "status": "unhealthy",
  "timestamp": "2024-01-01T12:00:00.000Z",
  "services": {
    "database": {
      "status": "down"
    },
    "session_manager": {
      "status": "up"
    },
    "websocket": {
      "status": "up",
      "clients": 0
    }
  }
}
```

---

### GET /metrics

**Get Prometheus metrics**

Returns system metrics in Prometheus format for monitoring and alerting.

**Rate Limit**: 30 requests/minute (API limit)

#### Responses

**200**: Metrics in Prometheus format

Content-Type: `text/plain`

```
# HELP ominari_markets_total Total number of markets
# TYPE ominari_markets_total gauge
ominari_markets_total 1500

# HELP ominari_real_odds_total Markets with real odds
# TYPE ominari_real_odds_total gauge
ominari_real_odds_total 850

# HELP ominari_bankroll_current Current bankroll
# TYPE ominari_bankroll_current gauge
ominari_bankroll_current 10000.0
```

---

### GET /cache-stats

**Get cache statistics**

Returns performance statistics for the caching layer.

**Rate Limit**: 30 requests/minute (API limit)

#### Responses

**200**: Cache statistics

```json
{
  "size": 45,
  "hits": 1250,
  "misses": 320,
  "evictions": 15,
  "hit_rate": "79.6%"
}
```

---


## Trading

Trading operations and positions

### GET /api/strategy

**Get current strategy configuration**

Returns the current Kelly strategy parameters and risk settings

#### Responses

**200**: Strategy configuration

Returns: [`StrategyConfig`](#strategyconfig)

---

### GET /api/trading/status

**Get trading system status**

Returns current trading system operational status

#### Responses

**200**: Trading status

Returns: [`TradingStatus`](#tradingstatus)

---

### GET /api/trading/portfolio

**Get portfolio overview**

Returns current portfolio state including positions, exposure, and risk metrics

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| session_id | query | string | Yes | Paper trading session ID |

#### Responses

**200**: Portfolio overview

Returns: [`PortfolioOverview`](#portfoliooverview)

---

### GET /api/trading/positions

**Get open positions**

Returns list of all open trading positions with details

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| session_id | query | string | No | Paper trading session ID |
| include_closed | query | boolean | No | Include closed positions |

#### Responses

**200**: List of positions

---

### POST /api/trading/positions/{position_id}/close

**Close a position**

Manually close an open position

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| position_id | path | string | Yes |  |

#### Request Body

Content-Type: `application/json`

```json
{
  "type": "object",
  "properties": {
    "final_odds": {
      "type": "number",
      "description": "Final odds for settlement"
    }
  }
}
```

#### Responses

**200**: Position closed successfully

Returns: [`PositionCloseResult`](#positioncloseresult)

**404**: Position not found

---

### POST /api/trading/execute

**Execute trading recommendations**

Generate and optionally execute trading recommendations based on current market conditions

#### Request Body

Content-Type: `application/json`

```json
{
  "type": "object",
  "properties": {
    "dry_run": {
      "type": "boolean",
      "default": true,
      "description": "If true, only generate recommendations without executing"
    },
    "session_id": {
      "type": "string",
      "description": "Paper trading session ID"
    }
  }
}
```

#### Responses

**200**: Trading execution result

Returns: [`TradingExecutionResult`](#tradingexecutionresult)

---

### GET /api/trading/recent

**Get recent trading activity**

Returns recent bets, signals, and market activity

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| hours | query | integer | No | Number of hours to look back |

#### Responses

**200**: Recent trading activity

Returns: [`RecentActivity`](#recentactivity)

---

### GET /api/trading/logs

**Get trading logs**

Returns recent trading system logs

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| limit | query | integer | No | Maximum number of log entries |

#### Responses

**200**: Trading logs

---

### GET /api/trading/sessions

**Get paper trading sessions**

Returns list of all paper trading sessions

#### Responses

**200**: List of sessions

---


## Markets

Market data and analysis

### GET /api/trading/evaluation-stats

**Get market evaluation statistics**

Returns statistics about current market opportunities and Kelly recommendations

#### Responses

**200**: Evaluation statistics

Returns: [`EvaluationStats`](#evaluationstats)

---


## Performance

Performance metrics and analytics

### GET /api/performance

**Get performance metrics**

Returns comprehensive performance metrics for the trading system including:
- P&L statistics
- Win rates
- Sharpe ratio
- Drawdown metrics
- Trade statistics by sport and strategy


#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| session_id | query | string | No | Paper trading session ID (optional) |
| days | query | integer | No | Number of days to analyze |

#### Responses

**200**: Performance metrics

Returns: [`PerformanceMetrics`](#performancemetrics)

---


## Dashboard

Dashboard and UI data endpoints

### GET /

**Main dashboard page**

Returns the HTML dashboard interface

#### Responses

**200**: HTML dashboard page

---

### GET /api/dashboard/unified

**Get unified dashboard data**

Returns all data needed for the main dashboard display

#### Parameters

| Name | In | Type | Required | Description |
|------|-----|------|----------|-------------|
| session_id | query | string | No | Paper trading session ID |

#### Responses

**200**: Unified dashboard data

Returns: [`UnifiedDashboard`](#unifieddashboard)

---


## Schemas

### SystemStatus

| Property | Type | Description |
|----------|------|-------------|
| status | string |  (enum: healthy, degraded, error) |
| uptime_hours | number |  |
| database_connected | boolean |  |
| trading_active | boolean |  |
| last_check | string |  |
| version | string |  |

### DatabaseStats

| Property | Type | Description |
|----------|------|-------------|
| total_markets | integer |  |
| total_odds | integer |  |
| total_bets | integer |  |
| active_markets | integer |  |
| database_size_mb | number |  |
| tables | array of object |  |

### StrategyConfig

| Property | Type | Description |
|----------|------|-------------|
| kelly_fraction | number | Kelly criterion fraction (0-1) |
| min_bet | number | Minimum bet size in dollars |
| min_bet_pct | number | Minimum bet as percentage of bankroll |
| bankroll | number | Total bankroll amount |
| cap_per_game | number | Maximum exposure per game as fraction |
| cap_per_bet | number | Maximum single bet as fraction |
| biases | object |  |

### PerformanceMetrics

| Property | Type | Description |
|----------|------|-------------|
| total_pnl | number |  |
| win_rate | number |  |
| total_bets | integer |  |
| winning_bets | integer |  |
| losing_bets | integer |  |
| avg_stake | number |  |
| avg_odds | number |  |
| sharpe_ratio | number |  |
| max_drawdown | number |  |
| current_drawdown | number |  |
| best_day | object |  |
| worst_day | object |  |
| by_sport | object |  |

### TradingStatus

| Property | Type | Description |
|----------|------|-------------|
| active | boolean |  |
| mode | string |  (enum: paper, live, simulation) |
| current_session_id | string |  |
| last_trade_time | string |  |

### PortfolioOverview

| Property | Type | Description |
|----------|------|-------------|
| total_value | number |  |
| cash_balance | number |  |
| positions_value | number |  |
| total_exposure | number |  |
| open_positions | integer |  |
| daily_pnl | number |  |
| daily_pnl_pct | number |  |
| total_pnl | number |  |
| total_pnl_pct | number |  |
| max_drawdown | number |  |
| current_drawdown | number |  |

### Position

| Property | Type | Description |
|----------|------|-------------|
| id | string |  |
| session_id | string |  |
| market_id | string |  |
| sport | string |  |
| league | string |  |
| home_team | string |  |
| away_team | string |  |
| market_type | string |  |
| outcome | string |  |
| odds | number |  |
| amount | number |  |
| potential_payout | number |  |
| created_at | string |  |
| expires_at | string |  |
| status | string |  (enum: pending, open, won, lost, void) |
| pnl | number |  |

### PositionCloseResult

| Property | Type | Description |
|----------|------|-------------|
| position_id | string |  |
| final_odds | number |  |
| result | string |  (enum: won, lost) |
| pnl | number |  |
| closed_at | string |  |

### TradingExecutionResult

| Property | Type | Description |
|----------|------|-------------|
| recommendations | array of object |  |
| executed | array of object |  |
| total_staked | number |  |
| expected_return | number |  |

### RecentActivity

| Property | Type | Description |
|----------|------|-------------|
| recent_bets | array of object |  |
| recent_signals | array of object |  |
| active_markets | integer |  |

### TradingSession

| Property | Type | Description |
|----------|------|-------------|
| id | string |  |
| name | string |  |
| created_at | string |  |
| initial_bankroll | number |  |
| current_bankroll | number |  |
| total_pnl | number |  |
| total_pnl_pct | number |  |
| status | string |  (enum: active, completed, paused) |
| stats | object |  |

### EvaluationStats

| Property | Type | Description |
|----------|------|-------------|
| open_markets | integer |  |
| upcoming_games | integer |  |
| next_game_time | string |  |
| recommended_bets | integer |  |
| total_edge | number |  |
| expected_return | number |  |

### UnifiedDashboard

| Property | Type | Description |
|----------|------|-------------|
| portfolio | [PortfolioOverview](#portfoliooverview) |  |
| positions | array of [Position](#position) |  |
| recent_activity | [RecentActivity](#recentactivity) |  |
| performance_metrics | [PerformanceMetrics](#performancemetrics) |  |
| system_status | [SystemStatus](#systemstatus) |  |
| current_time | string |  |


## WebSocket API

Real-time data streaming via WebSocket connections

### Connection

```javascript
const socket = io('ws://localhost:8888', {
  reconnection: true,
  reconnectionDelay: 1000,
  reconnectionAttempts: 10,
  timeout: 20000
});
```

### Events

#### Client → Server

##### connect
Establish WebSocket connection

**Rate Limit**: 120 events/minute

**Server Response**: Event `connected`
```json
{
  "status": "ok"
}
```

##### request_dashboard_data
Request latest dashboard data

**Rate Limit**: 120 events/minute

**Server Response**: Event `dashboard_update`
```json
{
  "markets": [...],
  "trading_status": {...},
  "stats": {...},
  "odds_distribution": [...]
}
```

#### Server → Client

##### dashboard_update
Dashboard data update with real-time market information

**Payload**:
```json
{
  "markets": [
    {
      "match_id": "abc123",
      "home_team": "Team A",
      "away_team": "Team B",
      "sport": "Soccer",
      "league": "Premier League",
      "maturity_date": "2024-01-15T20:00:00Z",
      "odds": 2.15,
      "position": "home",
      "source": "betfair",
      "blockchain_connected": true
    }
  ],
  "trading_status": {
    "status": "Active",
    "bankroll": 10000
  },
  "stats": {
    "total_markets": 150,
    "real_odds_count": 85,
    "odds_range": "1.05 - 15.00"
  },
  "odds_distribution": [
    {
      "odds": 2.0,
      "count": 25
    }
  ]
}
```

##### error
Error notification

**Payload**:
```json
{
  "error": "Database connection failed",
  "code": "DB_ERROR",
  "timestamp": "2024-01-01T12:00:00Z"
}
```


## Rate Limiting

API rate limiting to prevent abuse and ensure fair usage

### Limits

| Endpoint Type | Rate Limit | Window |
|--------------|------------|--------|
| General Endpoints | 60 requests | per minute |
| API Endpoints (/api/*) | 30 requests | per minute |
| WebSocket Events | 120 events | per minute |
| Health/Metrics | 30 requests | per minute |

### Headers

Rate limit information is included in response headers:

- `X-RateLimit-Limit`: Maximum requests allowed in window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Unix timestamp when limit resets

### Rate Limit Response

**429 Too Many Requests**
```json
{
  "error": "Rate limit exceeded",
  "message": "Maximum 60 requests per minute"
}
```

### Adaptive Rate Limiting

The system implements adaptive rate limiting that adjusts based on system load:
- High load (>80% CPU/memory): Rate reduced to 20% of base rate
- Low load (<30% CPU/memory): Rate increased to 120% of base rate
- Normal load: Gradual return to base rate


## Caching

Performance optimization through intelligent caching

### Cache Strategy

| Data Type | TTL | Cache Key Pattern |
|-----------|-----|-------------------|
| Market Data | 30 seconds | `real_odds_data` |
| Dashboard Data | 30 seconds | `dashboard:{session_id}` |
| Statistics | 30 seconds | `stats:{type}` |
| Health/Metrics | No caching | - |

### Cache Headers

Responses include cache information:

- `X-Cache`: `HIT` or `MISS`
- `X-Cache-TTL`: Remaining TTL in seconds
- `Cache-Control`: Standard HTTP caching directives

### Cache Statistics

Monitor cache performance via `/cache-stats` endpoint:
- Hit rate percentage
- Total hits/misses
- Current cache size
- Eviction count

### Cache Invalidation

Cache is automatically invalidated:
- On data updates
- After TTL expiration
- When cache size limits are reached

### Client-Side Caching

Recommended client-side caching:
```javascript
// Cache dashboard data for 30 seconds
const CACHE_TTL = 30000; // milliseconds
let cachedData = null;
let cacheTimestamp = 0;

function getDashboardData() {
  if (cachedData && Date.now() - cacheTimestamp < CACHE_TTL) {
    return Promise.resolve(cachedData);
  }
  return fetchDashboardData().then(data => {
    cachedData = data;
    cacheTimestamp = Date.now();
    return data;
  });
}
```
