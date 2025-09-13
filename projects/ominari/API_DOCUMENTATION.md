# Ominari Trading System API

Version: 1.0.0

The Ominari Trading System is a comprehensive sports betting platform with paper trading,
risk management, and multi-signal strategies. This API provides access to trading operations,
market data, performance metrics, and system monitoring.


## Table of Contents

- [System](#system)
- [Trading](#trading)
- [Markets](#markets)
- [Performance](#performance)
- [Dashboard](#dashboard)

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
